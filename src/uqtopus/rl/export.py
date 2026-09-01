"""
Policy Export to ONNX

Bakes the observation normalization, the network and the output transforms into
one self-describing graph, returned with the frozen statistics in an immutable
PolicyArtifact.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import onnx
import torch

from .spec import PolicySpec

logger = logging.getLogger(__name__)

DEFAULT_OPSET = 17

_LOG_STD_MIN = -5.0
_LOG_STD_MAX = 2.0


# Frozen normalization snapshot

@dataclass(frozen=True)
class Normalization:
    """
    Immutable observation normalization statistics.

    The arrays are marked non-writeable on construction, so a snapshot handed to
    the PPO update cannot be mutated by whatever keeps the running estimate.

    Parameters:
        mean (np.ndarray): per-component mean, shape (obs_dim,).
        std (np.ndarray): per-component standard deviation, shape (obs_dim,).
    """

    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        mean = np.array(self.mean, dtype=np.float64, copy=True).ravel()
        std = np.array(self.std, dtype=np.float64, copy=True).ravel()
        if mean.shape != std.shape:
            raise ValueError(
                f"mean and std must have the same shape, got {mean.shape} and {std.shape}"
            )
        if np.any(std <= 0):
            raise ValueError("Normalization.std must be strictly positive")
        mean.setflags(write=False)
        std.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    @classmethod
    def identity(cls, dim: int) -> "Normalization":
        """No-op normalization, for policies trained on raw observations."""
        return cls(mean=np.zeros(dim), std=np.ones(dim))

    @classmethod
    def from_observations(cls, obs: np.ndarray, eps: float = 1e-8) -> "Normalization":
        """Snapshot statistics from a batch of observations, shape (N, obs_dim)."""
        obs = np.asarray(obs, dtype=np.float64)
        if obs.ndim != 2:
            raise ValueError(f"obs must be 2-D (N, obs_dim), got shape {obs.shape}")
        return cls(mean=obs.mean(axis=0), std=obs.std(axis=0) + eps)

    def apply(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observations exactly as the exported graph does."""
        return (np.asarray(obs, dtype=np.float64) - self.mean) / self.std

    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, Sequence[float]]) -> "Normalization":
        return cls(mean=np.asarray(data["mean"]), std=np.asarray(data["std"]))


class RunningStatistics:
    """
    Mutable accumulator of observation statistics, producing frozen snapshots.

    Update order, per iteration: snapshot -> export -> run -> fill the buffer
    with the snapshot -> feed the new observations back in.
    """

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        self.dim = int(dim)
        self.eps = float(eps)
        self._count = 0
        self._mean = np.zeros(self.dim, dtype=np.float64)
        self._m2 = np.zeros(self.dim, dtype=np.float64)

    @property
    def count(self) -> int:
        return self._count

    def update(self, observations: np.ndarray) -> None:
        """Fold a batch of observations, shape (n, dim), into the estimate."""
        batch = np.atleast_2d(np.asarray(observations, dtype=np.float64))
        if batch.shape[1] != self.dim:
            raise ValueError(
                f"observations have {batch.shape[1]} components, expected {self.dim}"
            )
        n = batch.shape[0]
        if n == 0:
            return

        batch_mean = batch.mean(axis=0)
        batch_m2 = ((batch - batch_mean) ** 2).sum(axis=0)

        delta = batch_mean - self._mean
        total = self._count + n
        self._mean += delta * n / total
        self._m2 += batch_m2 + delta**2 * self._count * n / total
        self._count = total

    def snapshot(self) -> Normalization:
        """
        Freeze the current estimate.

        Before any observation has been seen, this is the identity, so the first
        iteration runs on raw observations rather than on a guess.
        """
        if self._count < 2:
            return Normalization.identity(self.dim)
        std = np.sqrt(self._m2 / self._count) + self.eps
        return Normalization(mean=self._mean, std=std)

    def __repr__(self) -> str:
        return f"RunningStatistics(dim={self.dim}, count={self._count})"


# Export artifact

@dataclass(frozen=True)
class PolicyArtifact:
    """
    Immutable handle to one exported policy file.

    This is what the rollout runner ships to the solver and what the PPO update
    reads its normalization from. Carrying the statistics here, instead of in a
    mutable object shared with the trainer, is what prevents the update from
    normalizing with statistics the graph never saw.

    Parameters:
        path (Path): the .onnx file.
        spec (PolicySpec): the contract the graph implements.
        normalization (Normalization): statistics baked into the graph.
        iteration (int or None): training iteration that produced this file.
        opset (int): ONNX opset version.
        file_sha256 (str): checksum of the .onnx file as written.
        created_at (str): UTC ISO-8601 timestamp.
    """

    path: Path
    spec: PolicySpec
    normalization: Normalization
    iteration: int | None
    opset: int
    file_sha256: str
    created_at: str

    @property
    def spec_hash(self) -> str:
        return self.spec.hash

    @property
    def manifest_path(self) -> Path:
        return self.path.with_suffix(".manifest.json")

    def manifest(self) -> dict[str, Any]:
        return {
            "path": self.path.name,
            "spec_hash": self.spec_hash,
            "spec": self.spec.to_dict(),
            "normalization": self.normalization.to_dict(),
            "iteration": self.iteration,
            "opset": self.opset,
            "file_sha256": self.file_sha256,
            "created_at": self.created_at,
        }

    def write_manifest(self) -> Path:
        """Write the sidecar manifest next to the .onnx file."""
        self.manifest_path.write_text(json.dumps(self.manifest(), indent=2))
        return self.manifest_path

    @classmethod
    def load(cls, manifest_path: str | Path) -> "PolicyArtifact":
        """Reload an artifact from its sidecar manifest."""
        manifest_path = Path(manifest_path)
        data = json.loads(manifest_path.read_text())
        return cls(
            path=manifest_path.parent / data["path"],
            spec=PolicySpec.from_dict(data["spec"]),
            normalization=Normalization.from_dict(data["normalization"]),
            iteration=data.get("iteration"),
            opset=int(data["opset"]),
            file_sha256=data["file_sha256"],
            created_at=data["created_at"],
        )

    def verify_file(self) -> bool:
        """Check the .onnx file on disk still matches the recorded checksum."""
        return _sha256(self.path) == self.file_sha256

    def __repr__(self) -> str:
        return (
            f"PolicyArtifact(path={self.path.name!r}, iter={self.iteration}, "
            f"spec_hash={self.spec_hash}, opset={self.opset})"
        )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _module_device(module: Any) -> torch.device:
    """Device the module's parameters live on, CPU when it has none."""
    for parameter in module.parameters():
        return parameter.device
    return torch.device("cpu")


# Graph wrapper (torch backend)

class _PolicyGraph(torch.nn.Module):
    """
    Self-contained graph: normalization, the actor, and the output transform.

    Takes (observation, noise) and returns the action with a Gaussian action,
    or takes (observation) and returns the two shape parameters with a Beta one.

    Parameters:
        net (Any): the actor module.
        spec (PolicySpec): the contract the graph implements.
        normalization (Normalization): statistics baked into the graph.
    """

    def __init__(self, net: Any, spec: PolicySpec, normalization: Normalization) -> None:
        super().__init__()
        self.net = net
        self.act_dim = spec.act_dim
        self.distribution = spec.action.distribution
        device = _module_device(net)
        for name, values in (
            ("obs_mean", normalization.mean),
            ("obs_std", normalization.std),
        ):
            self.register_buffer(
                name,
                torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0),
            )

    def forward(self, observation, noise=None):
        x = (observation - self.obs_mean) / self.obs_std
        head = self.net(x)
        if isinstance(head, (tuple, list)):
            if len(head) != 2:
                raise ValueError(
                    "A policy network returning a tuple must return exactly "
                    f"two tensors, got {len(head)}"
                )
            first, second = head
        else:
            first = head[:, : self.act_dim]
            second = head[:, self.act_dim :]

        if self.distribution == "beta":
            # softplus(x) + 1 keeps both parameters > 1, so the density is
            # unimodal and the mode is well defined for deterministic runs.
            alpha = torch.nn.functional.softplus(first) + 1.0
            beta = torch.nn.functional.softplus(second) + 1.0
            return alpha, beta

        log_std = torch.clamp(second, _LOG_STD_MIN, _LOG_STD_MAX)
        return first + torch.exp(log_std) * noise


class _SB3Actor(torch.nn.Module):
    """
    The actor half of a stable-baselines3 ActorCriticPolicy.

    Parameters:
        policy (Any): the ActorCriticPolicy to read.
    """

    def __init__(self, policy: Any) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, x):
        features = self.policy.extract_features(x)
        if isinstance(features, tuple):  # shared/separate extractors
            features = features[0]
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        mean = self.policy.action_net(latent_pi)
        log_std = self.policy.log_std.expand_as(mean)
        return mean, log_std


def _sb3_actor(model, spec: PolicySpec):
    """Extract the actor of a stable-baselines3 on-policy model as a Module."""
    policy = getattr(model, "policy", model)
    for attr in ("extract_features", "mlp_extractor", "action_net"):
        if not hasattr(policy, attr):
            raise TypeError(
                "The 'sb3' backend expects a stable-baselines3 ActorCriticPolicy "
                f"(or a model owning one); the object is missing '{attr}'."
            )
    if spec.action.distribution != "gaussian":
        raise ValueError(
            "The stable-baselines3 backend exports a diagonal Gaussian policy, "
            f"but the spec declares distribution={spec.action.distribution!r}. "
            "Use distribution='gaussian', or export from the UQTOPUS PPO for Beta."
        )

    return _SB3Actor(policy).eval()


def _looks_like_sb3(obj: Any) -> bool:
    policy = getattr(obj, "policy", None)
    return policy is not None and hasattr(policy, "mlp_extractor")


def _torch_onnx_export(
    graph: Any,
    dummy: Any,
    path: Path,
    spec: PolicySpec,
    output_names: list[str],
    opset: int,
) -> None:
    """
    Run torch.onnx.export across PyTorch versions.
    """
    input_names = list(spec.input_names)
    kwargs: dict[str, Any] = {
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": {
            **{name: {0: "batch"} for name in input_names},
            **{name: {0: "batch"} for name in output_names},
        },
        "opset_version": opset,
        "do_constant_folding": True,
    }

    has_dynamo = "dynamo" in inspect.signature(torch.onnx.export).parameters
    attempts: list[dict[str, Any]] = []
    if has_dynamo:
        attempts.append({**kwargs, "dynamo": False})
        attempts.append({**kwargs, "dynamo": True})
    else:
        attempts.append(kwargs)

    failures: list[tuple[Any, Exception]] = []
    for attempt in attempts:
        try:
            with torch.no_grad(), warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                torch.onnx.export(graph, dummy, str(path), **attempt)
            return
        except Exception as exc:
            failures.append((attempt.get("dynamo"), exc))
            logger.debug(
                "torch.onnx.export failed with dynamo=%s: %s",
                attempt.get("dynamo"),
                exc,
            )

    report = "; ".join(
        f"dynamo={flag}: {type(exc).__name__}: {exc}" for flag, exc in failures
    )
    raise RuntimeError(
        "torch.onnx.export failed with every available exporter. The dynamo one "
        "needs 'onnxscript' (pip install onnxscript). Failures: " + report
    ) from failures[-1][1]


def _stamp_metadata(path: Path, spec: PolicySpec, extra: dict[str, str]) -> None:
    """Replace the ONNX metadata_props with the UQTOPUS contract entries."""
    model = onnx.load(str(path))
    del model.metadata_props[:]
    metadata = spec.to_metadata()
    metadata.update(extra)
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, str(path))


def read_metadata(path: str | Path) -> dict[str, str]:
    """Read the metadata_props of an ONNX file as a plain dict."""
    model = onnx.load(str(path))
    return {entry.key: entry.value for entry in model.metadata_props}


# Public API
def export_policy(
    net: Any,
    spec: PolicySpec,
    path: str | Path,
    *,
    normalization: Normalization | None = None,
    iteration: int | None = None,
    backend: str = "auto",
    opset: int = DEFAULT_OPSET,
    write_manifest: bool = True,
) -> PolicyArtifact:
    """
    Export an actor to a self-describing ONNX policy file.

    Parameters:
        net (Any): the actor. With backend 'torch', a torch.nn.Module mapping
            (batch, obs_dim) to either a single (batch, 2 * act_dim) tensor or a
            pair of (batch, act_dim) tensors, holding the *raw* head outputs
            (before the softplus / clamp transforms). With backend 'sb3', a
            stable-baselines3 model or ActorCriticPolicy.
        spec (PolicySpec): the contract the graph must implement.
        path (str or Path): destination .onnx file.
        normalization (Normalization or None): statistics to bake into the graph.
            None means identity. Whatever is passed here is frozen into the
            returned artifact and is the only thing the PPO update may normalize
            with for this batch.
        iteration (int or None): training iteration, recorded in the manifest.
        backend ('auto', 'torch' or 'sb3'): 'auto' detects stable-baselines3
            models and falls back to 'torch'.
        opset (int): ONNX opset version.
        write_manifest (bool): also write the sidecar .manifest.json.

    Returns:
        PolicyArtifact: immutable handle to the exported file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if normalization is None:
        normalization = Normalization.identity(spec.obs_dim)
    if normalization.dim != spec.obs_dim:
        raise ValueError(
            f"normalization has dimension {normalization.dim} but the spec "
            f"declares obs_dim={spec.obs_dim}"
        )

    if backend == "auto":
        backend = "sb3" if _looks_like_sb3(net) else "torch"

    if backend == "sb3":
        actor = _sb3_actor(net, spec)
    elif backend == "torch":
        actor = net
    else:
        raise ValueError(f"Unknown backend {backend!r}; use 'auto', 'torch' or 'sb3'")

    graph = _PolicyGraph(actor, spec, normalization).eval()

    device = _module_device(graph)
    dummy: tuple[Any, ...] = (
        torch.zeros(1, spec.obs_dim, dtype=torch.float32, device=device),
    )
    if spec.noise_dim:
        dummy += (torch.zeros(1, spec.noise_dim, dtype=torch.float32, device=device),)
    output_names = list(spec.output_names)

    _torch_onnx_export(graph, dummy, path, spec, output_names, opset)

    _stamp_metadata(
        path,
        spec,
        extra={
            "uqtopus.backend": backend,
            "uqtopus.opset": str(opset),
            "uqtopus.iteration": "" if iteration is None else str(iteration),
            "uqtopus.normalization": json.dumps(normalization.to_dict()),
        },
    )

    artifact = PolicyArtifact(
        path=path,
        spec=spec,
        normalization=normalization,
        iteration=iteration,
        opset=opset,
        file_sha256=_sha256(path),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    if write_manifest:
        artifact.write_manifest()

    logger.info("Exported policy to %s (%s)", path, artifact.spec_hash)
    return artifact


def build_mlp(spec: PolicySpec, hidden: Sequence[int] = (64, 64), seed: int | None = None):
    """
    Build an untrained MLP actor matching a spec.

    Maps (batch, obs_dim) to (batch, 2 * act_dim) raw head outputs, which is the
    contract expected by export_policy's 'torch' backend.
    """
    if seed is not None:
        torch.manual_seed(seed)

    layers: list[Any] = []
    prev = spec.obs_dim
    for width in hidden:
        layers += [torch.nn.Linear(prev, width), torch.nn.Tanh()]
        prev = width
    layers.append(torch.nn.Linear(prev, 2 * spec.act_dim))
    return torch.nn.Sequential(*layers)


def export_random_policy(
    spec: PolicySpec,
    path: str | Path,
    *,
    hidden: Sequence[int] = (64, 64),
    seed: int | None = 0,
    **kwargs: Any,
) -> PolicyArtifact:
    """
    Export an untrained policy with random weights.

    This is the artifact used to validate the whole pipeline end to end before
    any training exists, and later to smoke-test the solver-side integration.
    """
    net = build_mlp(spec, hidden=hidden, seed=seed)
    return export_policy(net, spec, path, **kwargs)


class _TorchReference:
    """
    Numpy callable over the torch graph, matching the exported ONNX signature.

    Parameters:
        graph (Any): the exported graph.
        noise_dim (int): number of noise components the graph expects, 0 for none.
    """

    def __init__(self, graph: Any, noise_dim: int) -> None:
        self.graph = graph
        self.noise_dim = noise_dim

    def __call__(
        self, obs: np.ndarray, noise: np.ndarray | None = None
    ) -> tuple[np.ndarray, ...]:
        device = _module_device(self.graph)
        args = [torch.tensor(np.asarray(obs, dtype=np.float32), device=device)]
        if self.noise_dim:
            if noise is None:
                raise ValueError(
                    "this spec declares a Gaussian action, so the reference "
                    "needs the same noise the graph was given"
                )
            args.append(torch.tensor(np.asarray(noise, dtype=np.float32), device=device))
        with torch.no_grad():
            out = self.graph(*args)
        if isinstance(out, (tuple, list)):
            return tuple(item.cpu().numpy() for item in out)
        return (out.cpu().numpy(),)


def torch_reference(net: Any, spec: PolicySpec, normalization: Normalization) -> Callable:
    """
    Build a numpy callable reproducing the exported graph from the torch module.

    Parameters:
        net (Any): the actor module.
        spec (PolicySpec): the contract the graph implements.
        normalization (Normalization): statistics baked into the graph.

    Returns:
        Callable: takes (observation, noise) and returns the graph outputs as
        numpy arrays.
    """
    graph = _PolicyGraph(net, spec, normalization).eval()
    return _TorchReference(graph, spec.noise_dim)
