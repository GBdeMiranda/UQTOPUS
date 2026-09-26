"""
Policy Export to ONNX

Bakes the observation normalization, the network and the output transforms into
one self-describing graph, returned with the frozen statistics in an immutable
PolicyArtifact.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnx
import torch

from .spec import PolicySpec

logger = logging.getLogger(__name__)

_LOG_STD_MIN = -5.0
_LOG_STD_MAX = 2.0


@dataclass(frozen=True)
class Normalization:
    """
    Observation normalization statistics.

    Parameters:
        mean (np.ndarray): per-component mean, shape (obs_dim,).
        std (np.ndarray): per-component standard deviation, shape (obs_dim,).
    """

    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", np.array(self.mean, dtype=np.float64))
        object.__setattr__(self, "std", np.array(self.std, dtype=np.float64))

    @classmethod
    def identity(cls, dim: int) -> "Normalization":
        """No-op normalization, for policies trained on raw observations."""
        return cls(mean=np.zeros(dim), std=np.ones(dim))

    def apply(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observations exactly as the exported graph does."""
        return (np.asarray(obs, dtype=np.float64) - self.mean) / self.std

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, Sequence[float]]) -> "Normalization":
        return cls(mean=data["mean"], std=data["std"])


class RunningStatistics:
    """
    Running mean and variance of the observations seen so far.

    Parameters:
        dim (int): number of observation components.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.count = 0
        self.mean = np.zeros(dim)
        self.m2 = np.zeros(dim)

    def update(self, observations: np.ndarray) -> None:
        """Fold a batch of observations, shape (n, dim), into the estimate."""
        n = len(observations)
        batch_mean = observations.mean(axis=0)
        batch_m2 = ((observations - batch_mean) ** 2).sum(axis=0)

        delta = batch_mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        self.m2 += batch_m2 + delta**2 * self.count * n / total
        self.count = total

    def snapshot(self) -> Normalization:
        """Freeze the current estimate. The identity until two observations are in."""
        if self.count < 2:
            return Normalization.identity(self.dim)
        return Normalization(mean=self.mean, std=np.sqrt(self.m2 / self.count) + 1e-8)


@dataclass(frozen=True)
class PolicyArtifact:
    """
    One exported .onnx file, with the contract and the statistics baked into it.

    Parameters:
        path (Path): the .onnx file.
        spec (PolicySpec): the contract the graph implements.
        normalization (Normalization): statistics baked into the graph.
        iteration (int or None): training iteration that produced this file.
    """

    path: Path
    spec: PolicySpec
    normalization: Normalization
    iteration: int | None

    @classmethod
    def load(cls, path: str | Path) -> "PolicyArtifact":
        """Rebuild the artifact from the metadata of an exported .onnx file."""
        metadata = read_metadata(path)
        iteration = metadata["uqtopus.iteration"]
        return cls(
            path=Path(path),
            spec=PolicySpec.from_metadata(metadata),
            normalization=Normalization.from_dict(
                json.loads(metadata["uqtopus.normalization"])
            ),
            iteration=int(iteration) if iteration else None,
        )


def _module_device(module: Any) -> torch.device:
    """Device the module's parameters live on, CPU when it has none."""
    for parameter in module.parameters():
        return parameter.device
    return torch.device("cpu")


class _PolicyGraph(torch.nn.Module):
    """
    Normalization, actor and Gaussian draw as one module, mapping
    (observation, noise) to the action.

    Parameters:
        net (torch.nn.Module): the actor.
        spec (PolicySpec): the contract the graph implements.
        normalization (Normalization): statistics baked into the graph.
    """

    def __init__(self, net: Any, spec: PolicySpec, normalization: Normalization) -> None:
        super().__init__()
        self.net = net
        self.act_dim = spec.act_dim
        device = _module_device(net)
        self.register_buffer(
            "obs_mean", torch.tensor(normalization.mean, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "obs_std", torch.tensor(normalization.std, dtype=torch.float32, device=device)
        )

    def forward(self, observation, noise):
        head = self.net((observation - self.obs_mean) / self.obs_std)
        if isinstance(head, tuple):
            mean, log_std = head
        else:
            mean, log_std = head[:, : self.act_dim], head[:, self.act_dim :]
        log_std = torch.clamp(log_std, _LOG_STD_MIN, _LOG_STD_MAX)
        return mean + torch.exp(log_std) * noise


def read_metadata(path: str | Path) -> dict[str, str]:
    """Read the metadata_props of an ONNX file as a plain dict."""
    model = onnx.load(str(path))
    return {entry.key: entry.value for entry in model.metadata_props}


def export_policy(
    net: Any,
    spec: PolicySpec,
    path: str | Path,
    *,
    normalization: Normalization | None = None,
    iteration: int | None = None,
) -> PolicyArtifact:
    """
    Export an actor to a self-describing ONNX policy file.

    Parameters:
        net (torch.nn.Module): maps (batch, obs_dim) to the mean and the log_std
            before the clamp, either as a pair of (batch, act_dim) tensors or as
            one (batch, 2 * act_dim) tensor.
        spec (PolicySpec): the contract the graph must implement.
        path (str or Path): destination .onnx file.
        normalization (Normalization or None): statistics to bake into the graph.
            None means identity.
        iteration (int or None): training iteration, recorded in the metadata.

    Returns:
        PolicyArtifact
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if normalization is None:
        normalization = Normalization.identity(spec.obs_dim)

    graph = _PolicyGraph(net, spec, normalization).eval()
    device = _module_device(graph)
    dummy = (
        torch.zeros(1, spec.obs_dim, device=device),
        torch.zeros(1, spec.noise_dim, device=device),
    )

    with torch.no_grad():
        shape = tuple(graph(*dummy).shape)
    if shape != (1, spec.act_dim):
        raise ValueError(
            f"the graph built from this actor returns an action of shape {shape} "
            f"for one observation, and the spec declares (1, {spec.act_dim})"
        )

    names = spec.input_names + spec.output_names
    options = {
        "input_names": list(spec.input_names),
        "output_names": list(spec.output_names),
        "dynamic_axes": {name: {0: "batch"} for name in names},
        "opset_version": 17,
        "verbose": False,
    }
    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            torch.onnx.export(graph, dummy, str(path), dynamo=False, **options)
        except Exception:
            torch.onnx.export(graph, dummy, str(path), dynamo=True, **options)

    metadata = spec.to_metadata()
    metadata["uqtopus.iteration"] = "" if iteration is None else str(iteration)
    metadata["uqtopus.normalization"] = json.dumps(normalization.to_dict())
    model = onnx.load(str(path))
    onnx.helper.set_model_props(model, metadata)
    onnx.save(model, str(path))

    logger.info("Exported policy to %s (%s)", path, spec.hash)
    return PolicyArtifact(path, spec, normalization, iteration)


def build_mlp(spec: PolicySpec, hidden: Sequence[int] = (64, 64), seed: int | None = None):
    """
    Build an untrained MLP actor matching a spec.

    Maps (batch, obs_dim) to (batch, 2 * act_dim) raw head outputs, which is the
    contract expected by export_policy.
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
    """Export an untrained build_mlp() actor. kwargs go to export_policy()."""
    net = build_mlp(spec, hidden=hidden, seed=seed)
    return export_policy(net, spec, path, **kwargs)
