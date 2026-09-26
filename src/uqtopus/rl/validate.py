"""
Policy Validation

Checks an exported policy against its contract before it reaches a solver: the
contract its metadata declares, and a forward pass in ONNX Runtime.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from .export import PolicyArtifact, read_metadata
from .spec import PolicySpec


def validate_policy(
    policy: str | Path | PolicyArtifact,
    spec: PolicySpec | None = None,
    *,
    n_samples: int = 32,
    seed: int = 0,
) -> None:
    """
    Raise ValueError when an exported ONNX policy does not implement its contract.

    Parameters:
        policy (str, Path or PolicyArtifact): the .onnx file, or the artifact
            returned by export_policy().
        spec (PolicySpec or None): the contract to check against. None uses the
            spec embedded in the file.
        n_samples (int): batch size of the forward pass.
        seed (int): RNG seed for the sampled observations and noise.
    """
    path = Path(policy.path if isinstance(policy, PolicyArtifact) else policy)
    embedded = PolicySpec.from_metadata(read_metadata(path))
    if spec is None:
        spec = embedded
    if embedded.hash != spec.hash:
        raise ValueError(
            f"{path} implements contract {embedded.hash}, but {spec.hash} is expected"
        )

    rng = np.random.default_rng(seed)
    obs = rng.normal(0.0, 10.0, (n_samples, spec.obs_dim)).astype(np.float32)
    noise = rng.standard_normal((n_samples, spec.noise_dim)).astype(np.float32)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (action,) = session.run(None, {"observation": obs, "noise": noise})

    if action.shape != (n_samples, spec.act_dim):
        raise ValueError(
            f"{path} returns actions of shape {action.shape}, "
            f"expected {(n_samples, spec.act_dim)}"
        )
    if not np.all(np.isfinite(action)):
        raise ValueError(f"{path} returns non-finite actions")
