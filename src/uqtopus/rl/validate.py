"""
Policy Validation

Checks an exported policy against its contract before it reaches a solver: file,
metadata, spec hash, graph signature, forward pass, output bounds and torch parity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import onnx
import onnxruntime as ort

from .export import Normalization, PolicyArtifact, torch_reference
from .spec import PolicySpec

logger = logging.getLogger(__name__)

_LEVELS = ("ok", "warning", "error")


@dataclass
class Check:
    """One validation result."""

    name: str
    level: str
    message: str

    @property
    def ok(self) -> bool:
        return self.level == "ok"

    def __str__(self) -> str:
        mark = {"ok": "PASS", "warning": "WARN", "error": "FAIL"}[self.level]
        return f"[{mark}] {self.name}: {self.message}"


@dataclass
class ValidationReport:
    """Collected results of validate_policy()."""

    path: Path
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, level: str, message: str) -> None:
        if level not in _LEVELS:
            raise ValueError(f"Unknown level {level!r}")
        self.checks.append(Check(name, level, message))

    @property
    def errors(self) -> list[Check]:
        return [c for c in self.checks if c.level == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_status(self) -> None:
        if self.errors:
            details = "\n".join(f"  - {c.name}: {c.message}" for c in self.errors)
            raise ValueError(f"Policy validation failed for {self.path}:\n{details}")

    def __str__(self) -> str:
        status = "OK" if self.ok else f"FAILED ({len(self.errors)} error(s))"
        return "\n".join(
            [f"Policy validation: {self.path}", f"Status: {status}", ""]
            + [str(c) for c in self.checks]
        )

    def __repr__(self) -> str:
        return f"ValidationReport(ok={self.ok}, checks={len(self.checks)})"


def validate_policy(
    policy: str | Path | PolicyArtifact,
    spec: PolicySpec | None = None,
    *,
    reference: Callable[[np.ndarray], Sequence[np.ndarray]] | None = None,
    n_samples: int = 32,
    obs_scale: float = 10.0,
    tolerance: float = 1e-5,
    strict: bool = True,
    seed: int = 0,
) -> ValidationReport:
    """
    Validate an exported ONNX policy against its contract.

    Parameters:
        policy (str, Path or PolicyArtifact): the .onnx file, or the artifact
            returned by export_policy().
        spec (PolicySpec or None): the contract to check against. None means the
            spec embedded in the ONNX metadata is used, which checks internal
            consistency but not agreement with what the caller expects.
        reference (callable or None): maps an observation batch of shape
            (n, obs_dim) to the distribution parameters the graph should produce.
            Build one from a torch module with export.torch_reference().
        n_samples (int): batch size used for the numerical checks.
        obs_scale (float): observations are drawn from N(0, obs_scale).
        tolerance (float): absolute tolerance of the parity check.
        strict (bool): raise on error instead of only reporting.
        seed (int): RNG seed for the sampled observations.

    Returns:
        ValidationReport
    """
    artifact = policy if isinstance(policy, PolicyArtifact) else None
    path = Path(artifact.path if artifact else policy)
    report = ValidationReport(path=path)

    def bail() -> ValidationReport:
        if strict:
            report.raise_for_status()
        return report

    # --- the file is a readable ONNX model -------------------------------
    if not path.exists():
        report.add("file", "error", f"{path} does not exist")
        return bail()

    try:
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
    except Exception as exc:
        report.add("file", "error", f"not a valid ONNX model: {exc}")
        return bail()
    report.add("file", "ok", f"valid ONNX model, {path.stat().st_size} bytes")

    # --- it declares which contract it implements ------------------------
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    embedded: PolicySpec | None = None
    if "uqtopus.spec" in metadata:
        try:
            embedded = PolicySpec.from_metadata(metadata)
            report.add("metadata", "ok", f"spec_hash={embedded.hash}")
        except Exception as exc:
            report.add("metadata", "error", f"unreadable spec: {exc}")
    else:
        report.add(
            "metadata",
            "error",
            "no 'uqtopus.spec' entry; not produced by export_policy()",
        )

    if spec is not None and embedded is not None and spec.hash != embedded.hash:
        report.add(
            "contract",
            "error",
            f"the file implements contract {embedded.hash} but the caller expects "
            f"{spec.hash}; policy, case dictionary and parser disagree",
        )
    elif spec is not None and embedded is not None:
        report.add("contract", "ok", spec.hash)

    effective = spec or embedded
    if effective is None:
        report.add("contract", "error", "no spec available to validate against")
        return bail()

    # --- the signature matches the contract ------------------------------
    initializers = {init.name for init in model.graph.initializer}
    inputs = [i for i in model.graph.input if i.name not in initializers]

    if len(inputs) != 1 or inputs[0].name != effective.input_name:
        report.add(
            "signature.input",
            "error",
            f"expected a single input named {effective.input_name!r}, "
            f"found {[i.name for i in inputs]}",
        )
    else:
        tensor = inputs[0].type.tensor_type
        problems = []
        if tensor.elem_type != onnx.TensorProto.FLOAT:
            problems.append(
                f"dtype is {onnx.TensorProto.DataType.Name(tensor.elem_type)}, expected float32"
            )
        dims = tensor.shape.dim
        if len(dims) != 2:
            problems.append(f"rank is {len(dims)}, expected 2 (batch, obs_dim)")
        else:
            if not dims[0].dim_param:
                problems.append(
                    f"the batch axis is fixed at {dims[0].dim_value}; the solver "
                    "evaluates one observation and the update evaluates batches"
                )
            if dims[1].dim_value and dims[1].dim_value != effective.obs_dim:
                problems.append(
                    f"takes {dims[1].dim_value} components, spec declares "
                    f"obs_dim={effective.obs_dim}"
                )
        if problems:
            report.add("signature.input", "error", "; ".join(problems))
        else:
            report.add(
                "signature.input", "ok", f"float32 (batch, {effective.obs_dim})"
            )

    expected = list(effective.output_names)
    actual = [o.name for o in model.graph.output]
    if actual != expected:
        report.add(
            "signature.output",
            "error",
            f"outputs are {actual}, expected {expected} for distribution "
            f"{effective.action.distribution!r}",
        )
    else:
        report.add("signature.output", "ok", ", ".join(actual))

    # --- it loads and runs -----------------------------------------------
    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
    except Exception as exc:
        report.add("runtime", "error", f"onnxruntime failed to load the graph: {exc}")
        return bail()

    rng = np.random.default_rng(seed)
    obs = rng.normal(0.0, obs_scale, (n_samples, effective.obs_dim)).astype(np.float32)
    try:
        outputs = session.run(None, {input_name: obs})
    except Exception as exc:
        report.add("runtime", "error", f"forward pass failed: {exc}")
        return bail()
    report.add("runtime", "ok", f"batch of {n_samples} evaluated")

    # --- the outputs are usable -------------------------------------------
    problems = []
    for name, array in zip(actual, outputs):
        if array.shape != (n_samples, effective.act_dim):
            problems.append(
                f"{name} has shape {array.shape}, expected "
                f"{(n_samples, effective.act_dim)}"
            )
        if not np.all(np.isfinite(array)):
            problems.append(f"{name} has non-finite values")
    if problems:
        report.add("outputs", "error", "; ".join(problems))
    else:
        report.add(
            "outputs",
            "ok",
            ", ".join(f"{n} in [{a.min():.3g}, {a.max():.3g}]" for n, a in zip(actual, outputs)),
        )

    if effective.action.distribution == "beta" and len(outputs) == 2:
        alpha, beta = outputs
        if np.any(alpha <= 0) or np.any(beta <= 0):
            report.add(
                "distribution",
                "error",
                "Beta parameters must be strictly positive; the solver would "
                "draw NaN actions",
            )
        elif np.any(alpha <= 1) or np.any(beta <= 1):
            report.add(
                "distribution",
                "warning",
                "some Beta parameters are <= 1, so the mode used for "
                "deterministic runs is not well defined",
            )
        else:
            report.add("distribution", "ok", "Beta parameters are all > 1")

    # --- the graph computes what the source network computes ---------------
    if reference is None:
        report.add(
            "parity",
            "warning",
            "no reference supplied; the export was not checked against the "
            "source network",
        )
    else:
        try:
            ref_outputs = reference(obs)
        except Exception as exc:
            report.add("parity", "error", f"the reference raised: {exc}")
        else:
            worst = max(
                float(np.max(np.abs(got - np.asarray(want))))
                for got, want in zip(outputs, ref_outputs)
            )
            if worst > tolerance:
                report.add(
                    "parity",
                    "error",
                    f"exported graph and source network disagree by {worst:.3g} "
                    f"(tolerance {tolerance:g})",
                )
            else:
                report.add("parity", "ok", f"max abs difference {worst:.3g}")

    if strict:
        report.raise_for_status()
    return report


def validate_export(
    net: Any,
    spec: PolicySpec,
    path: str | Path,
    *,
    normalization: Normalization | None = None,
    **kwargs: Any,
) -> ValidationReport:
    """
    Validate an exported torch policy, building the parity reference from the
    source network.

    Parameters:
        net (Any): the same actor passed to export_policy().
        spec (PolicySpec): the same spec passed to export_policy().
        path (str or Path): the exported .onnx file.
        normalization (Normalization or None): the same statistics used at
            export. Passing a different snapshot here is exactly the mistake the
            parity check exists to catch.
    """
    if normalization is None:
        normalization = Normalization.identity(spec.obs_dim)
    reference = torch_reference(net, spec, normalization)
    return validate_policy(path, spec, reference=reference, **kwargs)
