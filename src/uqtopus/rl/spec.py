"""
Policy Specification for Intrusive (Closed-Loop) RL

Single source of truth for the Python <-> solver contract, shared by the controller
dictionary, the ONNX graph signature and the trajectory parser.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Mapping, Sequence

Vec3 = tuple[float, float, float]

_SOURCE_REGISTRY: dict[str, type["ObservationSource"]] = {}


def _register(cls: type["ObservationSource"]) -> type["ObservationSource"]:
    _SOURCE_REGISTRY[cls.kind] = cls
    return cls


# Observation sources

@dataclass(frozen=True)
class ObservationSource:
    """
    Base class for one contribution to the observation vector.

    A source declares *what* the solver must measure and *how many* scalar
    components that measurement contributes. The concatenation of all sources,
    in declaration order, is the canonical observation vector.
    """

    kind: ClassVar[str] = "base"

    @property
    def size(self) -> int:
        raise NotImplementedError

    def component_names(self) -> list[str]:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObservationSource":
        payload = dict(data)
        kind = payload.pop("kind")
        if kind not in _SOURCE_REGISTRY:
            raise ValueError(
                f"Unknown observation source kind {kind!r}. "
                f"Known kinds: {sorted(_SOURCE_REGISTRY)}"
            )
        return _SOURCE_REGISTRY[kind](**payload)


@_register
@dataclass(frozen=True)
class ProbeSource(ObservationSource):
    """
    Point probes of a field, sampled by the solver at control time.

    Parameters:
        field_name (str): OpenFOAM field name, e.g. 'p' or 'U'.
        positions (sequence of (x, y, z)): probe locations, in declaration order.
        components (sequence of int or None): for vector/tensor fields, which
            components to keep. None means the field is scalar (one component).
        name (str): label used in trajectory column names.
    """

    field_name: str
    positions: tuple[Vec3, ...]
    components: tuple[int, ...] | None = None
    name: str = "probes"

    kind: ClassVar[str] = "probe"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "positions", tuple(tuple(float(c) for c in p) for p in self.positions)
        )
        if self.components is not None:
            object.__setattr__(self, "components", tuple(int(c) for c in self.components))
        if not self.positions:
            raise ValueError("ProbeSource requires at least one position")
        for p in self.positions:
            if len(p) != 3:
                raise ValueError(f"Probe position must have 3 coordinates, got {p}")

    @property
    def size(self) -> int:
        per_point = len(self.components) if self.components else 1
        return len(self.positions) * per_point

    def component_names(self) -> list[str]:
        names = []
        for i in range(len(self.positions)):
            if self.components:
                names.extend(
                    f"{self.name}.{self.field_name}{c}.{i}" for c in self.components
                )
            else:
                names.append(f"{self.name}.{self.field_name}.{i}")
        return names

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "field_name": self.field_name,
            "positions": [list(p) for p in self.positions],
            "components": list(self.components) if self.components else None,
            "name": self.name,
        }


@_register
@dataclass(frozen=True)
class PatchSource(ObservationSource):
    """
    Scalar reduction of a field over a boundary patch.

    Parameters:
        field_name (str): field to reduce.
        patch (str): patch name.
        operation (str): one of 'average', 'areaAverage', 'integral',
            'areaIntegral', 'min', 'max'.
        component (int or None): component index for non-scalar fields.
    """

    field_name: str
    patch: str
    operation: Literal[
        "average", "areaAverage", "integral", "areaIntegral", "min", "max"
    ] = "areaAverage"
    component: int | None = None
    name: str = "patch"

    kind: ClassVar[str] = "patch"

    @property
    def size(self) -> int:
        return 1

    def component_names(self) -> list[str]:
        suffix = "" if self.component is None else str(self.component)
        return [f"{self.name}.{self.patch}.{self.field_name}{suffix}"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "field_name": self.field_name,
            "patch": self.patch,
            "operation": self.operation,
            "component": self.component,
            "name": self.name,
        }


@_register
@dataclass(frozen=True)
class ForceCoeffSource(ObservationSource):
    """
    Force coefficients on a patch, as produced by the forceCoeffs functionObject.

    Parameters:
        patch (str): patch the coefficients are computed on.
        coefficients (sequence of str): e.g. ('Cd', 'Cl'), in declaration order.
    """

    patch: str
    coefficients: tuple[str, ...] = ("Cd", "Cl")
    name: str = "forceCoeffs"

    kind: ClassVar[str] = "forceCoeffs"

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficients", tuple(self.coefficients))
        if not self.coefficients:
            raise ValueError("ForceCoeffSource requires at least one coefficient")

    @property
    def size(self) -> int:
        return len(self.coefficients)

    def component_names(self) -> list[str]:
        return [f"{self.name}.{self.patch}.{c}" for c in self.coefficients]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "patch": self.patch,
            "coefficients": list(self.coefficients),
            "name": self.name,
        }


@_register
@dataclass(frozen=True)
class CustomSource(ObservationSource):
    """
    Escape hatch for any measurement the solver-side controller knows how to
    produce but that has no dedicated source class yet.

    Parameters:
        name (str): identifier the solver-side controller dispatches on.
        n_components (int): how many scalars this source contributes.
        options (dict): free-form entries forwarded verbatim to the OpenFOAM dict.
    """

    name: str
    n_components: int
    options: dict[str, Any] = field(default_factory=dict)

    kind: ClassVar[str] = "custom"

    def __post_init__(self) -> None:
        if self.n_components < 1:
            raise ValueError("CustomSource.n_components must be >= 1")

    @property
    def size(self) -> int:
        return self.n_components

    def component_names(self) -> list[str]:
        if self.n_components == 1:
            return [self.name]
        return [f"{self.name}.{i}" for i in range(self.n_components)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "n_components": self.n_components,
            "options": dict(self.options),
        }


# Observation / action specs

@dataclass(frozen=True)
class ObservationSpec:
    """
    Declares the observation the policy consumes at control time.

    The observation vector is the concatenation of all sources in declaration
    order, optionally repeated over the last `stack` control steps (oldest
    first). The solver must build this exact vector and log it verbatim in the
    trajectory; Python reads that log instead of recomputing it.

    Parameters:
        sources (sequence of ObservationSource): measurements, in canonical order.
        stack (int): number of past control steps concatenated into one
            observation. 1 means only the current step.
    """

    sources: tuple[ObservationSource, ...]
    stack: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", tuple(self.sources))
        if not self.sources:
            raise ValueError("ObservationSpec requires at least one source")
        if self.stack < 1:
            raise ValueError("ObservationSpec.stack must be >= 1")

    @property
    def frame_dim(self) -> int:
        """Number of components contributed by one control step."""
        return sum(s.size for s in self.sources)

    @property
    def dim(self) -> int:
        """Total observation dimension, including stacking."""
        return self.frame_dim * self.stack

    def component_names(self) -> list[str]:
        frame = [n for s in self.sources for n in s.component_names()]
        if self.stack == 1:
            return frame
        return [f"t-{self.stack - 1 - k}|{n}" for k in range(self.stack) for n in frame]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources": [s.to_dict() for s in self.sources],
            "stack": self.stack,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObservationSpec":
        return cls(
            sources=tuple(ObservationSource.from_dict(d) for d in data["sources"]),
            stack=int(data.get("stack", 1)),
        )


@dataclass(frozen=True)
class ActionSpec:
    """
    Declares the action the policy produces and how the solver applies it.

    Bounds are always physical. For the 'beta' distribution the graph emits
    parameters over [0, 1] and the solver rescales into [low, high] using these
    bounds, read from the ONNX metadata. For 'gaussian' the graph emits mean and
    log_std already in physical units.

    Parameters:
        name (str): action label, used in trajectory columns.
        targets (str or sequence of str): what the action drives, in order.
            Target i receives component i of the action, so the order here is
            the order of the action vector, as it already is for low and high.
            A single name is the one component case.
        n_components (int or None): action dimension. None takes it from the
            number of targets, which is the usual case.
        low, high (float or sequence of float): physical bounds, scalar or per component.
        distribution ('gaussian' or 'beta'): policy distribution family.
            Gaussian is the default; beta never leaves the bounds and needs no
            clipping.
        ramp_fraction (float): the solver ramps linearly from the previous
            action to the new one over this fraction of the control interval.
            0 means an immediate step.
    """

    name: str
    targets: Any
    low: float | Sequence[float]
    high: float | Sequence[float]
    n_components: int | None = None
    distribution: Literal["gaussian", "beta"] = "gaussian"
    ramp_fraction: float = 0.0

    def __post_init__(self) -> None:
        targets = self._normalize_targets(self.targets)
        object.__setattr__(self, "targets", targets)

        n_components = len(targets) if self.n_components is None else self.n_components
        if n_components != len(targets):
            raise ValueError(
                f"the action declares {n_components} components but has "
                f"{len(targets)} target(s) {list(targets)}; one target drives "
                "one component"
            )
        object.__setattr__(self, "n_components", n_components)

        low = self._broadcast(self.low, "low")
        high = self._broadcast(self.high, "high")
        for lo, hi in zip(low, high):
            if not lo < hi:
                raise ValueError(f"ActionSpec requires low < high, got ({lo}, {hi})")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

        if self.distribution not in ("gaussian", "beta"):
            raise ValueError(
                f"Unsupported distribution {self.distribution!r}; use 'gaussian' or 'beta'"
            )
        if not 0.0 <= self.ramp_fraction <= 1.0:
            raise ValueError(
                f"ramp_fraction must be in [0, 1], got {self.ramp_fraction}"
            )

    @staticmethod
    def _normalize_targets(value: Any) -> tuple[str, ...]:
        """Bring the targets to a tuple of names, in action component order."""
        names = (value,) if isinstance(value, str) else tuple(str(v) for v in value)
        if not names:
            raise ValueError("ActionSpec requires at least one target")
        if len(set(names)) != len(names):
            raise ValueError(f"ActionSpec has duplicate targets: {list(names)}")
        return names

    def _broadcast(self, value: Any, label: str) -> tuple[float, ...]:
        if isinstance(value, (int, float)):
            return tuple(float(value) for _ in range(self.n_components))
        values = tuple(float(v) for v in value)
        if len(values) != self.n_components:
            raise ValueError(
                f"ActionSpec.{label} has {len(values)} entries "
                f"but n_components is {self.n_components}"
            )
        return values

    @property
    def dim(self) -> int:
        return self.n_components

    def component_names(self) -> list[str]:
        if self.n_components == 1:
            return [self.name]
        return [f"{self.name}.{i}" for i in range(self.n_components)]

    @property
    def target_names(self) -> list[str]:
        return list(self.targets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "targets": list(self.targets),
            "n_components": self.n_components,
            "low": list(self.low),
            "high": list(self.high),
            "distribution": self.distribution,
            "ramp_fraction": self.ramp_fraction,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionSpec":
        payload = dict(data)
        payload["targets"] = tuple(payload["targets"])
        return cls(**payload)


# Policy spec

@dataclass(frozen=True)
class PolicySpec:
    """
    Complete Python <-> solver contract for one closed-loop control setup.

    Parameters:
        observation (ObservationSpec): what the policy sees.
        action (ActionSpec): what the policy drives.
        control_interval (float): simulated time between policy evaluations.
        start_time (float): time at which control begins. Before it, the solver
            runs uncontrolled (warm start from a developed base state).
        end_time (float or None): time at which control stops. None means the
            end of the run.
    """

    observation: ObservationSpec
    action: ActionSpec
    control_interval: float
    start_time: float = 0.0
    end_time: float | None = None

    def __post_init__(self) -> None:
        if self.control_interval <= 0:
            raise ValueError("control_interval must be > 0")
        if self.end_time is not None and self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")

    @property
    def obs_dim(self) -> int:
        return self.observation.dim

    @property
    def act_dim(self) -> int:
        return self.action.dim

    @property
    def input_names(self) -> tuple[str, ...]:
        """ONNX input names implied by the distribution family."""
        if self.action.distribution == "beta":
            return ("observation",)
        return ("observation", "noise")

    @property
    def output_names(self) -> tuple[str, ...]:
        """ONNX output names implied by the distribution family."""
        if self.action.distribution == "beta":
            return ("alpha", "beta")
        return ("action",)

    @property
    def noise_dim(self) -> int:
        """Number of standard normal values the solver must supply per step."""
        return self.act_dim if self.action.distribution == "gaussian" else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "action": self.action.to_dict(),
            "control_interval": self.control_interval,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolicySpec":
        return cls(
            observation=ObservationSpec.from_dict(data["observation"]),
            action=ActionSpec.from_dict(data["action"]),
            control_interval=float(data["control_interval"]),
            start_time=float(data.get("start_time", 0.0)),
            end_time=data.get("end_time"),
        )

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "PolicySpec":
        return cls.from_dict(json.loads(text))

    @property
    def hash(self) -> str:
        """
        Stable identifier of the contract.

        Written into the ONNX metadata, copied by the solver into the trajectory
        header, and compared on read. A mismatch means the policy, the case
        dictionary and the parser disagree, and is caught before any physics runs.
        """
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def to_metadata(self) -> dict[str, str]:
        """
        Key/value pairs stamped into the ONNX metadata_props.

        The full spec is stored under 'uqtopus.spec' for Python, while the flat
        scalar entries exist so the solver can read what it needs without a JSON
        parser.
        """
        return {
            "uqtopus.spec_hash": self.hash,
            "uqtopus.spec": self.to_json(),
            "uqtopus.obs_dim": str(self.obs_dim),
            "uqtopus.act_dim": str(self.act_dim),
            "uqtopus.distribution": self.action.distribution,
            "uqtopus.action_low": " ".join(str(v) for v in self.action.low),
            "uqtopus.action_high": " ".join(str(v) for v in self.action.high),
            "uqtopus.control_interval": str(self.control_interval),
            "uqtopus.start_time": str(self.start_time),
            "uqtopus.ramp_fraction": str(self.action.ramp_fraction),
            "uqtopus.action_targets": " ".join(
                f"{name}:{i}" for i, name in enumerate(self.action.targets)
            ),
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, str]) -> "PolicySpec":
        if "uqtopus.spec" not in metadata:
            raise ValueError(
                "ONNX metadata has no 'uqtopus.spec' entry; the file was not "
                "produced by uqtopus.rl.export_policy()"
            )
        return cls.from_json(metadata["uqtopus.spec"])

    def trajectory_columns(self) -> list[str]:
        """Canonical column order of the trajectory file written by the solver."""
        return (
            ["time"]
            + self.observation.component_names()
            + self.action.component_names()
            + ["seed"]
        )

    def __repr__(self) -> str:
        return (
            f"PolicySpec(obs_dim={self.obs_dim}, act_dim={self.act_dim}, "
            f"dist={self.action.distribution!r}, dt={self.control_interval}, "
            f"hash={self.hash})"
        )
