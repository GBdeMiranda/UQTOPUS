"""
Policy Specification for Intrusive (Closed-Loop) RL

Single source of truth for the Python <-> solver contract, shared by the controller
dictionary, the ONNX graph signature and the trajectory parser.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, ClassVar, Sequence

Vec3 = tuple[float, float, float]

@dataclass(frozen=True)
class ProbeSource:
    """
    Point probes of a field, sampled by the solver at control time.

    Parameters:
        field_name (str): OpenFOAM field name.
        positions (sequence of (x, y, z)): probe locations, in declaration order.
        components (sequence of int or None): for a vector field, which of its
            three components to keep. None means the field is scalar.
        name (str): label used in trajectory column names.
        interpolation (str): OpenFOAM interpolation scheme the solver reads the
            field with, 'cell' for the value of the cell holding the point,
            'cellPoint' for one that follows where inside the cell it lies.
    """

    field_name: str
    positions: tuple[Vec3, ...]
    components: tuple[int, ...] | None = None
    name: str = "probes"
    interpolation: str = "cellPoint"

    kind: ClassVar[str] = "probe"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "positions", tuple(tuple(float(c) for c in p) for p in self.positions)
        )
        if self.components is not None:
            object.__setattr__(self, "components", tuple(int(c) for c in self.components))
            for c in self.components:
                if not 0 <= c < 3:
                    raise ValueError(
                        f"Component {c} is out of range; the solver reads vector "
                        "fields, whose components are 0, 1 and 2"
                    )
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
            "interpolation": self.interpolation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProbeSource":
        payload = dict(data)
        payload.pop("kind", None)
        return cls(**payload)


@dataclass(frozen=True)
class RegistrySource:
    """
    One scalar the case publishes under a name, read by the solver at control
    time.

    Whatever computes it, a functionObject of the case among others, has to
    store it in the mesh registry before the control step reads it.

    Parameters:
        name (str): name the scalar is registered under, also the trajectory
            column name.
    """

    name: str

    kind: ClassVar[str] = "registry"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RegistrySource requires a name")

    @property
    def size(self) -> int:
        return 1

    def component_names(self) -> list[str]:
        return [self.name]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "name": self.name}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegistrySource":
        payload = dict(data)
        payload.pop("kind", None)
        return cls(**payload)


Source = ProbeSource | RegistrySource

_SOURCE_TYPES = {ProbeSource.kind: ProbeSource, RegistrySource.kind: RegistrySource}


@dataclass(frozen=True)
class ObservationSpec:
    """
    Declares the observation the policy consumes at control time.

    The observation vector is the concatenation of all sources in declaration
    order.

    Parameters:
        sources (sequence of ProbeSource or RegistrySource): measurements, in
            canonical order.
    """

    sources: tuple[Source, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", tuple(self.sources))
        if not self.sources:
            raise ValueError("ObservationSpec requires at least one source")

    @property
    def dim(self) -> int:
        return sum(s.size for s in self.sources)

    def component_names(self) -> list[str]:
        return [n for s in self.sources for n in s.component_names()]

    def to_dict(self) -> dict[str, Any]:
        return {"sources": [s.to_dict() for s in self.sources]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObservationSpec":
        return cls(
            sources=tuple(
                _SOURCE_TYPES[d.get("kind", ProbeSource.kind)].from_dict(d)
                for d in data["sources"]
            )
        )


@dataclass(frozen=True)
class ActionSpec:
    """
    Declares the action the policy produces and how the solver applies it.

    The policy samples the action from a Gaussian without bounds, and the solver
    clips it to [low, high] when applying it.

    Parameters:
        name (str): action label, used in trajectory columns.
        targets (str or sequence of str): what the action drives. Target i
            receives component i, so this order is the action vector order.
        low, high (float or sequence of float): bounds in the units of the
            action, scalar or one per target.
        ramp_fraction (float): the solver ramps linearly from the previous
            action to the new one over this fraction of the control interval.
            0 means an immediate step.
    """

    name: str
    targets: Any
    low: float | Sequence[float]
    high: float | Sequence[float]
    ramp_fraction: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.targets, str):
            targets = (self.targets,)
        else:
            targets = tuple(str(t) for t in self.targets)
        if not targets:
            raise ValueError("ActionSpec requires at least one target")
        if len(set(targets)) != len(targets):
            raise ValueError(f"ActionSpec has duplicate targets: {list(targets)}")
        object.__setattr__(self, "targets", targets)

        for label in ("low", "high"):
            value = getattr(self, label)
            if isinstance(value, (int, float)):
                values = (float(value),) * len(targets)
            else:
                values = tuple(float(v) for v in value)
            if len(values) != len(targets):
                raise ValueError(
                    f"ActionSpec.{label} has {len(values)} entries for {len(targets)} targets"
                )
            object.__setattr__(self, label, values)

        if not all(lo < hi for lo, hi in zip(self.low, self.high)):
            raise ValueError(f"ActionSpec requires low < high, got {self.low} and {self.high}")
        if not 0.0 <= self.ramp_fraction <= 1.0:
            raise ValueError(f"ramp_fraction must be in [0, 1], got {self.ramp_fraction}")

    @property
    def n_components(self) -> int:
        return len(self.targets)

    def component_names(self) -> list[str]:
        if self.n_components == 1:
            return [self.name]
        return [f"{self.name}.{i}" for i in range(self.n_components)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "targets": list(self.targets),
            "n_components": self.n_components,
            "low": list(self.low),
            "high": list(self.high),
            "ramp_fraction": self.ramp_fraction,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionSpec":
        payload = dict(data)
        payload.pop("n_components")
        return cls(**payload)


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
        end_time (float or None): time at which control stops, the last action
            holding from then on. None means the end of the run.
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
        return self.action.n_components

    @property
    def input_names(self) -> tuple[str, ...]:
        """ONNX input names."""
        return ("observation", "noise")

    @property
    def output_names(self) -> tuple[str, ...]:
        """ONNX output names."""
        return ("action",)

    @property
    def noise_dim(self) -> int:
        """Number of standard normal values the solver supplies per step."""
        return self.act_dim

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

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "PolicySpec":
        return cls.from_dict(json.loads(text))

    @property
    def hash(self) -> str:
        """Stable identifier of the contract, carried by the ONNX file and the trajectory."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def to_metadata(self) -> dict[str, str]:
        """Key/value pairs stamped into the ONNX metadata_props."""
        return {
            "uqtopus.spec_hash": self.hash,
            "uqtopus.spec": self.to_json(),
            "uqtopus.obs_dim": str(self.obs_dim),
            "uqtopus.act_dim": str(self.act_dim),
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

    def __repr__(self) -> str:
        return (
            f"PolicySpec(obs_dim={self.obs_dim}, act_dim={self.act_dim}, "
            f"dt={self.control_interval}, hash={self.hash})"
        )
