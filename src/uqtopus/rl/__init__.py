"""
Intrusive (closed-loop) reinforcement learning for OpenFOAM.

One solver run is one episode. Python and the solver exchange two files: a policy
in ONNX, and a trajectory log with one row per control step. The PPO trainer is
uqtopus.rl.algos.PPO.
"""

from .spec import (
    ActionSpec,
    ObservationSpec,
    PolicySpec,
    ProbeSource,
    RegistrySource,
)
from .export import (
    Normalization,
    PolicyArtifact,
    export_policy,
    export_random_policy,
    read_metadata,
)
from .foam import render_controller
from .reward import align_to_control, read_function_object
from .trajectory import read_trajectory
from .progress import TrainingLog
from .runner import ClosedLoopRunner
from .validate import validate_policy

__all__ = [
    "ActionSpec",
    "ClosedLoopRunner",
    "Normalization",
    "ObservationSpec",
    "PolicyArtifact",
    "PolicySpec",
    "ProbeSource",
    "RegistrySource",
    "TrainingLog",
    "align_to_control",
    "export_policy",
    "export_random_policy",
    "read_function_object",
    "read_metadata",
    "read_trajectory",
    "render_controller",
    "validate_policy",
]
