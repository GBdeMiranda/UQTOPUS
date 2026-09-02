"""
Intrusive (closed-loop) reinforcement learning for OpenFOAM.

One solver run is one episode. Python and the solver exchange two files: a policy
in ONNX, and a trajectory log with one row per control step.

See docs/rl_intrusive_design.md for the design rationale.
"""

from .spec import (
    ActionSpec,
    ObservationSpec,
    PolicySpec,
    ProbeSource,
)
from .export import (
    Normalization,
    PolicyArtifact,
    build_mlp,
    export_policy,
    export_random_policy,
    read_metadata,
    RunningStatistics,
    torch_reference,
)
from .buffer import build_rollout_buffer, normalized_observations, rollout_statistics
from .foam import controller_params, render_controller
from .reward import (
    align_to_control,
    attach,
    evaluate_reward,
    moving_average,
    read_function_object,
)
from .trajectory import (
    TrajectoryError,
    find_trajectory,
    read_trajectory,
    write_trajectory,
)
from .progress import TrainingLog
from .runner import ClosedLoopRunner, EpisodeFailure, Rollout
from .validate import ValidationReport, validate_export, validate_policy

__all__ = [
    "ActionSpec",
    "ClosedLoopRunner",
    "EpisodeFailure",
    "Normalization",
    "ObservationSpec",
    "PolicyArtifact",
    "PolicySpec",
    "ProbeSource",
    "Rollout",
    "RunningStatistics",
    "TrainingLog",
    "TrajectoryError",
    "ValidationReport",
    "align_to_control",
    "attach",
    "build_mlp",
    "build_rollout_buffer",
    "evaluate_reward",
    "export_policy",
    "export_random_policy",
    "find_trajectory",
    "controller_params",
    "moving_average",
    "normalized_observations",
    "read_function_object",
    "read_metadata",
    "render_controller",
    "rollout_statistics",
    "read_trajectory",
    "torch_reference",
    "validate_export",
    "validate_policy",
    "write_trajectory",
]
