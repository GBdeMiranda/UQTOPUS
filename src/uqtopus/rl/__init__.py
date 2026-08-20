"""
Intrusive (closed-loop) reinforcement learning for OpenFOAM.

One solver run is one episode. Python and the solver exchange two files: a policy
in ONNX, and a trajectory log with one row per control step.

See docs/rl_intrusive_design.md for the design rationale.
"""

from .spec import (
    CONTRACT_VERSION,
    ActionSpec,
    CustomSource,
    ForceCoeffSource,
    ObservationSource,
    ObservationSpec,
    PatchSource,
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
    RewardError,
    align_to_control,
    attach,
    evaluate_reward,
    moving_average,
    read_function_object,
)
from .trajectory import (
    TrajectoryError,
    concat_trajectories,
    find_trajectory,
    read_trajectory,
    write_trajectory,
)
from .runner import ClosedLoopRunner, EpisodeFailure, Rollout
from .validate import ValidationReport, validate_export, validate_policy

__all__ = [
    "CONTRACT_VERSION",
    "ActionSpec",
    "ClosedLoopRunner",
    "CustomSource",
    "EpisodeFailure",
    "ForceCoeffSource",
    "Normalization",
    "ObservationSource",
    "ObservationSpec",
    "PatchSource",
    "PolicyArtifact",
    "PolicySpec",
    "ProbeSource",
    "Rollout",
    "RunningStatistics",
    "RewardError",
    "TrajectoryError",
    "ValidationReport",
    "align_to_control",
    "attach",
    "build_mlp",
    "build_rollout_buffer",
    "concat_trajectories",
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
