"""
PPO over Closed-Loop Rollouts

stable-baselines3's PPO with `collect_rollouts` replaced: exports the policy,
launches the solvers, reads the trajectories back and fills the buffer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import stable_baselines3 as sb3

from .buffer import build_rollout_buffer, rollout_statistics
from .export import PolicyArtifact, RunningStatistics, export_policy
from .runner import ClosedLoopRunner, Rollout
from .validate import validate_policy

logger = logging.getLogger(__name__)


class PPO(sb3.PPO):
    """
    PPO whose experience comes from intrusive closed-loop solver runs.

    Parameters:
        policy: as in stable-baselines3, e.g. 'MlpPolicy'.
        runner (ClosedLoopRunner): produces the episodes and supplies the
            spaces. Its spec must declare a Gaussian policy.
        n_episodes (int): solver runs per training iteration, sharing one frozen
            policy.
        n_jobs (int): how many of those run at once.
        export_dir (str or Path): where the per-iteration .onnx files and
            manifests are written.
        normalize_observations (bool): bake each iteration's observation
            statistics into that iteration's graph.
        validate (bool): check every exported policy against the contract before
            it reaches a solver.
        **kwargs: passed to stable-baselines3. Device defaults to 'cpu'.
    """

    def __init__(
        self,
        policy: Any,
        runner: ClosedLoopRunner,
        *,
        n_episodes: int = 4,
        n_jobs: int = 1,
        export_dir: str | Path = "policies",
        normalize_observations: bool = True,
        validate: bool = True,
        **kwargs: Any,
    ) -> None:
        if runner.spec.action.distribution != "gaussian":
            raise ValueError(
                "A stable-baselines3 actor emits a diagonal Gaussian, but the "
                f"spec declares distribution={runner.spec.action.distribution!r}. "
                "Use distribution='gaussian' with this class."
            )

        # n_steps is a placeholder; the buffer is rebuilt every iteration to fit
        # the episodes collected. Equal to batch_size to keep the base class's
        # divisibility warning quiet.
        kwargs.setdefault("batch_size", 64)
        kwargs.setdefault("n_steps", kwargs["batch_size"])
        kwargs.setdefault("device", "cpu")

        super().__init__(policy, runner.stub_env(), **kwargs)

        self.runner = runner
        self.spec = runner.spec
        self.n_episodes = n_episodes
        self.n_jobs = n_jobs
        self.export_dir = Path(export_dir)
        self.validate = validate

        self.statistics = (
            RunningStatistics(self.spec.obs_dim) if normalize_observations else None
        )
        self.artifacts: list[PolicyArtifact] = []
        self.rollouts: list[Rollout] = []
        self._uqtopus_iteration = 0

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps) -> bool:
        """
        Export the policy, run the solvers, and fill the buffer from what they
        wrote. The env and rollout_buffer arguments are ignored.
        """
        callback.on_rollout_start()

        iteration = self._uqtopus_iteration
        artifact = self._export(iteration)
        self.artifacts.append(artifact)

        rollout = self.runner.collect(
            artifact,
            n_episodes=self.n_episodes,
            iteration=iteration,
            n_jobs=self.n_jobs,
        )
        self.rollouts.append(rollout)

        self.rollout_buffer = build_rollout_buffer(
            rollout,
            self.policy,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        # The statistics advance only after the buffer is filled from this
        # iteration's snapshot.
        if self.statistics is not None:
            self.statistics.update(rollout.observations)

        self.num_timesteps += rollout.n_steps
        for key, value in rollout_statistics(rollout).items():
            self.logger.record(key, value)
        self.logger.record("rollout/policy", str(artifact.path.name))

        self._uqtopus_iteration += 1

        callback.update_locals(locals())
        callback.on_rollout_end()
        return not callback.on_step() is False

    # helpers

    def _export(self, iteration: int) -> PolicyArtifact:
        normalization = (
            self.statistics.snapshot() if self.statistics is not None else None
        )
        artifact = export_policy(
            self.policy,
            self.spec,
            self.export_dir / f"policy_iter{iteration:04d}.onnx",
            normalization=normalization,
            iteration=iteration,
            backend="sb3",
        )
        if self.validate:
            validate_policy(artifact, self.spec, strict=True)
        return artifact

    def export_current_policy(self, path: str | Path) -> PolicyArtifact:
        """
        Export the policy and the current statistics as they stand.
        """
        normalization = (
            self.statistics.snapshot() if self.statistics is not None else None
        )
        return export_policy(
            self.policy,
            self.spec,
            path,
            normalization=normalization,
            iteration=self._uqtopus_iteration,
            backend="sb3",
        )

    @property
    def returns_history(self) -> np.ndarray:
        """Mean undiscounted return per training iteration."""
        return np.array([r.returns.mean() for r in self.rollouts])
