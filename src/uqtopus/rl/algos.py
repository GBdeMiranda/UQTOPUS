"""
PPO over Closed-Loop Rollouts

stable-baselines3's PPO with `collect_rollouts` replaced: exports the policy,
launches the solvers, reads the trajectories back and fills the buffer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
import torch

from .buffer import build_rollout_buffer, rollout_statistics
from .export import PolicyArtifact, RunningStatistics, export_policy
from .runner import ClosedLoopRunner, Rollout
from .spec import PolicySpec
from .validate import validate_policy


class _SB3Actor(torch.nn.Module):
    """
    The actor half of a stable-baselines3 ActorCriticPolicy, returning
    (mean, log_std).

    Parameters:
        policy: the ActorCriticPolicy to read.
    """

    def __init__(self, policy: Any) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, x):
        features = self.policy.pi_features_extractor(x)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        mean = self.policy.action_net(latent_pi)
        return mean, self.policy.log_std.expand_as(mean)


class _SpacesOnlyEnv(gym.Env):
    """The spaces stable-baselines3 asks for at construction, and nothing else."""

    def __init__(self, spec: PolicySpec) -> None:
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(spec.obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.asarray(spec.action.low, dtype=np.float32),
            high=np.asarray(spec.action.high, dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}


class PPO(sb3.PPO):
    """
    PPO whose experience comes from intrusive closed-loop solver runs.

    Parameters:
        policy: as in stable-baselines3, e.g. 'MlpPolicy'.
        runner (ClosedLoopRunner): produces the episodes.
        n_episodes (int): solver runs per training iteration, sharing one frozen
            policy.
        n_jobs (int): how many of those run at once.
        export_dir (str or Path): where the per-iteration .onnx files are
            written.
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
        **kwargs: Any,
    ) -> None:
        # n_steps is a placeholder; the buffer is rebuilt every iteration to fit
        # the episodes collected. Equal to batch_size to keep the base class's
        # divisibility warning quiet.
        kwargs.setdefault("batch_size", 64)
        kwargs.setdefault("n_steps", kwargs["batch_size"])
        kwargs.setdefault("device", "cpu")

        super().__init__(policy, _SpacesOnlyEnv(runner.spec), **kwargs)

        self.runner = runner
        self.n_episodes = n_episodes
        self.n_jobs = n_jobs
        self.export_dir = Path(export_dir)
        self.statistics = RunningStatistics(runner.spec.obs_dim)
        self.artifacts: list[PolicyArtifact] = []
        self.rollouts: list[Rollout] = []

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps) -> bool:
        """
        Export the policy, run the solvers, and fill the buffer from what they
        wrote. The env and rollout_buffer arguments are ignored.
        """
        callback.on_rollout_start()

        iteration = len(self.artifacts)
        artifact = self.export_current_policy(self.export_dir / f"policy_iter{iteration:04d}.onnx")
        validate_policy(artifact, self.runner.spec)
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
        self.statistics.update(rollout.observations)

        self.num_timesteps += rollout.n_steps
        for key, value in rollout_statistics(rollout).items():
            self.logger.record(key, value)
        self.logger.record("rollout/policy", artifact.path.name)

        callback.update_locals(locals())
        callback.on_rollout_end()
        return callback.on_step()

    def export_current_policy(self, path: str | Path) -> PolicyArtifact:
        """Export the policy with the observation statistics as they stand."""
        return export_policy(
            _SB3Actor(self.policy),
            self.runner.spec,
            path,
            normalization=self.statistics.snapshot(),
            iteration=len(self.artifacts),
        )

    @property
    def returns_history(self) -> np.ndarray:
        """Mean undiscounted return per training iteration."""
        return np.array([r.returns.mean() for r in self.rollouts])
