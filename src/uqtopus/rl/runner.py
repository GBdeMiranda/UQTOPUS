"""
Closed-Loop Rollout Collection

One solver run is one episode. Exports the policy, launches N cases and reads back
N trajectories. Exposes collect(artifact) rather than a gymnasium step().
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import repeat
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import gymnasium as gym
import numpy as np
import xarray as xr

from ..exceptions import SolverDivergedError
from ..simulation import OpenFOAMSimulator, run_simulation
from .export import PolicyArtifact
from .foam import controller_params
from .reward import RewardFn, attach, evaluate_reward, read_function_object
from .spec import PolicySpec
from .trajectory import TrajectoryError, read_trajectory

logger = logging.getLogger(__name__)

RunFn = Callable[[Path, dict[str, Any]], None]


@dataclass(frozen=True)
class EpisodeFailure:
    """One case that did not produce usable data."""

    index: int
    case_dir: Path
    reason: str

    def __str__(self) -> str:
        return f"episode {self.index} ({self.case_dir.name}): {self.reason}"


@dataclass
class Rollout:
    """
    Experience collected with one frozen policy.

    The flat views are laid out the way an on-policy buffer wants them: every
    control step of every episode concatenated, with episode_starts marking the
    boundaries.
    """

    artifact: PolicyArtifact
    episodes: list[xr.Dataset] = field(default_factory=list)
    failures: list[EpisodeFailure] = field(default_factory=list)

    @property
    def lengths(self) -> list[int]:
        return [int(ds.sizes["time"]) for ds in self.episodes]

    @property
    def n_steps(self) -> int:
        return sum(self.lengths)

    @property
    def observations(self) -> np.ndarray:
        return self._stack("observation")

    @property
    def actions(self) -> np.ndarray:
        return self._stack("action")

    @property
    def rewards(self) -> np.ndarray:
        return np.concatenate([ds["reward"].values for ds in self.episodes])

    @property
    def episode_starts(self) -> np.ndarray:
        """True on the first control step of each episode."""
        flags = np.zeros(self.n_steps, dtype=bool)
        index = 0
        for length in self.lengths:
            flags[index] = True
            index += length
        return flags

    @property
    def returns(self) -> np.ndarray:
        """Undiscounted sum of rewards per episode."""
        return np.array([float(ds["reward"].sum()) for ds in self.episodes])

    @property
    def fraction_at_bounds(self) -> float:
        """Share of recorded action components that reached or passed their bounds."""
        actions = self.actions
        low = np.asarray(self.artifact.spec.action.low)
        high = np.asarray(self.artifact.spec.action.high)
        outside = (actions <= low) | (actions >= high)
        return float(np.mean(outside))

    def _stack(self, name: str) -> np.ndarray:
        if not self.episodes:
            raise ValueError("the rollout holds no episodes")
        return np.concatenate([ds[name].values for ds in self.episodes], axis=0)

    def __repr__(self) -> str:
        return (
            f"Rollout(episodes={len(self.episodes)}, steps={self.n_steps}, "
            f"failures={len(self.failures)}, "
            f"mean_return={self.returns.mean():.4g})"
            if self.episodes
            else f"Rollout(episodes=0, failures={len(self.failures)})"
        )


class _SpacesOnlyEnv(gym.Env):
    """
    Environment that carries the spaces and nothing else.

    Parameters:
        observation_space (gym.spaces.Box): the observation space.
        action_space (gym.spaces.Box): the action space.
    """

    metadata: dict = {"render_modes": []}

    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Box) -> None:
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        raise NotImplementedError(
            "this environment carries only the spaces; use "
            "ClosedLoopRunner.collect() to produce episodes"
        )


class ClosedLoopRunner:
    """
    Runs episodes of intrusive closed-loop control and returns their experience.

    Parameters:
        simulator (OpenFOAMSimulator): supplies the case template, the solver
            script and the output directory. Its run() is not used.
        spec (PolicySpec): the contract, rendered into the case and checked
            against what the solver wrote.
        reward_fn (callable): maps the trajectory, with any requested
            functionObject output merged in, to one reward per control step.
        controller_keys (str or sequence of str): where the controller block is
            rendered, in 'folder__file__variable' form, e.g. '0__U__controller'.
        function_objects (sequence of str): functionObject names read from each
            case and aligned onto the control steps before reward_fn sees them.
        run_fn (callable or None): executes a case, as run_fn(case_dir, params).
            None runs the solver locally.
    """

    def __init__(
        self,
        simulator: OpenFOAMSimulator,
        spec: PolicySpec,
        reward_fn: RewardFn,
        *,
        controller_keys: str | Sequence[str],
        function_objects: Sequence[str] = (),
        run_fn: RunFn | None = None,
    ) -> None:
        self.simulator = simulator
        self.spec = spec
        self.reward_fn = reward_fn
        self.controller_keys = (
            [controller_keys] if isinstance(controller_keys, str) else list(controller_keys)
        )
        self.function_objects = list(function_objects)
        self.run_fn = run_fn or self._run_locally

    # gym vocabulary, for the parts of it that apply

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.spec.obs_dim,), dtype=np.float32
        )

    @property
    def action_space(self):
        return gym.spaces.Box(
            low=np.asarray(self.spec.action.low, dtype=np.float32),
            high=np.asarray(self.spec.action.high, dtype=np.float32),
            dtype=np.float32,
        )

    def stub_env(self) -> gym.Env:
        """
        Build a gymnasium.Env carrying only the observation and action spaces.

        Returns:
            gymnasium.Env: reset() returns a zero observation, step() raises.
        """
        return _SpacesOnlyEnv(self.observation_space, self.action_space)

    # collection

    def collect(
        self,
        artifact: PolicyArtifact,
        n_episodes: int = 1,
        *,
        seeds: Sequence[int] | None = None,
        iteration: int | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> Rollout:
        """
        Run n_episodes with one frozen policy and return their experience.

        Parameters:
            artifact (PolicyArtifact): the exported policy. Its spec must match
                the runner's, otherwise the cases and the policy disagree.
            n_episodes (int): how many cases to run with this policy.
            seeds (sequence of int or None): one RNG seed per episode, written
                into the case and recorded in the trajectory so a run can be
                replayed. None derives them from the iteration.
            iteration (int): training iteration, used to name the run
                directories. Falls back to the artifact's.
            n_jobs (int): how many cases to run at once, on threads.

        Returns:
            Rollout
        """
        if artifact.spec.hash != self.spec.hash:
            raise ValueError(
                f"the policy implements contract {artifact.spec.hash} but the "
                f"runner is configured for {self.spec.hash}"
            )

        iteration = iteration if iteration is not None else (artifact.iteration or 0)
        if seeds is None:
            seeds = [iteration * 100_000 + i for i in range(n_episodes)]
        if len(seeds) != n_episodes:
            raise ValueError(f"got {len(seeds)} seeds for {n_episodes} episodes")

        policy_path = artifact.path.resolve()
        indices = range(n_episodes)
        values = [int(seed) for seed in seeds]

        if n_jobs > 1:
            with ThreadPoolExecutor(max_workers=n_jobs) as pool:
                results = list(
                    pool.map(
                        self._run_episode,
                        indices,
                        values,
                        repeat(policy_path),
                        repeat(iteration),
                        repeat(verbose),
                    )
                )
        else:
            results = [
                self._run_episode(index, seed, policy_path, iteration, verbose)
                for index, seed in zip(indices, values)
            ]

        rollout = Rollout(artifact=artifact)
        for episode, failure in results:
            if episode is not None:
                rollout.episodes.append(episode)
            if failure is not None:
                rollout.failures.append(failure)

        if rollout.failures:
            logger.warning(
                "%d of %d episodes failed: %s",
                len(rollout.failures),
                n_episodes,
                "; ".join(str(f) for f in rollout.failures),
            )
        if not rollout.episodes:
            raise RuntimeError(
                f"all {n_episodes} episodes failed; nothing to learn from. "
                + "; ".join(str(f) for f in rollout.failures)
            )
        return rollout

    def _run_episode(
        self,
        index: int,
        seed: int,
        policy_path: Path,
        iteration: int,
        verbose: bool,
    ) -> tuple[xr.Dataset | None, EpisodeFailure | None]:
        case_dir = self.simulator.output_path / f"iter{iteration:04d}_ep{index:02d}"

        params = controller_params(
            self.spec, policy_path, self.controller_keys, seed=seed
        )

        diverged: str | None = None
        try:
            self.run_fn(case_dir, params)
        except SolverDivergedError as exc:
            diverged = f"solver exited with code {exc.returncode}"
            logger.warning("Episode %d diverged in %s", index, case_dir)
        except Exception as exc:  # a launcher failure is not a diverged run
            return None, EpisodeFailure(index, case_dir, f"launch failed: {exc}")

        try:
            episode = self._read_episode(case_dir, seed, verbose)
        except (TrajectoryError, FileNotFoundError) as exc:
            reason = f"{diverged}; no usable trajectory" if diverged else str(exc)
            return None, EpisodeFailure(index, case_dir, reason)
        except Exception as exc:
            return None, EpisodeFailure(index, case_dir, f"reward failed: {exc}")

        if diverged:
            episode.attrs["diverged"] = True
            return episode, EpisodeFailure(index, case_dir, f"{diverged}; partial kept")

        episode.attrs["diverged"] = False
        return episode, None

    def _read_episode(self, case_dir: Path, seed: int, verbose: bool) -> xr.Dataset:
        trajectory = read_trajectory(case_dir, self.spec)

        series = [read_function_object(case_dir, name) for name in self.function_objects]
        data = attach(trajectory, *series) if series else trajectory

        rewards = evaluate_reward(data, self.reward_fn)
        data = data.assign(reward=("time", rewards))
        data.attrs["seed"] = seed
        data.attrs["case_dir"] = str(case_dir)

        if verbose:
            logger.info(
                "%s: %d steps, return %.4g", case_dir.name, data.sizes["time"], rewards.sum()
            )
        return data

    def _run_locally(self, case_dir: Path, params: dict[str, Any]) -> None:
        run_simulation(
            params=params,
            exp_config={
                "input_path": str(self.simulator.template_path),
                "output_path": str(case_dir),
                "solver": self.simulator.solver_script,
            },
        )

    def __repr__(self) -> str:
        return (
            f"ClosedLoopRunner(obs_dim={self.spec.obs_dim}, "
            f"act_dim={self.spec.act_dim}, spec_hash={self.spec.hash})"
        )
