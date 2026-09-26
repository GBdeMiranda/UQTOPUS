"""
Closed-Loop Rollout Collection

One solver run is one episode. Launches N cases with one exported policy and reads
back N trajectories.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import repeat
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import xarray as xr

from ..exceptions import SolverDivergedError
from ..simulation import OpenFOAMSimulator, run_simulation
from .export import PolicyArtifact
from .foam import controller_params
from .reward import RewardFn, attach, evaluate_reward, read_function_object
from .spec import PolicySpec
from .trajectory import read_trajectory

logger = logging.getLogger(__name__)

RunFn = Callable[[Path, dict[str, Any]], None]


@dataclass(frozen=True)
class EpisodeFailure:
    """One case whose solver diverged."""

    index: int
    case_dir: Path
    reason: str

    def __str__(self) -> str:
        return f"episode {self.index} ({self.case_dir.name}): {self.reason}"


@dataclass
class Rollout:
    """
    Experience collected with one frozen policy.

    The flat views concatenate every control step of every episode, with
    episode_starts marking the boundaries.
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
        return self._stack("reward")

    @property
    def episode_starts(self) -> np.ndarray:
        """True on the first control step of each episode."""
        return np.concatenate([np.arange(n) == 0 for n in self.lengths])

    @property
    def returns(self) -> np.ndarray:
        """Undiscounted sum of rewards per episode."""
        return np.array([float(ds["reward"].sum()) for ds in self.episodes])

    @property
    def fraction_at_bounds(self) -> float:
        """Share of recorded action components that reached or passed their bounds."""
        low = np.asarray(self.artifact.spec.action.low)
        high = np.asarray(self.artifact.spec.action.high)
        return float(np.mean((self.actions <= low) | (self.actions >= high)))

    def _stack(self, name: str) -> np.ndarray:
        return np.concatenate([ds[name].values for ds in self.episodes], axis=0)


class ClosedLoopRunner:
    """
    Runs episodes of intrusive closed-loop control and returns their experience.

    Parameters:
        simulator (OpenFOAMSimulator): supplies the case template, the solver
            script and the output directory.
        spec (PolicySpec): the contract, rendered into the case and checked
            against what the solver wrote.
        reward_fn (callable): maps the trajectory, with any requested
            functionObject output merged in, to one reward per control step.
        controller_keys (str or sequence of str): where the controller block is
            rendered, in 'folder__file__variable' form, e.g.
            'system__controlDict__controller'.
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

    def collect(
        self,
        artifact: PolicyArtifact,
        n_episodes: int = 1,
        *,
        seeds: Sequence[int] | None = None,
        iteration: int | None = None,
        n_jobs: int = 1,
        deterministic: bool = False,
    ) -> Rollout:
        """
        Run n_episodes with one frozen policy and return their experience.

        Parameters:
            artifact (PolicyArtifact): the exported policy, implementing the
                runner's spec.
            n_episodes (int): how many cases to run with this policy.
            seeds (sequence of int or None): one RNG seed per episode, written
                into the case and recorded in the trajectory. None derives them
                from the iteration.
            iteration (int or None): training iteration. The cases go to
                iter<iteration>_ep<index>; None names them after the policy file.
            n_jobs (int): how many cases to run at once, on threads.
            deterministic (bool): apply the mean action instead of a draw, to
                evaluate a policy.

        Returns:
            Rollout
        """
        if artifact.spec.hash != self.spec.hash:
            raise ValueError(
                f"the policy implements contract {artifact.spec.hash} but the "
                f"runner is configured for {self.spec.hash}"
            )
        if seeds is None:
            seeds = [(iteration or 0) * 100_000 + i for i in range(n_episodes)]
        if len(seeds) != n_episodes:
            raise ValueError(f"got {len(seeds)} seeds for {n_episodes} episodes")

        name = artifact.path.stem if iteration is None else f"iter{iteration:04d}"
        case_dirs = [self.simulator.output_path / f"{name}_ep{i:02d}" for i in range(n_episodes)]

        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            results = list(
                pool.map(
                    self._run_episode,
                    range(n_episodes),
                    case_dirs,
                    [int(seed) for seed in seeds],
                    repeat(artifact.path.resolve()),
                    repeat(deterministic),
                )
            )

        rollout = Rollout(artifact=artifact)
        for episode, failure in results:
            if episode is not None:
                rollout.episodes.append(episode)
            if failure is not None:
                rollout.failures.append(failure)

        if rollout.failures:
            logger.warning(
                "%d of %d episodes diverged: %s",
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
        case_dir: Path,
        seed: int,
        policy_path: Path,
        deterministic: bool,
    ) -> tuple[xr.Dataset | None, EpisodeFailure | None]:
        params = controller_params(
            self.spec,
            policy_path,
            self.controller_keys,
            seed=seed,
            deterministic=deterministic,
        )

        try:
            self.run_fn(case_dir, params)
        except SolverDivergedError as exc:
            logger.warning("Episode %d diverged in %s", index, case_dir)
            reason = f"solver exited with code {exc.returncode}"
            try:
                episode = self._read_episode(case_dir)
            except Exception as read_error:
                return None, EpisodeFailure(
                    index, case_dir, f"{reason}; no usable trajectory: {read_error}"
                )
            episode.attrs["diverged"] = True
            return episode, EpisodeFailure(index, case_dir, f"{reason}; partial kept")

        episode = self._read_episode(case_dir)
        episode.attrs["diverged"] = False
        return episode, None

    def _read_episode(self, case_dir: Path) -> xr.Dataset:
        trajectory = read_trajectory(case_dir, self.spec)
        series = [read_function_object(case_dir, name) for name in self.function_objects]
        data = attach(trajectory, *series)
        data = data.assign(reward=("time", evaluate_reward(data, self.reward_fn)))
        data.attrs["case_dir"] = str(case_dir)
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
