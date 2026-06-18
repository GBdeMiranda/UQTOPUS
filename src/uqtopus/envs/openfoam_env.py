"""
OpenFOAM Gymnasium Environment

Wraps OpenFOAMSimulator as a gymnasium.Env for deep RL training.
Each step() corresponds to one full OpenFOAM simulation run.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import xarray as xr

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError(
        "gymnasium is required for OpenFOAMEnv. "
        "Install it with: pip install uqtopus[rl]"
    ) from exc

from ..simulation import OpenFOAMSimulator

logger = logging.getLogger(__name__)


class OpenFOAMEnv(gym.Env):
    """
    Gymnasium environment wrapping an OpenFOAM simulation.

    Parameters:
        simulator (OpenFOAMSimulator)
            The bridge object that manages simulation execution and result parsing.

        param_ranges (dict)
            Ordered mapping of parameter keys to (min, max) physical bounds.
            Keys must use the 'folder__filename__paramname' encoding.
            Defines the action space.

        observation_fn (callable)
            Extracts the observation array from an xr.Dataset.
            Must return a numpy array with shape matching obs_shape.

        reward_fn (callable)
            Computes the scalar reward from an xr.Dataset.

        obs_shape (tuple)
            Shape of the array returned by observation_fn.
            Required so gymnasium can define the observation space at init time,
            before any simulation runs.

        max_episode_steps (int)
            Maximum number of steps per episode. When reached, truncated=True
            is returned. Default: 50.

        terminated_fn (callable or None)
            Optional early termination condition.
            Signature: (dataset: xr.Dataset, step: int) -> bool
            Return True to end the episode early with terminated=True.
            If None, episodes end only via max_episode_steps.

        initial_params (dict or None)
            Fixed starting parameters for every reset(). Keys must match
            param_ranges. If None, parameters are sampled uniformly from
            param_ranges on each reset.

        obs_bounds (tuple)
            (low, high) bounds for the observation space. Default: (-inf, +inf).

        verbose (bool)
            Forward verbose output to the simulator.
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        simulator: OpenFOAMSimulator,
        param_ranges: dict[str, tuple[float, float]],
        observation_fn: Callable[[xr.Dataset], np.ndarray],
        reward_fn: Callable[[xr.Dataset], float],
        obs_shape: tuple[int, ...],
        max_episode_steps: int = 50,
        terminated_fn: Callable[[xr.Dataset, int], bool] | None = None,
        initial_params: dict[str, float] | None = None,
        obs_bounds: tuple[float, float] = (-np.inf, np.inf),
        verbose: bool = False,
    ) -> None:
        super().__init__()

        self.simulator = simulator
        self.param_ranges = param_ranges
        self.param_keys = list(param_ranges.keys())
        self.observation_fn = observation_fn
        self.reward_fn = reward_fn
        self.max_episode_steps = max_episode_steps
        self.terminated_fn = terminated_fn
        self.initial_params = initial_params
        self.verbose = verbose

        self._param_lows = np.array(
            [v[0] for v in param_ranges.values()], dtype=np.float64
        )
        self._param_highs = np.array(
            [v[1] for v in param_ranges.values()], dtype=np.float64
        )

        # Action space uses actual physical parameter ranges.
        # If normalization is needed, use gymnasium.wrappers.RescaleAction.
        self.action_space = spaces.Box(
            low=self._param_lows.astype(np.float32),
            high=self._param_highs.astype(np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=obs_bounds[0],
            high=obs_bounds[1],
            shape=obs_shape,
            dtype=np.float32,
        )

        self._current_step: int = 0
        self._last_dataset: xr.Dataset | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Start a new episode.

        Resets the simulator run counter and runs one simulation with the
        initial parameters to produce the first observation.

        Returns (obs, info) where info contains 'params' and 'step'.
        """
        super().reset(seed=seed)
        self.simulator.reset()
        self._current_step = 0

        params = self._initial_params_dict(seed)
        dataset = self.simulator.run(params, verbose=self.verbose)
        self._last_dataset = dataset

        obs = np.array(self.observation_fn(dataset), dtype=np.float32)
        info = {"params": params, "step": 0, "dataset": dataset}
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one simulation step.

        Parameters:
            action (np.ndarray)
                Parameter values in the physical ranges defined by param_ranges.

        Returns (obs, reward, terminated, truncated, info).
        Info contains 'params', 'step', and 'dataset' (xr.Dataset).
        """
        self._current_step += 1

        params = self._action_to_params(action)
        dataset = self.simulator.run(
            params, step=self._current_step, verbose=self.verbose
        )
        self._last_dataset = dataset

        obs = np.array(self.observation_fn(dataset), dtype=np.float32)
        reward = float(self.reward_fn(dataset))

        truncated = self._current_step >= self.max_episode_steps
        terminated = (
            bool(self.terminated_fn(dataset, self._current_step))
            if self.terminated_fn is not None
            else False
        )

        info = {
            "params": params,
            "step": self._current_step,
            "dataset": dataset,
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> None:  # type: ignore[override]
        pass

    def _action_to_params(self, action: np.ndarray) -> dict[str, float]:
        action = np.clip(
            np.asarray(action, dtype=np.float64),
            self._param_lows,
            self._param_highs,
        )
        return dict(zip(self.param_keys, action.tolist()))

    def _initial_params_dict(self, seed: int | None) -> dict[str, float]:
        if self.initial_params is not None:
            missing = set(self.param_keys) - set(self.initial_params)
            if missing:
                raise ValueError(f"initial_params is missing keys: {missing}")
            return {k: self.initial_params[k] for k in self.param_keys}

        rng = np.random.default_rng(seed)
        values = rng.uniform(self._param_lows, self._param_highs)
        return dict(zip(self.param_keys, values.tolist()))

    def __repr__(self) -> str:
        return (
            f"OpenFOAMEnv("
            f"params={self.param_keys}, "
            f"obs_shape={self.observation_space.shape}, "
            f"max_steps={self.max_episode_steps})"
        )
