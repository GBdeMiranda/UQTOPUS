"""
Tests for the open-loop environment, where one step is one whole OpenFOAM run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("gymnasium")

from uqtopus.envs import OpenLoopEnv  # noqa: E402

DT_KEY = "constant__transportProperties__DT"
NU_KEY = "constant__transportProperties__nu"
RANGES = {DT_KEY: (0.1, 0.5), NU_KEY: (1.0, 2.0)}


class RecordingSimulator:
    """
    Stand-in for OpenFOAMSimulator that keeps what it was asked to run.

    Its run() returns a one-time-step dataset whose field is the sum of the
    parameters, so a test can read the parameters back out of the observation.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, float]] = []
        self.steps: list[int | None] = []
        self.output_path = Path(".")

    def reset(self) -> None:
        self.calls.clear()

    def run(self, params, step=None, verbose=False, cleanup=False) -> xr.Dataset:
        self.calls.append(dict(params))
        self.steps.append(step)
        return xr.Dataset(
            {"T": (("time", "cell"), np.full((1, 3), sum(params.values())))},
            coords={"time": [1.0]},
        )


def make_env(**kwargs) -> OpenLoopEnv:
    kwargs.setdefault("simulator", RecordingSimulator())
    kwargs.setdefault("param_ranges", RANGES)
    kwargs.setdefault("observation_fn", lambda ds: ds["T"].values)
    kwargs.setdefault("reward_fn", lambda ds: float(ds["T"].mean()))
    kwargs.setdefault("obs_shape", (3,))
    return OpenLoopEnv(**kwargs)


# ---------------------------------------------------------------------------
# the action is the case parameters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "action, expected",
    [
        ([0.3, 1.5], {DT_KEY: 0.3, NU_KEY: 1.5}),
        ([9.0, -9.0], {DT_KEY: 0.5, NU_KEY: 1.0}),      # clipped to the ranges
    ],
)
def test_the_action_becomes_the_parameters_in_order_and_inside_the_ranges(action, expected):
    env = make_env()
    env.reset(seed=0)
    env.step(np.array(action, dtype=np.float32))

    assert list(env.simulator.calls[-1]) == [DT_KEY, NU_KEY]
    assert env.simulator.calls[-1] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# the first run of an episode
# ---------------------------------------------------------------------------

def test_reset_samples_inside_the_ranges_and_repeats_for_the_same_seed():
    first, again = make_env(), make_env()
    first.reset(seed=7)
    again.reset(seed=7)

    sampled = first.simulator.calls[0]
    assert 0.1 <= sampled[DT_KEY] <= 0.5 and 1.0 <= sampled[NU_KEY] <= 2.0
    assert again.simulator.calls[0] == pytest.approx(sampled)

    other = make_env()
    other.reset(seed=8)
    assert other.simulator.calls[0] != pytest.approx(sampled)


def test_initial_params_fix_the_first_run():
    env = make_env(initial_params={DT_KEY: 0.25, NU_KEY: 1.75})
    env.reset(seed=0)

    assert env.simulator.calls[0] == pytest.approx({DT_KEY: 0.25, NU_KEY: 1.75})


def test_initial_params_missing_a_key_are_refused():
    env = make_env(initial_params={DT_KEY: 0.25})

    with pytest.raises(ValueError, match=NU_KEY):
        env.reset(seed=0)


# ---------------------------------------------------------------------------
# what comes back from a step
# ---------------------------------------------------------------------------

def test_the_observation_is_float32_with_the_time_of_one_run_dropped():
    env = make_env()
    first, _ = env.reset(seed=0)
    stepped, reward, _, _, _ = env.step(np.array([0.3, 1.5], dtype=np.float32))

    assert first.shape == stepped.shape == (3,)
    assert first.dtype == stepped.dtype == np.float32
    assert reward == pytest.approx(1.8)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"max_episode_steps": 2}, (False, True)),
        ({"terminated_fn": lambda ds, step: float(ds["T"].mean()) > 2.0}, (True, False)),
    ],
)
def test_the_episode_ends_at_the_step_limit_or_when_terminated_fn_says(kwargs, expected):
    env = make_env(**kwargs)
    env.reset(seed=0)

    _, _, terminated, truncated, _ = env.step(np.array([0.1, 1.0], dtype=np.float32))
    assert (terminated, truncated) == (False, False)

    _, _, terminated, truncated, _ = env.step(np.array([0.5, 2.0], dtype=np.float32))
    assert (terminated, truncated) == expected


def test_each_run_gets_its_own_directory_index():
    env = make_env()
    env.reset(seed=0)
    env.step(np.array([0.3, 1.5], dtype=np.float32))
    env.step(np.array([0.3, 1.5], dtype=np.float32))

    assert env.simulator.steps == [0, 1, 2]
