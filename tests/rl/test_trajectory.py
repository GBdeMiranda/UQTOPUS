"""
Tests for the solver -> Python channel: the trajectory log, the functionObject
output attached to it, and the reward evaluated on top.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from uqtopus.rl import align_to_control, read_function_object, read_trajectory
from uqtopus.rl.reward import attach, evaluate_reward

from conftest import make_spec, write_force_coeffs, write_trajectory


def make_episode(spec, n_steps: int = 10, seed: int = 7):
    rng = np.random.default_rng(seed)
    times = spec.start_time + spec.control_interval * np.arange(1, n_steps + 1)
    obs = rng.normal(size=(n_steps, spec.obs_dim))
    act = rng.uniform(spec.action.low[0], spec.action.high[0], (n_steps, spec.act_dim))
    return times, obs, act


def zero_trajectory(spec, path, n: int) -> xr.Dataset:
    """An episode of n control steps with zero observations and actions, read back."""
    times = spec.start_time + spec.control_interval * np.arange(1, n + 1)
    write_trajectory(path, spec, times, np.zeros((n, spec.obs_dim)), np.zeros((n, spec.act_dim)))
    return read_trajectory(path, spec)


def ramp(n: int = 12, dt: float = 0.1) -> xr.Dataset:
    """A signal that ramps 0, 1, 2, ... one sample every dt."""
    return xr.Dataset(
        {"v": ("time", np.arange(n, dtype=float))},
        coords={"time": np.round(np.arange(n) * dt + dt, 6)},
    )


# ---------------------------------------------------------------------------
# the trajectory file
# ---------------------------------------------------------------------------

def test_round_trip_preserves_values_and_names(spec, tmp_path):
    times, obs, act = make_episode(spec)
    path = write_trajectory(tmp_path / "trajectory.dat", spec, times, obs, act, seed=123)

    ds = read_trajectory(path, spec)

    assert ds.sizes["time"] == len(times)
    assert np.allclose(ds["observation"].values, obs, atol=1e-9)
    assert np.allclose(ds["action"].values, act, atol=1e-9)
    assert np.allclose(ds["time"].values, times)
    assert ds.attrs["seed"] == 123
    assert ds.attrs["spec_hash"] == spec.hash
    assert list(ds["obs_component"].values) == spec.observation.component_names()
    assert list(ds["act_component"].values) == spec.action.component_names()


def test_a_trajectory_from_another_contract_is_refused(spec, tmp_path):
    times, obs, act = make_episode(spec)
    path = write_trajectory(tmp_path / "t.dat", spec, times, obs, act)

    with pytest.raises(ValueError, match="contract"):
        read_trajectory(path, make_spec(control_interval=0.02))


def test_non_finite_values_are_rejected(spec, tmp_path):
    times, obs, act = make_episode(spec)
    obs[4, 1] = np.nan
    path = write_trajectory(tmp_path / "t.dat", spec, times, obs, act)

    with pytest.raises(ValueError, match="non-finite"):
        read_trajectory(path, spec)


def test_empty_file_is_rejected(spec, tmp_path):
    path = write_trajectory(tmp_path / "t.dat", spec, [], [], [])
    with pytest.raises(ValueError, match="no control steps"):
        read_trajectory(path, spec)


def test_a_case_directory_reads_the_latest_restart(spec, tmp_path):
    with pytest.raises(ValueError, match="no postProcessing"):
        read_trajectory(tmp_path, spec)

    times, obs, act = make_episode(spec)
    for start in ("0", "4", "8"):
        write_trajectory(
            tmp_path / "postProcessing/uqtopusPolicy" / start / "trajectory.dat",
            spec, times, obs, act,
        )

    ds = read_trajectory(tmp_path, spec)
    assert Path(ds.attrs["source"]).parent.name == "8"
    assert ds.sizes["time"] == len(times)


# ---------------------------------------------------------------------------
# functionObject output
# ---------------------------------------------------------------------------

def test_reads_columns_from_the_header(tmp_path):
    times = np.arange(0.0, 1.0, 0.01)
    write_force_coeffs(tmp_path, 0, times, np.full_like(times, 1.4), np.zeros_like(times))

    ds = read_function_object(tmp_path, "forceCoeffs")

    assert set(ds.data_vars) == {"Cd", "Cl"}
    assert ds.sizes["time"] == len(times)
    assert np.allclose(ds["Cd"].values, 1.4)


def test_restarts_are_merged_with_the_later_run_winning(tmp_path):
    first = np.arange(0.0, 1.0, 0.1)
    write_force_coeffs(tmp_path, 0, first, np.ones_like(first), np.zeros_like(first))
    second = np.arange(0.5, 1.5, 0.1)
    write_force_coeffs(tmp_path, "0.5", second, np.full_like(second, 2.0), np.zeros_like(second))

    ds = read_function_object(tmp_path, "forceCoeffs")

    assert float(ds["Cd"].sel(time=0.2)) == 1.0   # only in the first run
    assert float(ds["Cd"].sel(time=0.7)) == 2.0   # overlapping: the restart wins
    assert float(ds["Cd"].sel(time=1.2)) == 2.0   # only in the restart
    assert np.all(np.diff(ds["time"].values) > 0)


def test_reader_failures_name_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="forceCoeffs"):
        read_function_object(tmp_path, "forceCoeffs")

    times = np.arange(0.0, 0.1, 0.01)
    write_force_coeffs(tmp_path, 0, times, np.ones_like(times), np.zeros_like(times))
    (tmp_path / "postProcessing" / "forceCoeffs" / "0" / "other.dat").write_text("# Time\tx\n0 1\n")

    with pytest.raises(ValueError, match="pass file="):
        read_function_object(tmp_path, "forceCoeffs")

    assert set(read_function_object(tmp_path, "forceCoeffs", file="other.dat").data_vars) == {"x"}


# ---------------------------------------------------------------------------
# alignment onto the control grid
# ---------------------------------------------------------------------------

def test_alignment_averages_each_control_interval():
    # (0.0, 0.4] holds samples 0.1..0.4 -> values 0,1,2,3 -> mean 1.5
    aligned = align_to_control(ramp(), np.array([0.4, 0.8, 1.2]))
    assert np.allclose(aligned["v"].values, [1.5, 5.5, 9.5])


def test_empty_interval_warns_and_holds_the_previous_value(caplog):
    source = xr.Dataset({"v": ("time", np.array([5.0]))}, coords={"time": np.array([0.05])})
    aligned = align_to_control(source, np.array([0.1, 0.2, 0.3]))

    assert np.allclose(aligned["v"].values, 5.0)
    assert "contain no samples" in caplog.text


def test_attach_merges_onto_the_trajectory(spec, tmp_path):
    trajectory = zero_trajectory(spec, tmp_path / "t.dat", 5)

    cfd_times = np.round(np.arange(1, 201) * 0.01, 6)
    write_force_coeffs(tmp_path, 0, cfd_times, np.full(200, 1.4), np.zeros(200))

    merged = attach(trajectory, read_function_object(tmp_path, "forceCoeffs"))

    assert set(merged.data_vars) >= {"observation", "action", "Cd", "Cl"}
    assert merged.sizes["time"] == 5
    assert np.allclose(merged["Cd"].values, 1.4)


# ---------------------------------------------------------------------------
# reward
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "reward_fn, match",
    [
        (lambda ds: 1.0, "returned 1 values"),
        (lambda ds: np.ones(3), "returned 3 values"),
        (lambda ds: np.full(ds.sizes["time"], np.nan), "non-finite"),
    ],
)
def test_evaluate_reward_wants_one_finite_value_per_step(spec, tmp_path, reward_fn, match):
    trajectory = zero_trajectory(spec, tmp_path / "t.dat", 6)

    with pytest.raises(ValueError, match=match):
        evaluate_reward(trajectory, reward_fn)
