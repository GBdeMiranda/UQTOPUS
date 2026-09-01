"""
Tests for the solver -> Python channel: the trajectory log, the functionObject
output attached to it, and the reward evaluated on top.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from uqtopus.rl import (
    ProbeSource,
    TrajectoryError,
    align_to_control,
    attach,
    evaluate_reward,
    find_trajectory,
    moving_average,
    read_function_object,
    read_trajectory,
    write_trajectory,
)
from uqtopus.rl.trajectory import TRAJECTORY_NAME, TRAJECTORY_SUBDIR

from conftest import make_spec, write_force_coeffs


def make_episode(spec, n_steps: int = 10, seed: int = 7):
    rng = np.random.default_rng(seed)
    times = spec.start_time + spec.control_interval * np.arange(1, n_steps + 1)
    obs = rng.normal(size=(n_steps, spec.obs_dim))
    act = rng.uniform(spec.action.low[0], spec.action.high[0], (n_steps, spec.act_dim))
    return times, obs, act


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


def test_reads_without_a_spec_using_the_header(spec, tmp_path):
    times, obs, act = make_episode(spec)
    path = write_trajectory(tmp_path / "t.dat", spec, times, obs, act, seed=1)

    ds = read_trajectory(path)

    assert ds.sizes["time"] == len(times)
    assert np.allclose(ds["observation"].values, obs, atol=1e-9)


def test_contract_mismatch_is_caught(spec, tmp_path):
    times, obs, act = make_episode(spec)
    path = write_trajectory(tmp_path / "t.dat", spec, times, obs, act)

    other = make_spec(control_interval=0.02)
    with pytest.raises(TrajectoryError, match="contract"):
        read_trajectory(path, other)

    # non-strict downgrades it to a warning but still refuses on the shape
    narrower = make_spec(
        sources=(ProbeSource(field_name="p", positions=[(0.1, 0.0, 0.005)]),)
    )
    with pytest.raises(TrajectoryError, match="columns"):
        read_trajectory(path, narrower, strict=False)


def test_write_rejects_wrong_shapes(spec, tmp_path):
    times, obs, act = make_episode(spec)
    with pytest.raises(ValueError, match="observations"):
        write_trajectory(tmp_path / "t.dat", spec, times, obs[:, :-1], act)


def test_non_increasing_time_is_rejected(spec, tmp_path):
    times, obs, act = make_episode(spec)
    times[3] = times[2]
    path = tmp_path / "t.dat"
    # bypass the writer, which would not produce this
    columns = " ".join(spec.trajectory_columns())
    rows = [
        " ".join(f"{v:.10g}" for v in (t, *o, *a)) for t, o, a in zip(times, obs, act)
    ]
    path.write_text(f"# specHash {spec.hash}\n# columns {columns}\n" + "\n".join(rows))

    with pytest.raises(TrajectoryError, match="non-increasing"):
        read_trajectory(path, spec)


def test_non_finite_values_are_rejected(spec, tmp_path):
    times, obs, act = make_episode(spec)
    obs[4, 1] = np.nan
    path = write_trajectory(tmp_path / "t.dat", spec, times, obs, act)

    with pytest.raises(TrajectoryError, match="non-finite"):
        read_trajectory(path, spec)


def test_empty_file_is_rejected(spec, tmp_path):
    path = tmp_path / "t.dat"
    path.write_text("# specHash abc\n")
    with pytest.raises(TrajectoryError, match="no data rows"):
        read_trajectory(path, spec)


def test_find_trajectory_picks_the_latest_restart(spec, tmp_path):
    with pytest.raises(TrajectoryError, match="no postProcessing"):
        find_trajectory(tmp_path)

    times, obs, act = make_episode(spec)
    for start in ("0", "4", "8"):
        write_trajectory(
            tmp_path / TRAJECTORY_SUBDIR / start / TRAJECTORY_NAME, spec, times, obs, act
        )

    assert find_trajectory(tmp_path).parent.name == "8"
    # a case directory can be handed straight to the reader
    assert read_trajectory(tmp_path, spec).sizes["time"] == len(times)


def test_custom_reader_is_honored(spec, tmp_path):
    """A solver writing CSV instead of the canonical format needs no core change."""
    times, obs, act = make_episode(spec)
    path = tmp_path / "t.csv"
    rows = [",".join(f"{v:.10g}" for v in (t, *o, *a)) for t, o, a in zip(times, obs, act)]
    path.write_text("\n".join(rows))

    def csv_reader(p):
        table = np.array(
            [[float(v) for v in line.split(",")] for line in p.read_text().splitlines()]
        )
        return {"specHash": spec.hash}, table

    ds = read_trajectory(path, spec, reader=csv_reader)
    assert np.allclose(ds["action"].values, act, atol=1e-9)


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
    with pytest.raises(ValueError, match="no functionObject output"):
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

@pytest.mark.parametrize(
    "how, expected",
    [
        # (0.0, 0.4] holds samples 0.1..0.4 -> values 0,1,2,3 -> mean 1.5
        ("mean", [1.5, 5.5, 9.5]),
        ("last", [3.0, 7.0, 11.0]),
        ("interp", [3.0, 7.0, 11.0]),
    ],
)
def test_alignment_reduces_each_control_interval(how, expected):
    aligned = align_to_control(ramp(), np.array([0.4, 0.8, 1.2]), how=how)
    assert np.allclose(aligned["v"].values, expected)


def test_interp_samples_between_stored_instants():
    source = xr.Dataset(
        {"v": ("time", np.array([0.0, 10.0]))}, coords={"time": np.array([0.0, 1.0])}
    )
    aligned = align_to_control(source, np.array([0.25, 0.5]), how="interp")
    assert np.allclose(aligned["v"].values, [2.5, 5.0])


def test_empty_interval_warns_and_holds_the_previous_value(caplog):
    source = xr.Dataset({"v": ("time", np.array([5.0]))}, coords={"time": np.array([0.05])})
    aligned = align_to_control(source, np.array([0.1, 0.2, 0.3]), how="mean")

    assert np.allclose(aligned["v"].values, 5.0)
    assert "contain no samples" in caplog.text


def test_attach_merges_onto_the_trajectory(spec, tmp_path):
    n = 5
    times = spec.start_time + spec.control_interval * np.arange(1, n + 1)
    path = write_trajectory(
        tmp_path / "t.dat", spec, times, np.zeros((n, spec.obs_dim)), np.zeros((n, 1))
    )
    trajectory = read_trajectory(path, spec)

    cfd_times = np.round(np.arange(1, 201) * 0.01, 6)
    write_force_coeffs(tmp_path, 0, cfd_times, np.full(200, 1.4), np.zeros(200))

    merged = attach(trajectory, read_function_object(tmp_path, "forceCoeffs"))

    assert set(merged.data_vars) >= {"observation", "action", "Cd", "Cl"}
    assert merged.sizes["time"] == n
    assert np.allclose(merged["Cd"].values, 1.4)


# ---------------------------------------------------------------------------
# reward
# ---------------------------------------------------------------------------

def test_moving_average_is_causal_and_cancels_a_full_period():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.allclose(moving_average(values, 1), values)
    # leading steps average over what exists so far
    assert np.allclose(moving_average(values, 3), [1.0, 1.5, 2.0, 3.0, 4.0])

    t = np.arange(0, 100) * 0.1
    signal = 2.0 + np.sin(2 * np.pi * t / 1.0)   # period 1.0 == 10 samples
    assert np.allclose(moving_average(signal, 10)[20:], 2.0, atol=1e-2)


def test_evaluate_reward_checks_the_shape(spec, tmp_path):
    n = 6
    times = spec.control_interval * np.arange(1, n + 1)
    path = write_trajectory(
        tmp_path / "t.dat", spec, times, np.zeros((n, spec.obs_dim)), np.zeros((n, 1))
    )
    trajectory = read_trajectory(path, spec)

    assert evaluate_reward(trajectory, lambda ds: np.ones(n)).shape == (n,)

    with pytest.raises(ValueError, match="single value"):
        evaluate_reward(trajectory, lambda ds: 1.0)
    with pytest.raises(ValueError, match="returned 3 values"):
        evaluate_reward(trajectory, lambda ds: np.ones(3))
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_reward(trajectory, lambda ds: np.full(n, np.nan))
