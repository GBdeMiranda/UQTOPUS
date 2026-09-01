"""
Tests for closed-loop rollout collection.
"""

from __future__ import annotations

import numpy as np
import pytest

from uqtopus import OpenFOAMSimulator
from uqtopus.rl import export_random_policy
from uqtopus.rl.runner import ClosedLoopRunner

from conftest import N_STEPS, fake_solver, make_runner, make_spec


@pytest.fixture
def coeff_runner(simulator, spec) -> ClosedLoopRunner:
    """A runner rewarding on lift, so the functionObject path is exercised."""
    return make_runner(
        simulator,
        spec,
        fake_solver(spec),
        reward_fn=lambda ds: -np.abs(ds["Cl"].values),
        function_objects=["forceCoeffs"],
    )


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------

def test_flat_views_line_up_with_episode_boundaries(coeff_runner, spec, artifact):
    rollout = coeff_runner.collect(artifact, n_episodes=3)

    assert len(rollout.episodes) == 3
    assert rollout.n_steps == 3 * N_STEPS
    assert rollout.observations.shape == (3 * N_STEPS, spec.obs_dim)
    assert rollout.actions.shape == (3 * N_STEPS, spec.act_dim)
    assert rollout.rewards.shape == (3 * N_STEPS,)
    assert rollout.returns.shape == (3,)
    assert np.flatnonzero(rollout.episode_starts).tolist() == [0, N_STEPS, 2 * N_STEPS]
    assert not rollout.failures


def test_each_episode_gets_its_own_seed(coeff_runner, artifact):
    rollout = coeff_runner.collect(artifact, n_episodes=3, iteration=7)

    assert [ds.attrs["seed"] for ds in rollout.episodes] == [700000, 700001, 700002]
    # different seeds must produce different trajectories
    assert not np.allclose(
        rollout.episodes[0]["observation"], rollout.episodes[1]["observation"]
    )


def test_explicit_seeds_are_honored(coeff_runner, artifact):
    rollout = coeff_runner.collect(artifact, n_episodes=2, seeds=[11, 22])
    assert [ds.attrs["seed"] for ds in rollout.episodes] == [11, 22]

    with pytest.raises(ValueError, match="seeds"):
        coeff_runner.collect(artifact, n_episodes=3, seeds=[1, 2])


def test_parallel_collection_matches_serial(coeff_runner, artifact):
    serial = coeff_runner.collect(artifact, n_episodes=4, n_jobs=1)
    parallel = coeff_runner.collect(artifact, n_episodes=4, n_jobs=4)

    assert np.allclose(serial.observations, parallel.observations)
    assert np.allclose(serial.rewards, parallel.rewards)


def test_the_reward_sees_the_function_object_output(simulator, spec, artifact):
    seen = {}

    def reward_fn(ds):
        seen["vars"] = set(ds.data_vars)
        return np.full(ds.sizes["time"], 0.5)

    runner = make_runner(
        simulator,
        spec,
        fake_solver(spec),
        reward_fn=reward_fn,
        function_objects=["forceCoeffs"],
    )
    rollout = runner.collect(artifact, n_episodes=1)

    assert {"Cd", "Cl", "observation", "action"} <= seen["vars"]
    assert np.allclose(rollout.rewards, 0.5)


# ---------------------------------------------------------------------------
# failures
# ---------------------------------------------------------------------------

def test_a_diverged_run_keeps_its_partial_trajectory(simulator, spec, artifact):
    runner = make_runner(simulator, spec, fake_solver(spec, fail_after=5))
    rollout = runner.collect(artifact, n_episodes=1)

    assert rollout.n_steps == 5
    assert rollout.episodes[0].attrs["diverged"] is True
    assert len(rollout.failures) == 1
    assert "partial kept" in str(rollout.failures[0])


def test_one_bad_case_does_not_sink_the_batch(simulator, spec, artifact):
    good = fake_solver(spec)
    bad = fake_solver(spec, fail_after=0, write_coeffs=False)
    calls = {"n": 0}

    def run_fn(case_dir, params):
        calls["n"] += 1
        (bad if calls["n"] == 2 else good)(case_dir, params)

    rollout = make_runner(simulator, spec, run_fn).collect(artifact, n_episodes=3)

    assert len(rollout.episodes) == 2
    assert len(rollout.failures) == 1


def test_a_launcher_error_is_reported_not_raised(simulator, spec, artifact):
    def run_fn(case_dir, params):
        raise OSError("scheduler refused the job")

    runner = make_runner(simulator, spec, run_fn)
    with pytest.raises(RuntimeError, match="all 2 episodes failed"):
        runner.collect(artifact, n_episodes=2)


# ---------------------------------------------------------------------------
# contract and gym vocabulary
# ---------------------------------------------------------------------------

def test_a_policy_from_another_contract_is_refused(coeff_runner, tmp_path):
    other = make_spec(control_interval=0.2)
    wrong = export_random_policy(other, tmp_path / "other.onnx", seed=0)

    with pytest.raises(ValueError, match="contract"):
        coeff_runner.collect(wrong, n_episodes=1)


def test_spaces_come_from_the_spec_and_the_stub_env_refuses_to_step(coeff_runner, spec):
    assert coeff_runner.observation_space.shape == (spec.obs_dim,)
    assert coeff_runner.action_space.shape == (spec.act_dim,)
    assert np.allclose(coeff_runner.action_space.low, -0.1)
    assert np.allclose(coeff_runner.action_space.high, 0.1)

    env = coeff_runner.stub_env()
    obs, info = env.reset()
    assert obs.shape == (spec.obs_dim,)
    assert env.observation_space.shape == (spec.obs_dim,)

    with pytest.raises(NotImplementedError, match="carries only the spaces"):
        env.step(np.zeros(spec.act_dim, dtype=np.float32))


# ---------------------------------------------------------------------------
# the real launch path
# ---------------------------------------------------------------------------

def test_the_block_reaches_the_case_through_jinja(tmp_path, spec, artifact):
    """The default run_fn: rsync the template, render 0/U, run the solver script."""
    template = tmp_path / "template"
    (template / "0").mkdir(parents=True)
    (template / "0" / "U").write_text(
        "boundaryField\n{\n    jet1\n    {\n{{ controller }}\n    }\n}\n"
    )

    # the "solver" copies the rendered dictionary aside and writes a trajectory
    steps = 4
    columns = " ".join(spec.trajectory_columns()[:-1])
    rows = [
        " ".join(
            [f"{spec.control_interval * (k + 1):g}"]
            + ["0.1"] * spec.obs_dim
            + ["0.05"] * spec.act_dim
        )
        for k in range(steps)
    ]
    allrun = template / "Allrun"
    allrun.write_text(
        "#!/bin/sh\n"
        "set -e\n"
        "cp 0/U rendered_U\n"
        "mkdir -p postProcessing/uqtopusPolicy/0\n"
        "{\n"
        f'  echo "# specHash {spec.hash}"\n'
        f'  echo "# columns {columns}"\n'
        + "".join(f'  echo "{row}"\n' for row in rows)
        + "} > postProcessing/uqtopusPolicy/0/trajectory.dat\n"
    )
    allrun.chmod(0o755)

    simulator = OpenFOAMSimulator(
        template_path=template,
        solver_script="Allrun",
        output_path=tmp_path / "runs",
        qoi_variables=["p"],
    )
    runner = ClosedLoopRunner(
        simulator=simulator,
        spec=spec,
        reward_fn=lambda ds: np.zeros(ds.sizes["time"]),
        controller_keys="0__U__controller",
    )

    rollout = runner.collect(artifact, n_episodes=1, seeds=[99])

    rendered = (tmp_path / "runs" / "iter0000_ep00" / "rendered_U").read_text()
    assert spec.hash in rendered
    assert "seed            99;" in rendered
    assert str(artifact.path.resolve()) in rendered
    assert rendered.count("{") == rendered.count("}")
    assert rollout.n_steps == steps
