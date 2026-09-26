"""
Tests for closed-loop rollout collection.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from uqtopus import OpenFOAMSimulator
from uqtopus.rl import export_random_policy
from uqtopus.rl.runner import ClosedLoopRunner

from conftest import N_STEPS, fake_solver, make_runner, make_spec


def recording_solver(rendered, spec, case_dir, params):
    """The fake solver, keeping each controller block it was given."""
    rendered.append(params["system__controlDict__controller"])
    fake_solver(spec)(case_dir, params)


def second_episode_fails(good, bad, case_dir, params):
    """Runs bad for the second episode and good for the others."""
    (bad if case_dir.name.endswith("ep01") else good)(case_dir, params)


def refusing_launcher(case_dir, params):
    raise OSError("scheduler refused the job")


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


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"iteration": 7}, [700000, 700001, 700002]),
        ({"seeds": [11, 22, 33]}, [11, 22, 33]),
    ],
)
def test_each_episode_gets_its_own_seed(coeff_runner, artifact, kwargs, expected):
    rollout = coeff_runner.collect(artifact, n_episodes=3, **kwargs)

    assert [ds.attrs["seed"] for ds in rollout.episodes] == expected
    # different seeds must produce different trajectories
    assert not np.allclose(
        rollout.episodes[0]["observation"], rollout.episodes[1]["observation"]
    )


def test_one_seed_per_episode_is_required(coeff_runner, artifact):
    with pytest.raises(ValueError, match="seeds"):
        coeff_runner.collect(artifact, n_episodes=3, seeds=[1, 2])


def test_deterministic_collection_reaches_the_case(simulator, spec, artifact):
    rendered = []
    runner = make_runner(simulator, spec, partial(recording_solver, rendered, spec))
    runner.collect(artifact, n_episodes=1)
    runner.collect(artifact, n_episodes=1, iteration=1, deterministic=True)

    assert "deterministic   no;" in rendered[0]
    assert "deterministic   yes;" in rendered[1]


def test_parallel_collection_matches_serial(coeff_runner, artifact):
    serial = coeff_runner.collect(artifact, n_episodes=4, n_jobs=1)
    parallel = coeff_runner.collect(artifact, n_episodes=4, n_jobs=4)

    assert np.allclose(serial.observations, parallel.observations)
    assert np.allclose(serial.rewards, parallel.rewards)


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
    runner = make_runner(simulator, spec, partial(second_episode_fails, good, bad))
    rollout = runner.collect(artifact, n_episodes=3)

    assert len(rollout.episodes) == 2
    assert len(rollout.failures) == 1


def test_a_launcher_error_is_raised(simulator, spec, artifact):
    runner = make_runner(simulator, spec, refusing_launcher)
    with pytest.raises(OSError, match="scheduler refused"):
        runner.collect(artifact, n_episodes=2)


# ---------------------------------------------------------------------------
# contract
# ---------------------------------------------------------------------------

def test_a_policy_from_another_contract_is_refused(coeff_runner, tmp_path):
    other = make_spec(control_interval=0.2)
    wrong = export_random_policy(other, tmp_path / "other.onnx", seed=0)

    with pytest.raises(ValueError, match="contract"):
        coeff_runner.collect(wrong, n_episodes=1)


# ---------------------------------------------------------------------------
# the real launch path
# ---------------------------------------------------------------------------

def test_the_block_reaches_the_case_through_jinja(tmp_path, spec, artifact):
    """The default run_fn: rsync the template, render controlDict, run the solver script."""
    template = tmp_path / "template"
    (template / "system").mkdir(parents=True)
    (template / "system" / "controlDict").write_text("{{ controller }}\n")

    # the "solver" copies the rendered dictionary aside and writes a trajectory
    steps = 4
    names = ["time", *spec.observation.component_names(), *spec.action.component_names()]
    columns = " ".join(names)
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
        "cp system/controlDict rendered_controlDict\n"
        "mkdir -p postProcessing/uqtopusPolicy/0\n"
        "{\n"
        f'  echo "# specHash {spec.hash}"\n'
        '  echo "# seed 99"\n'
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
        controller_keys="system__controlDict__controller",
    )

    rollout = runner.collect(artifact, n_episodes=1, seeds=[99])

    rendered = (tmp_path / "runs" / "policy_ep00" / "rendered_controlDict").read_text()
    assert spec.hash in rendered
    assert "seed            99;" in rendered
    assert str(artifact.path.resolve()) in rendered
    assert rendered.count("{") == rendered.count("}")
    assert rollout.n_steps == steps


