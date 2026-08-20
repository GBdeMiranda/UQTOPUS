"""
Shared fixtures and helpers for the intrusive RL tests.
"""

from __future__ import annotations

import re
from functools import partial

import numpy as np
import pytest

# sb3 pulls in torch, onnx and gymnasium, all imported eagerly by uqtopus.rl.
pytest.importorskip("stable_baselines3")

from uqtopus import OpenFOAMSimulator  # noqa: E402
from uqtopus.exceptions import SolverDivergedError  # noqa: E402
from uqtopus.rl import (  # noqa: E402
    ActionSpec,
    ObservationSpec,
    PolicySpec,
    ProbeSource,
    export_random_policy,
    write_trajectory,
)
from uqtopus.rl.runner import ClosedLoopRunner  # noqa: E402

N_STEPS = 8

PROBES = ProbeSource(field_name="p", positions=[(0.5, 0.0, 0.005), (1.0, 0.0, 0.005)])
JETS = ActionSpec(name="Q", targets={"jet1": 1.0, "jet2": -1.0}, low=-0.1, high=0.1)


def make_spec(
    sources=(PROBES,), stack=1, action=JETS, control_interval=0.4, start_time=0.0,
    end_time=None,
) -> PolicySpec:
    """Two probes in the wake driving a pair of jets with zero net flow."""
    return PolicySpec(
        observation=ObservationSpec(sources=sources, stack=stack),
        action=action,
        control_interval=control_interval,
        start_time=start_time,
        end_time=end_time,
    )


def write_force_coeffs(case_dir, start, times, cd, cl):
    """Stand in for OpenFOAM's forceCoeffs functionObject."""
    path = case_dir / "postProcessing" / "forceCoeffs" / str(start) / "coefficient.dat"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Force coefficients", "# dragDir       : (1 0 0)", "# Time\tCd\tCl"]
    lines += [f"{t:.6g}\t{d:.6g}\t{l:.6g}" for t, d, l in zip(times, cd, cl)]
    path.write_text("\n".join(lines) + "\n")
    return path


def run_fake_solver(case_dir, params, *, spec, n_steps, fail_after, write_coeffs):
    """Write the trajectory and coefficients the compiled controller would."""
    case_dir.mkdir(parents=True, exist_ok=True)
    block = next(v for k, v in params.items() if "controller" in k)
    assert spec.hash in block, "the case did not receive the expected contract"
    seed = int(re.search(r"^\s*seed\s+(-?\d+);", block, re.M).group(1))

    steps = n_steps if fail_after is None else fail_after
    rng = np.random.default_rng(seed)
    times = spec.control_interval * np.arange(1, steps + 1)
    # far from zero, so a normalization mistake shows up downstream
    obs = 5.0 + 2.0 * rng.normal(size=(steps, spec.obs_dim))
    act = rng.uniform(-0.1, 0.1, (steps, spec.act_dim))
    write_trajectory(
        case_dir / "postProcessing/uqtopusPolicy/0/trajectory.dat",
        spec, times, obs, act, seed=seed,
    )

    if write_coeffs:
        cfd = np.round(np.arange(1, steps * 40 + 1) * 0.01, 6)
        write_force_coeffs(case_dir, 0, cfd, np.full(cfd.size, 1.4), 0.1 * np.sin(cfd))

    if fail_after is not None:
        raise SolverDivergedError("diverged", returncode=1, stdout="", stderr="")


def fake_solver(spec, n_steps=N_STEPS, fail_after=None, write_coeffs=True):
    """run_fake_solver bound to one spec, as the run_fn(case_dir, params) callback."""
    return partial(
        run_fake_solver,
        spec=spec,
        n_steps=n_steps,
        fail_after=fail_after,
        write_coeffs=write_coeffs,
    )


def make_runner(simulator, spec, run_fn=None, **kwargs) -> ClosedLoopRunner:
    kwargs.setdefault("reward_fn", lambda ds: -np.abs(ds["action"].values).ravel())
    return ClosedLoopRunner(
        simulator=simulator,
        spec=spec,
        controller_keys="0__U__controller",
        run_fn=fake_solver(spec) if run_fn is None else run_fn,
        **kwargs,
    )


@pytest.fixture
def spec() -> PolicySpec:
    return make_spec()


@pytest.fixture
def artifact(spec, tmp_path):
    return export_random_policy(spec, tmp_path / "policy.onnx", iteration=0, seed=0)


@pytest.fixture
def simulator(tmp_path) -> OpenFOAMSimulator:
    template = tmp_path / "template"
    (template / "0").mkdir(parents=True)
    (template / "0" / "U").write_text("jet1\n{\n{{ controller }}\n}\n")
    (template / "Allrun").write_text("#!/bin/sh\n")
    return OpenFOAMSimulator(
        template_path=template,
        solver_script="Allrun",
        output_path=tmp_path / "runs",
        qoi_variables=["p"],
    )


@pytest.fixture
def runner(simulator, spec) -> ClosedLoopRunner:
    return make_runner(simulator, spec)
