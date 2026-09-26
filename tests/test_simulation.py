"""
Tests for the OpenFOAM case runner and the parallel UQ ensemble.
"""

from __future__ import annotations

import importlib
import os
import inspect
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from uqtopus.exceptions import SolverDivergedError
from uqtopus.policy_build import find_openfoam
from uqtopus.sampler import generate_samples
from uqtopus.simulation import (
    _run_ensemble,
    _solver_environment,
    run_simulation,
    run_uq_study,
    uq_simulation,
)

pytestmark = pytest.mark.skipif(
    shutil.which("rsync") is None,
    reason="run_simulation copies the case template with rsync",
)

DT_KEY = "constant__transportProperties__DT"
CMU_KEY = "constant__momentumTransport__Cmu"
C1_KEY = "constant__momentumTransport__C1"

# The solver reads back the value the template engine wrote, so a rendering
# mistake shows up as a wrong number rather than as a passing test. A negative
# diffusivity stands in for a diverged run.
ALLRUN = """#!/bin/sh
DT=$(sed -n 's/^DT[[:space:]]*//p' constant/transportProperties | tr -d ';')
case "$DT" in
  -*) echo "solver diverged with DT=$DT" >&2; exit 1 ;;
esac
mkdir -p 1
printf '%s\\n' "$DT" > 1/T
"""


@pytest.fixture
def template(tmp_path) -> Path:
    """A minimal case: two rendered dictionaries and a stand-in for Allrun."""
    root = tmp_path / "template"
    (root / "constant").mkdir(parents=True)
    (root / "constant" / "transportProperties").write_text("DT  {{ DT }};\n")
    (root / "constant" / "momentumTransport").write_text(
        "Cmu  {{ Cmu }};\nC1  {{ C1 }};\n"
    )
    allrun = root / "Allrun"
    allrun.write_text(ALLRUN)
    allrun.chmod(0o755)
    return root


@pytest.fixture
def exp_config(template, tmp_path) -> dict:
    return {
        "input_path": str(template),
        "output_path": str(tmp_path / "run"),
        "solver": "Allrun",
    }


def solved_dt(case_dir: Path) -> float:
    """The value the solver read back out of the rendered case."""
    return float((case_dir / "1" / "T").read_text())


def _raise_no_openfoam(*args):
    raise RuntimeError("no OpenFOAM installation found")


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["lhs", "random"])
def test_generate_samples_is_reproducible_under_a_seed(method):
    ranges = {"a": [0.0, 1.0], "b": [10.0, 20.0]}

    first = generate_samples(5, ranges, method=method, seed=42)
    again = generate_samples(5, ranges, method=method, seed=42)
    other = generate_samples(5, ranges, method=method, seed=7)

    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


@pytest.mark.parametrize("n_samples", [1, 8])
def test_generate_samples_stays_inside_the_ranges(n_samples):
    ranges = {"a": [0.0, 1.0], "b": [10.0, 20.0]}

    samples = generate_samples(n_samples, ranges, method="lhs", seed=0)

    assert samples.shape == (n_samples, 2)
    assert ((samples[:, 0] >= 0.0) & (samples[:, 0] <= 1.0)).all()
    assert ((samples[:, 1] >= 10.0) & (samples[:, 1] <= 20.0)).all()


# ---------------------------------------------------------------------------
# single case
# ---------------------------------------------------------------------------

def test_run_simulation_renders_the_template_and_runs_the_solver(exp_config, tmp_path):
    run_simulation({DT_KEY: 0.05}, exp_config)

    case = tmp_path / "run"
    # the renderer runs with keep_trailing_newline=False, so the final newline goes
    assert (case / "constant" / "transportProperties").read_text() == "DT  0.05;"
    assert solved_dt(case) == 0.05


def test_run_simulation_renders_params_sharing_a_file_in_one_pass(exp_config, tmp_path):
    run_simulation({DT_KEY: 0.05, CMU_KEY: 0.09, C1_KEY: 1.44}, exp_config)

    rendered = (tmp_path / "run" / "constant" / "momentumTransport").read_text()
    assert rendered == "Cmu  0.09;\nC1  1.44;"


@pytest.mark.parametrize(
    "params, match",
    [
        ({"DT": 0.05}, "folder__filename__paramname"),
        ({}, "params must"),
        ([DT_KEY], "params must"),
    ],
)
def test_run_simulation_rejects_malformed_params(exp_config, params, match):
    with pytest.raises(ValueError, match=match):
        run_simulation(params, exp_config)


def test_run_simulation_raises_solver_diverged_with_the_solver_output(exp_config):
    with pytest.raises(SolverDivergedError) as excinfo:
        run_simulation({DT_KEY: -1.0}, exp_config)

    assert excinfo.value.returncode == 1
    assert "diverged" in excinfo.value.stderr


def test_run_simulation_overwrites_an_existing_case(exp_config, tmp_path):
    run_simulation({DT_KEY: 0.05}, exp_config)
    stale = tmp_path / "run" / "stale.txt"
    stale.write_text("left over from the previous run")

    run_simulation({DT_KEY: 0.07}, exp_config)

    assert solved_dt(tmp_path / "run") == 0.07
    assert not stale.exists(), "rsync --delete should clear the previous run"


def test_solver_environment_is_inherited_under_a_sourced_openfoam(monkeypatch):
    monkeypatch.setenv("WM_PROJECT_DIR", "/opt/openfoam9")

    assert _solver_environment() is None


def test_solver_environment_comes_from_the_installation_when_the_shell_lacks_it(monkeypatch):
    try:
        find_openfoam()
    except RuntimeError:
        pytest.skip("no OpenFOAM installation on this machine")
    monkeypatch.delenv("WM_PROJECT_DIR", raising=False)

    environment = _solver_environment()

    assert environment["FOAM_APPBIN"] in environment["PATH"].split(os.pathsep)


def test_solver_environment_is_inherited_with_no_openfoam_installed(monkeypatch):
    monkeypatch.delenv("WM_PROJECT_DIR", raising=False)
    monkeypatch.setattr("uqtopus.simulation.find_openfoam", _raise_no_openfoam)

    assert _solver_environment() is None


def test_run_simulation_runs_with_no_openfoam_in_the_environment(exp_config, tmp_path, monkeypatch):
    monkeypatch.delenv("WM_PROJECT_DIR", raising=False)

    run_simulation({DT_KEY: 0.05}, exp_config)

    assert solved_dt(tmp_path / "run") == 0.05


# ---------------------------------------------------------------------------
# ensemble
# ---------------------------------------------------------------------------

def test_run_ensemble_gives_each_sample_its_own_row_and_directory(exp_config, tmp_path):
    X = np.array([[0.1], [0.2], [0.3], [0.4]])

    failures = _run_ensemble(X, [DT_KEY], exp_config, nthreads=2)

    assert failures == []
    out = tmp_path / "run"
    assert sorted(p.name for p in out.iterdir()) == [
        "sample_0000", "sample_0001", "sample_0002", "sample_0003",
    ]
    # imap_unordered must not scramble which row landed in which directory
    for i, expected in enumerate(X.ravel()):
        assert solved_dt(out / f"sample_{i:04d}") == pytest.approx(expected)


def test_run_ensemble_records_failures_without_stopping_the_study(exp_config, tmp_path):
    X = np.array([[0.1], [-1.0], [0.3], [-2.0]])

    failures = _run_ensemble(X, [DT_KEY], exp_config, nthreads=2)

    assert [i for i, _ in failures] == [1, 3]
    assert all("SolverDivergedError" in error for _, error in failures)
    # the healthy samples still ran to completion
    assert solved_dt(tmp_path / "run" / "sample_0000") == pytest.approx(0.1)
    assert solved_dt(tmp_path / "run" / "sample_0002") == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# UQ[py]Lab entry point
# ---------------------------------------------------------------------------

def test_uq_simulation_is_addressable_as_a_uqlab_model():
    """
    UQ[py]Lab resolves 'ModelFun': 'uqtopus.uq_simulation' as a dotted path and
    calls it as f(X, Parameters). Anything that changes this breaks example 03.
    """
    module_name, _, attr = "uqtopus.uq_simulation".rpartition(".")
    func = getattr(importlib.import_module(module_name), attr)

    assert inspect.isfunction(func), "a bound method is not reachable by dotted path"
    params = list(inspect.signature(func).parameters.values())
    assert [p.name for p in params] == ["X", "Params"]
    assert all(p.kind is p.POSITIONAL_OR_KEYWORD for p in params)
    assert all(p.default is p.empty for p in params)


def test_uq_simulation_runs_the_design_and_returns_none(template, tmp_path):
    """A return value would be read back by UQLab as the model response Y."""
    params = {
        "input_path": str(template),
        "output_path": str(tmp_path / "study"),
        "parameter_ranges": {DT_KEY: [0.01, 0.3]},
        "solver": "Allrun",
        "nthreads": 1,
    }

    assert uq_simulation(np.array([[0.1], [0.2]]), params) is None

    assert solved_dt(tmp_path / "study" / "sample_0000") == pytest.approx(0.1)
    assert solved_dt(tmp_path / "study" / "sample_0001") == pytest.approx(0.2)


@pytest.mark.parametrize(
    "change, X, match",
    [
        ({"qoi_variables": ["T"]}, [[0.1]], "Unknown key 'qoi_variables'"),
        ({}, [[0.1, 0.2]], "number of sampled parameters"),
        ({"input_path": "/nonexistent/template"}, [[0.1]], "does not exist"),
    ],
)
def test_uq_simulation_rejects_a_bad_call(template, change, X, match):
    params = {
        "input_path": str(template),
        "parameter_ranges": {DT_KEY: [0.01, 0.3]},
        "solver": "Allrun",
        **change,
    }

    with pytest.raises(Exception, match=match):
        uq_simulation(np.array(X), params)


# ---------------------------------------------------------------------------
# config-driven entry point
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("as_type", [str, Path])
def test_run_uq_study_samples_the_ranges_from_the_config(template, tmp_path, as_type):
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.safe_dump({
        "input_path": str(template),
        "output_path": str(tmp_path / "study"),
        "parameter_ranges": {DT_KEY: [0.01, 0.3]},
        "solver": "Allrun",
        "nthreads": 1,
    }))

    run_uq_study(as_type(config_file), n_samples=3)

    out = tmp_path / "study"
    assert sorted(p.name for p in out.iterdir()) == [
        "sample_0000", "sample_0001", "sample_0002",
    ]
    for i in range(3):
        assert 0.01 <= solved_dt(out / f"sample_{i:04d}") <= 0.3


def test_run_uq_study_requires_the_mandatory_keys(tmp_path):
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.safe_dump({"solver": "Allrun"}))

    with pytest.raises(ValueError, match="must be provided"):
        run_uq_study(config_file, n_samples=2)
