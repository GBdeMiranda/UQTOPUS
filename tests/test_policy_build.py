"""
Checks around the packaged C++ sources and the discovery the CLI does.

Nothing here compiles anything: the parts that need OpenFOAM are exercised by
running `uqtopus rl-build` by hand.
"""

from __future__ import annotations

import pytest

from uqtopus import policy_build
from uqtopus.cli import build_parser


def test_the_cpp_sources_ship_with_the_package():
    sources = policy_build.policy_sources()

    assert (sources / "Make" / "files").exists()
    assert (sources / "Make" / "options").exists()
    assert sorted(p.name for p in sources.glob("*.C")) == [
        "onnxPolicy.C",
        "uqtopusBoundaryConditionFvPatchVectorField.C",
        "uqtopusController.C",
    ]


def test_make_files_names_the_library_the_cases_load():
    text = (policy_build.policy_sources() / "Make" / "files").read_text()

    assert "libuqtopusPolicy" in text


def test_an_incomplete_onnxruntime_is_not_accepted(tmp_path):
    (tmp_path / "include").mkdir()
    (tmp_path / "include" / "onnxruntime_cxx_api.h").touch()

    assert policy_build.find_onnxruntime(tmp_path) is None


def test_a_complete_onnxruntime_is_accepted(tmp_path):
    (tmp_path / "include").mkdir()
    (tmp_path / "include" / "onnxruntime_cxx_api.h").touch()
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "libonnxruntime.so").touch()

    assert policy_build.find_onnxruntime(tmp_path) == tmp_path


def test_a_missing_openfoam_is_reported(tmp_path):
    with pytest.raises(RuntimeError, match="no OpenFOAM bashrc"):
        policy_build.find_openfoam(tmp_path / "nowhere")


def test_rl_build_is_registered_as_a_subcommand():
    args = build_parser().parse_args(["rl-build", "--check"])

    assert args.check is True
    assert args.handler.__name__ == "rl_build"
