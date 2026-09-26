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
        "uqtopusBoundaryConditionFvPatchField.C",
        "uqtopusBoundaryConditionFvPatchFields.C",
        "uqtopusController.C",
        "uqtopusPolicyFunction1.C",
        "uqtopusSource.C",
    ]


def test_make_files_names_the_library_the_cases_load():
    text = (policy_build.policy_sources() / "Make" / "files").read_text()

    assert "libuqtopusPolicy" in text


@pytest.mark.parametrize(
    "files, found",
    [
        (["include/onnxruntime_cxx_api.h"], False),
        (["include/onnxruntime_cxx_api.h", "lib/libonnxruntime.so"], True),
    ],
)
def test_onnxruntime_needs_the_header_and_the_library(tmp_path, files, found):
    for name in files:
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).touch()

    assert (policy_build.find_onnxruntime(tmp_path) == tmp_path) is found


def test_a_missing_openfoam_is_reported(tmp_path):
    with pytest.raises(RuntimeError, match="no OpenFOAM bashrc"):
        policy_build.find_openfoam(tmp_path / "nowhere")


def test_rl_build_is_registered_as_a_subcommand():
    args = build_parser().parse_args(["rl-build", "--check"])

    assert args.check is True
    assert args.handler.__name__ == "rl_build"
