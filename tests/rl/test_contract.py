"""
Tests for the policy contract: the spec, the ONNX artifact carrying it, and the
OpenFOAM dictionary the solver reads it from.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import onnxruntime as ort
import pytest
import torch

from uqtopus.rl import (
    ActionSpec,
    CustomSource,
    ForceCoeffSource,
    Normalization,
    ObservationSpec,
    PatchSource,
    PolicyArtifact,
    PolicySpec,
    ProbeSource,
    build_mlp,
    export_policy,
    export_random_policy,
    read_metadata,
    validate_export,
    validate_policy,
)
from uqtopus.rl.foam import (
    controller_mapping,
    controller_params,
    format_block,
    render_controller,
)

from conftest import PROBES, make_spec


def an_action(**overrides):
    """A deferred ActionSpec call, valid except for the given fields."""
    fields = {"name": "a", "targets": "t", "low": -1.0, "high": 1.0, **overrides}
    return partial(ActionSpec, **fields)


@pytest.fixture
def rich() -> PolicySpec:
    """Probes plus force coefficients, driving a ramped pair of jets."""
    return make_spec(
        sources=(
            ProbeSource(
                field_name="p", positions=[(0.55, -0.6, 0.005), (2.0, 0.0, 0.005)]
            ),
            ForceCoeffSource(patch="cylinder", coefficients=("Cd", "Cl")),
        ),
        action=ActionSpec(
            name="Q",
            targets={"jet1": 1.0, "jet2": -1.0},
            low=-0.1,
            high=0.1,
            ramp_fraction=0.5,
        ),
        start_time=4.0,
    )


@pytest.fixture
def beta() -> PolicySpec:
    """A single rotating cylinder with a Beta-distributed action."""
    return make_spec(
        action=ActionSpec(
            name="omega", targets="cylinder", low=-5.0, high=5.0, distribution="beta"
        ),
        control_interval=0.05,
        start_time=4.0,
    )


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------

def test_dimensions_follow_the_sources(rich):
    assert rich.observation.frame_dim == 4      # 2 probes + Cd + Cl
    assert rich.obs_dim == 4
    assert rich.act_dim == 1

    stacked = make_spec(sources=rich.observation.sources, stack=2)
    assert stacked.obs_dim == 8
    assert len(stacked.observation.component_names()) == 8


def test_spec_round_trips_through_json(rich):
    assert PolicySpec.from_json(rich.to_json()) == rich


def test_hash_is_stable_and_sensitive(rich):
    assert rich.hash == PolicySpec.from_json(rich.to_json()).hash

    moved = make_spec(
        sources=(
            ProbeSource(
                field_name="p", positions=[(0.15, -0.6, 0.005), (2.0, 0.0, 0.005)]
            ),
            ForceCoeffSource(patch="cylinder", coefficients=("Cd", "Cl")),
        ),
        action=rich.action,
        start_time=rich.start_time,
    )
    assert moved.hash != rich.hash


def test_source_order_defines_column_order(rich):
    columns = rich.trajectory_columns()
    assert columns[0] == "time"
    assert columns[-1] == "seed"
    assert columns[-2] == "Q"
    assert len(columns) == 1 + rich.obs_dim + rich.act_dim + 1


@pytest.mark.parametrize(
    "targets, expected",
    [
        ("cylinder", (("cylinder", 1.0),)),
        (["jetA", "jetB"], (("jetA", 1.0), ("jetB", 1.0))),
        ({"jet1": 1.0, "jet2": -1.0}, (("jet1", 1.0), ("jet2", -1.0))),
    ],
)
def test_targets_normalize_to_name_coefficient_pairs(targets, expected):
    action = ActionSpec(name="Q", targets=targets, low=-1.0, high=1.0)

    assert action.targets == expected
    assert action.target_names == [name for name, _ in expected]
    assert action.distribution == "gaussian"    # the default


@pytest.mark.parametrize(
    "distribution, expected",
    [("gaussian", ("mean", "log_std")), ("beta", ("alpha", "beta"))],
)
def test_output_names_follow_the_distribution(distribution, expected):
    action = ActionSpec(
        name="Q", targets="jet1", low=-0.1, high=0.1, distribution=distribution
    )
    assert make_spec(action=action).output_names == expected


@pytest.mark.parametrize(
    "build, match",
    [
        (an_action(low=1.0, high=1.0), "low < high"),
        (an_action(low=[0.0, 1.0], n_components=3), "entries"),
        (an_action(targets=["j", "j"]), "duplicate"),
        (an_action(targets=[]), "at least one target"),
        (an_action(ramp_fraction=-0.1), "ramp_fraction"),
        (an_action(ramp_fraction=1.5), "ramp_fraction"),
        (an_action(distribution="cauchy"), "Unsupported distribution"),
        (partial(ObservationSpec, sources=()), "at least one source"),
        (partial(ObservationSpec, sources=(PROBES,), stack=0), "stack must be"),
        (partial(make_spec, control_interval=0.0), "control_interval"),
        (partial(make_spec, start_time=1.0, end_time=0.5), "end_time"),
    ],
)
def test_invalid_specs_are_rejected(build, match):
    with pytest.raises(ValueError, match=match):
        build()


def test_solver_readable_metadata_covers_targets_and_ramp(rich):
    metadata = rich.to_metadata()
    assert metadata["uqtopus.action_targets"] == "jet1:1.0 jet2:-1.0"
    assert metadata["uqtopus.ramp_fraction"] == "0.5"


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------

def test_normalization_is_a_defensive_snapshot():
    source = np.ones(4)
    norm = Normalization(mean=source, std=np.ones(4))

    source[0] = 99.0
    assert norm.mean[0] == 1.0                  # the input was copied
    for array in (norm.mean, norm.std):
        with pytest.raises(ValueError):
            array[0] = 1.0                      # and the copy is read-only

    with pytest.raises(ValueError, match="strictly positive"):
        Normalization(mean=np.zeros(3), std=np.array([1.0, 0.0, 1.0]))


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def test_exported_policy_passes_validation(beta, tmp_path):
    net = build_mlp(beta, hidden=(16, 16), seed=0)
    norm = Normalization.from_observations(
        np.random.default_rng(0).normal(3.0, 2.0, (100, beta.obs_dim))
    )
    artifact = export_policy(net, beta, tmp_path / "policy.onnx", normalization=norm)

    report = validate_export(net, beta, artifact.path, normalization=norm, strict=False)
    assert report.ok, [str(c) for c in report.errors]


def test_metadata_is_self_describing(beta, tmp_path):
    artifact = export_random_policy(beta, tmp_path / "policy.onnx", seed=0)
    metadata = read_metadata(artifact.path)

    assert metadata["uqtopus.spec_hash"] == beta.hash
    assert int(metadata["uqtopus.obs_dim"]) == beta.obs_dim
    assert int(metadata["uqtopus.act_dim"]) == beta.act_dim
    assert metadata["uqtopus.distribution"] == "beta"
    # the flat entries exist so the solver can read bounds without a JSON parser
    assert metadata["uqtopus.action_low"] == "-5.0"
    assert metadata["uqtopus.action_high"] == "5.0"
    assert PolicySpec.from_metadata(metadata) == beta


def test_normalization_is_baked_into_the_graph(beta, tmp_path):
    net = build_mlp(beta, hidden=(16, 16), seed=0)
    norm = Normalization(
        mean=np.full(beta.obs_dim, 10.0), std=np.full(beta.obs_dim, 2.0)
    )
    artifact = export_policy(net, beta, tmp_path / "policy.onnx", normalization=norm)

    raw = np.random.default_rng(1).normal(10.0, 2.0, (8, beta.obs_dim)).astype(np.float32)
    session = ort.InferenceSession(str(artifact.path), providers=["CPUExecutionProvider"])
    from_graph = session.run(None, {"observation": raw})

    # the graph on raw input must equal the bare network on normalized input
    with torch.no_grad():
        head = net(torch.tensor(norm.apply(raw), dtype=torch.float32))
    expected_alpha = torch.nn.functional.softplus(head[:, : beta.act_dim]) + 1.0

    assert np.allclose(from_graph[0], expected_alpha.numpy(), atol=1e-5)


def test_artifact_manifest_round_trips(beta, tmp_path):
    artifact = export_random_policy(beta, tmp_path / "policy.onnx", iteration=3, seed=0)
    reloaded = PolicyArtifact.load(artifact.manifest_path)

    assert reloaded.spec == beta
    assert reloaded.iteration == 3
    assert reloaded.verify_file()
    assert np.array_equal(reloaded.normalization.mean, artifact.normalization.mean)


def test_export_rejects_mismatched_normalization(beta, tmp_path):
    net = build_mlp(beta, hidden=(8,), seed=0)
    with pytest.raises(ValueError, match="dimension"):
        export_policy(
            net,
            beta,
            tmp_path / "policy.onnx",
            normalization=Normalization.identity(beta.obs_dim + 1),
        )


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def test_stale_normalization_is_caught_by_parity(beta, tmp_path):
    """The graph was exported with one set of statistics, the update uses another."""
    net = build_mlp(beta, hidden=(16, 16), seed=0)
    exported_with = Normalization(
        mean=np.full(beta.obs_dim, 3.0), std=np.full(beta.obs_dim, 2.0)
    )
    artifact = export_policy(
        net, beta, tmp_path / "policy.onnx", normalization=exported_with
    )

    moved_on = Normalization(
        mean=np.full(beta.obs_dim, 3.5), std=np.full(beta.obs_dim, 2.1)
    )
    report = validate_export(net, beta, artifact.path, normalization=moved_on, strict=False)

    assert not report.ok
    assert "parity" in [c.name for c in report.errors]


def test_contract_mismatch_is_caught(beta, tmp_path):
    artifact = export_random_policy(beta, tmp_path / "policy.onnx", seed=0)

    other = make_spec(
        sources=(PatchSource(field_name="p", patch="inlet"),),
        action=beta.action,
        control_interval=beta.control_interval,
    )
    report = validate_policy(artifact.path, other, strict=False)

    assert not report.ok
    assert "contract" in [c.name for c in report.errors]


def test_missing_metadata_is_reported(beta, tmp_path):
    """A hand-made ONNX file is a valid graph but not a valid policy."""
    net = build_mlp(beta, hidden=(8,), seed=0)
    path = tmp_path / "bare.onnx"
    torch.onnx.export(
        net,
        torch.zeros(1, beta.obs_dim),
        str(path),
        input_names=["observation"],
        output_names=["alpha"],
        opset_version=17,
        dynamo=False,
    )

    report = validate_policy(path, beta, strict=False)
    assert not report.ok
    assert "metadata" in [c.name for c in report.errors]


def test_strict_mode_raises(beta, tmp_path):
    with pytest.raises(ValueError, match="validation failed"):
        validate_policy(tmp_path / "does_not_exist.onnx", beta, strict=True)


# ---------------------------------------------------------------------------
# serialization of the controller dictionary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mapping, expected",
    [
        ({"type": "onnxPolicy"}, "type            onnxPolicy;"),
        ({"n": 3}, "n               3;"),
        ({"dt": 0.4}, "dt              0.4;"),
        ({"on": True}, "on              yes;"),
        ({"low": [-0.1, -0.2]}, "low             (-0.1 -0.2);"),
        # paths and hashes are not bare words, so they travel quoted
        ({"policy": "/tmp/policies/iter001.onnx"}, '"/tmp/policies/iter001.onnx"'),
        ({"specHash": "0def707f"}, '"0def707f"'),
    ],
)
def test_entries_are_serialized_by_type(mapping, expected):
    assert expected in format_block(mapping)


def test_bare_words_are_unquoted_and_none_is_dropped():
    assert '"' not in format_block({"type": "onnxPolicy"})
    assert "endTime" not in format_block({"startTime": 4.0, "endTime": None})


def test_nested_structures_stay_balanced():
    text = format_block({"a": {"b": {"c": 1}}, "list": [{"x": 1}, {"y": 2}]})
    assert text.count("{") == text.count("}")
    assert text.count("(") == text.count(")")

    spread = format_block({"positions": [(0.0, 1.0, 2.0), (3.0, 4.0, 5.0)]})
    assert "(0 1 2)" in spread and "(3 4 5)" in spread
    assert spread.count("\n") > 3


# ---------------------------------------------------------------------------
# the controller block
# ---------------------------------------------------------------------------

def test_block_carries_the_contract(rich, tmp_path):
    text = render_controller(rich, tmp_path / "policy.onnx", seed=42)

    assert f'specHash        "{rich.hash}";' in text
    assert "controlInterval 0.4;" in text
    assert "startTime       4;" in text
    assert "seed            42;" in text
    assert "policy.onnx" in text


def test_named_and_bare_forms(rich, tmp_path):
    bare = render_controller(rich, tmp_path / "p.onnx")
    named = render_controller(rich, tmp_path / "p.onnx", name="uqtopusController")

    assert not bare.lstrip().startswith("{")    # goes inside an existing entry
    assert named.startswith("uqtopusController\n{")
    assert named.count("{") == named.count("}")


def test_observation_sources_are_rendered_in_order(rich, tmp_path):
    text = render_controller(rich, tmp_path / "p.onnx")

    assert text.index("kind") < text.index("forceCoeffs")
    assert "(0.55 -0.6 0.005)" in text
    assert "dim             4;" in text          # 2 probes + Cd + Cl


def test_paired_jets_keep_their_coefficients(rich, tmp_path):
    text = render_controller(rich, tmp_path / "p.onnx")

    assert "name            jet1;" in text
    assert "coefficient     1;" in text
    assert "coefficient     -1;" in text
    assert "rampFraction    0.5;" in text


def test_the_block_is_extensible(rich, tmp_path):
    """An MPC controller reuses the same plumbing; extra entries pass through."""
    text = render_controller(
        rich, tmp_path / "p.onnx", controller_type="mpc", extra={"deterministic": True}
    )
    assert "type            mpc;" in text
    assert "deterministic   yes;" in text


def test_uncommon_sources_reach_the_dictionary(tmp_path):
    """A source the package has no dedicated class for still gets rendered."""
    spec = make_spec(
        sources=(
            PatchSource(field_name="p", patch="inlet", operation="areaAverage"),
            CustomSource(name="wakeWidth", n_components=2, options={"tol": 0.01}),
        )
    )
    text = render_controller(spec, tmp_path / "p.onnx")

    assert "fieldName       p;" in text
    assert "operation       areaAverage;" in text
    assert "kind            custom;" in text
    assert "nComponents     2;" in text
    assert "tol" in text


def test_mapping_is_available_before_rendering(rich, tmp_path):
    mapping = controller_mapping(rich, tmp_path / "p.onnx")

    assert mapping["specHash"] == rich.hash
    assert mapping["observation"]["dim"] == rich.obs_dim
    assert mapping["action"]["targets"][1] == {"name": "jet2", "coefficient": -1.0}


def test_params_target_the_template_keys(rich, tmp_path):
    params = controller_params(rich, tmp_path / "p.onnx", ["0__U__controller"])
    assert list(params) == ["0__U__controller"]
    assert rich.hash in params["0__U__controller"]

    several = controller_params(
        rich, tmp_path / "p.onnx", ["0__U__controller", "constant__fvOptions__controller"]
    )
    assert len(several) == 2
    assert len(set(several.values())) == 1      # the same block in both files
