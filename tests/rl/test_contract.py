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
    Normalization,
    ObservationSpec,
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
from uqtopus.rl.export import _PolicyGraph
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
    """Two probe sources, driving a ramped pair of jets."""
    return make_spec(
        sources=(
            ProbeSource(
                field_name="p", positions=[(0.55, -0.6, 0.005), (2.0, 0.0, 0.005)]
            ),
            ProbeSource(
                field_name="U", positions=[(3.0, 0.0, 0.005)], components=(0, 1),
                name="wake"
            ),
        ),
        action=ActionSpec(
            name="Q",
            targets="jet",
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
    assert rich.obs_dim == 4                    # 2 pressure probes + 2 U components
    assert rich.act_dim == 1
    assert len(rich.observation.component_names()) == 4


def test_spec_round_trips_through_json(rich):
    assert PolicySpec.from_json(rich.to_json()) == rich


def test_hash_is_stable_and_sensitive(rich):
    assert rich.hash == PolicySpec.from_json(rich.to_json()).hash

    moved = make_spec(
        sources=(
            ProbeSource(
                field_name="p", positions=[(0.15, -0.6, 0.005), (2.0, 0.0, 0.005)]
            ),
            ProbeSource(
                field_name="U", positions=[(3.0, 0.0, 0.005)], components=(0, 1),
                name="wake"
            ),
        ),
        action=rich.action,
        start_time=rich.start_time,
    )
    assert moved.hash != rich.hash


def test_source_order_defines_column_order(rich):
    columns = rich.trajectory_columns()
    assert columns[0] == "time"
    assert columns[-1] == "Q"
    assert len(columns) == 1 + rich.obs_dim + rich.act_dim


@pytest.mark.parametrize(
    "targets, expected",
    [
        ("cylinder", ("cylinder",)),
        (["jetA", "jetB"], ("jetA", "jetB")),
        (("rate", "pressure", "quality"), ("rate", "pressure", "quality")),
    ],
)
def test_targets_are_positional(targets, expected):
    action = ActionSpec(name="Q", targets=targets, low=-1.0, high=1.0)

    assert action.targets == expected
    assert action.target_names == list(expected)
    assert action.n_components == len(expected)
    assert action.distribution == "gaussian"    # the default


def test_each_target_takes_its_own_action_component():
    """Three quantities driven by three components of one action."""
    action = ActionSpec(
        name="injection",
        targets=("rate", "pressure", "quality"),
        low=[0.0, 1e5, 0.0],
        high=[1.0, 5e5, 1.0],
    )
    spec = make_spec(action=action)

    assert spec.act_dim == 3
    assert spec.action.component_names() == [
        "injection.0", "injection.1", "injection.2"
    ]
    assert PolicySpec.from_dict(spec.to_dict()) == spec

    rendered = controller_mapping(spec, "p.onnx")["action"]["targets"]
    assert [entry["component"] for entry in rendered] == [0, 1, 2]


@pytest.mark.parametrize(
    "distribution, inputs, outputs, noise_dim",
    [
        ("gaussian", ("observation", "noise"), ("action",), 1),
        ("beta", ("observation",), ("alpha", "beta"), 0),
    ],
)
def test_signature_follows_the_distribution(distribution, inputs, outputs, noise_dim):
    action = ActionSpec(
        name="Q", targets="jet1", low=-0.1, high=0.1, distribution=distribution
    )
    spec = make_spec(action=action)

    assert spec.input_names == inputs
    assert spec.output_names == outputs
    assert spec.noise_dim == noise_dim


@pytest.mark.parametrize(
    "build, match",
    [
        (an_action(low=1.0, high=1.0), "low < high"),
        (
            an_action(
                low=[0.0, 1.0],
                n_components=3,
                targets=["a", "b", "c"],
            ),
            "entries",
        ),
        (an_action(targets="a", n_components=2), "one target drives one component"),
        (an_action(targets=["j", "j"]), "duplicate"),
        (an_action(targets=[]), "at least one target"),
        (an_action(ramp_fraction=-0.1), "ramp_fraction"),
        (an_action(ramp_fraction=1.5), "ramp_fraction"),
        (an_action(distribution="cauchy"), "Unsupported distribution"),
        (partial(ObservationSpec, sources=()), "at least one source"),
        (partial(make_spec, control_interval=0.0), "control_interval"),
        (partial(make_spec, start_time=1.0, end_time=0.5), "end_time"),
    ],
)
def test_invalid_specs_are_rejected(build, match):
    with pytest.raises(ValueError, match=match):
        build()


def test_solver_readable_metadata_covers_targets_and_ramp(rich):
    metadata = rich.to_metadata()
    assert metadata["uqtopus.action_targets"] == "jet:0"
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


def test_graph_buffers_follow_the_actor(spec):
    """
    A policy that lives off the CPU exports from there.

    The meta device stands in for a GPU: what matters is that the graph does not
    end up with the actor on one device and the normalization buffers on
    another, which fails in the forward pass and in torch.export alike.
    """
    net = build_mlp(spec, hidden=(16, 16), seed=0).to("meta")
    graph = _PolicyGraph(net, spec, Normalization.identity(spec.obs_dim))

    devices = {p.device.type for p in graph.parameters()}
    devices |= {b.device.type for b in graph.buffers()}
    assert devices == {"meta"}


def test_gaussian_graph_draws_without_bounding(spec, tmp_path):
    """
    The draw enters the graph and leaves it unbounded.

    The update needs the sample the distribution produced, so the bounds are the
    solver's to enforce when it drives the actuator. A graph that clipped would
    hand the update a likelihood the policy never drew from.
    """
    net = build_mlp(spec, hidden=(16, 16), seed=0)
    artifact = export_policy(net, spec, tmp_path / "policy.onnx")
    session = ort.InferenceSession(str(artifact.path), providers=["CPUExecutionProvider"])

    obs = np.random.default_rng(0).normal(size=(4, spec.obs_dim)).astype(np.float32)
    high = spec.action.high[0]
    with torch.no_grad():
        mean = net(torch.tensor(obs))[:, : spec.act_dim].numpy()

    at_zero = session.run(
        None, {"observation": obs, "noise": np.zeros((4, spec.act_dim), np.float32)}
    )[0]
    assert np.allclose(at_zero, mean, atol=1e-5)

    far = session.run(
        None, {"observation": obs, "noise": np.full((4, spec.act_dim), 1e3, np.float32)}
    )[0]
    assert np.all(far > high)


def test_gaussian_export_passes_validation(spec, tmp_path):
    net = build_mlp(spec, hidden=(16, 16), seed=0)
    norm = Normalization.from_observations(
        np.random.default_rng(0).normal(3.0, 2.0, (100, spec.obs_dim))
    )
    artifact = export_policy(net, spec, tmp_path / "policy.onnx", normalization=norm)

    report = validate_export(net, spec, artifact.path, normalization=norm, strict=False)
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
        sources=(ProbeSource(field_name="p", positions=[(9.0, 9.0, 0.005)]),),
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
        ({"type": "uqtopusBoundaryCondition"}, "type            uqtopusBoundaryCondition;"),
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
    assert '"' not in format_block({"type": "uqtopusBoundaryCondition"})
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

    # quoted or bare depending on whether the hash starts with a digit
    assert rich.hash in text
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

    assert text.index("probes") < text.index("wake")
    assert "(0.55 -0.6 0.005)" in text
    assert "dim             4;" in text          # 2 pressure probes + 2 U components


def test_every_target_reaches_the_block_with_its_component(tmp_path):
    spec = make_spec(
        action=ActionSpec(
            name="Q", targets=("rate", "quality"), low=-1.0, high=1.0,
            ramp_fraction=0.5,
        )
    )
    text = render_controller(spec, tmp_path / "p.onnx")

    assert "name            rate;\n" in text and "component       0;" in text
    assert "name            quality;\n" in text and "component       1;" in text
    assert "rampFraction    0.5;" in text


def test_the_block_is_extensible(rich, tmp_path):
    """An MPC controller reuses the same plumbing; extra entries pass through."""
    text = render_controller(
        rich, tmp_path / "p.onnx", controller_type="mpc", extra={"deterministic": True}
    )
    assert "type            mpc;" in text
    assert "deterministic   yes;" in text


def test_mapping_is_available_before_rendering(rich, tmp_path):
    mapping = controller_mapping(rich, tmp_path / "p.onnx")

    assert mapping["specHash"] == rich.hash
    assert mapping["observation"]["dim"] == rich.obs_dim
    assert mapping["action"]["targets"][0] == {"name": "jet", "component": 0}


def test_params_target_the_template_keys(rich, tmp_path):
    params = controller_params(rich, tmp_path / "p.onnx", ["0__U__controller"])
    assert list(params) == ["0__U__controller"]
    assert rich.hash in params["0__U__controller"]

    several = controller_params(
        rich, tmp_path / "p.onnx", ["0__U__controller", "constant__fvOptions__controller"]
    )
    assert len(several) == 2
    assert len(set(several.values())) == 1      # the same block in both files
