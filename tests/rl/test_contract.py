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
    export_policy,
    export_random_policy,
    read_metadata,
    validate_policy,
)
from uqtopus.rl.export import _PolicyGraph, build_mlp
from uqtopus.rl.foam import (
    controller_params,
    format_block,
    render_controller,
)

from conftest import make_spec


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


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------

def test_the_sources_define_the_dimension_and_the_columns(rich):
    assert rich.obs_dim == 4                    # 2 pressure probes + 2 U components
    assert rich.observation.component_names() == [
        "probes.p.0", "probes.p.1", "wake.U0.0", "wake.U1.0",
    ]


def test_spec_round_trips_through_json(rich):
    assert PolicySpec.from_json(rich.to_json()) == rich


def test_the_hash_changes_when_a_probe_moves(rich):
    moved = make_spec(
        sources=(
            ProbeSource(field_name="p", positions=[(0.15, -0.6, 0.005), (2.0, 0.0, 0.005)]),
            *rich.observation.sources[1:],
        ),
        action=rich.action,
        start_time=rich.start_time,
    )
    assert moved.hash != rich.hash


@pytest.mark.parametrize(
    "targets, expected, names",
    [
        ("cylinder", ("cylinder",), ["Q"]),
        (["jetA", "jetB"], ("jetA", "jetB"), ["Q.0", "Q.1"]),
        (("rate", "pressure", "quality"), ("rate", "pressure", "quality"), ["Q.0", "Q.1", "Q.2"]),
    ],
)
def test_targets_are_positional(targets, expected, names):
    action = ActionSpec(name="Q", targets=targets, low=-1.0, high=1.0)

    assert action.targets == expected
    assert action.n_components == len(expected)
    assert action.component_names() == names


@pytest.mark.parametrize(
    "build, match",
    [
        (an_action(low=1.0, high=1.0), "low < high"),
        (an_action(low=[0.0, 1.0], targets=["a", "b", "c"]), "entries"),
        (an_action(targets=["j", "j"]), "duplicate"),
        (an_action(targets=[]), "at least one target"),
        (an_action(ramp_fraction=-0.1), "ramp_fraction"),
        (an_action(ramp_fraction=1.5), "ramp_fraction"),
        (partial(ObservationSpec, sources=()), "at least one source"),
        (partial(make_spec, control_interval=0.0), "control_interval"),
        (partial(make_spec, start_time=1.0, end_time=0.5), "end_time"),
    ],
)
def test_invalid_specs_are_rejected(build, match):
    with pytest.raises(ValueError, match=match):
        build()


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

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
    norm = Normalization(mean=np.full(spec.obs_dim, 3.0), std=np.full(spec.obs_dim, 2.0))
    artifact = export_policy(net, spec, tmp_path / "policy.onnx", normalization=norm)

    validate_policy(artifact, spec)


def test_export_refuses_an_actor_too_narrow_for_the_spec(spec, tmp_path):
    net = torch.nn.Linear(spec.obs_dim, spec.act_dim)

    with pytest.raises(ValueError, match="returns an action of shape"):
        export_policy(net, spec, tmp_path / "policy.onnx")
    assert not (tmp_path / "policy.onnx").exists()


def test_metadata_is_self_describing(rich, tmp_path):
    artifact = export_random_policy(rich, tmp_path / "policy.onnx", seed=0)
    metadata = read_metadata(artifact.path)

    assert metadata["uqtopus.spec_hash"] == rich.hash
    assert (metadata["uqtopus.obs_dim"], metadata["uqtopus.act_dim"]) == ("4", "1")
    # the flat entries exist so the solver can read them without a JSON parser
    assert metadata["uqtopus.action_low"] == "-0.1"
    assert metadata["uqtopus.action_high"] == "0.1"
    assert metadata["uqtopus.action_targets"] == "jet:0"
    assert metadata["uqtopus.ramp_fraction"] == "0.5"
    assert PolicySpec.from_metadata(metadata) == rich


def test_normalization_is_baked_into_the_graph(spec, tmp_path):
    net = build_mlp(spec, hidden=(16, 16), seed=0)
    norm = Normalization(
        mean=np.full(spec.obs_dim, 10.0), std=np.full(spec.obs_dim, 2.0)
    )
    artifact = export_policy(net, spec, tmp_path / "policy.onnx", normalization=norm)

    raw = np.random.default_rng(1).normal(10.0, 2.0, (8, spec.obs_dim)).astype(np.float32)
    session = ort.InferenceSession(str(artifact.path), providers=["CPUExecutionProvider"])
    noise = np.zeros((8, spec.act_dim), np.float32)
    from_graph = session.run(None, {"observation": raw, "noise": noise})

    # the graph on raw input must equal the bare network on normalized input
    with torch.no_grad():
        head = net(torch.tensor(norm.apply(raw), dtype=torch.float32))
    expected_mean = head[:, : spec.act_dim]

    assert np.allclose(from_graph[0], expected_mean.numpy(), atol=1e-5)


def test_artifact_reloads_from_the_onnx_file(spec, tmp_path):
    norm = Normalization(mean=np.full(spec.obs_dim, 3.0), std=np.full(spec.obs_dim, 2.0))
    artifact = export_random_policy(
        spec, tmp_path / "policy.onnx", iteration=3, seed=0, normalization=norm
    )
    reloaded = PolicyArtifact.load(artifact.path)

    assert reloaded.spec == spec
    assert reloaded.iteration == 3
    assert np.array_equal(reloaded.normalization.mean, norm.mean)
    assert np.array_equal(reloaded.normalization.std, norm.std)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def test_contract_mismatch_is_caught(spec, tmp_path):
    artifact = export_random_policy(spec, tmp_path / "policy.onnx", seed=0)

    other = make_spec(
        sources=(ProbeSource(field_name="p", positions=[(9.0, 9.0, 0.005)]),),
        action=spec.action,
        control_interval=spec.control_interval,
    )
    with pytest.raises(ValueError, match="contract"):
        validate_policy(artifact.path, other)


def test_missing_metadata_is_reported(spec, tmp_path):
    """A hand-made ONNX file is a valid graph but not a valid policy."""
    net = build_mlp(spec, hidden=(8,), seed=0)
    path = tmp_path / "bare.onnx"
    torch.onnx.export(
        net,
        torch.zeros(1, spec.obs_dim),
        str(path),
        input_names=["observation"],
        output_names=["action"],
        opset_version=17,
        dynamo=False,
    )

    with pytest.raises(ValueError, match="uqtopus.spec"):
        validate_policy(path, spec)


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
    assert "dim             4;" in text          # 2 pressure probes + 2 U components
    assert "(0.55 -0.6 0.005)" in text
    assert text.index("probes") < text.index("wake")


def test_a_relative_policy_path_is_quoted(rich):
    assert 'policy          "runs/policy.onnx";' in render_controller(rich, "runs/policy.onnx")


def test_the_block_is_the_uqtopusPolicy_entry(rich, tmp_path):
    text = render_controller(rich, tmp_path / "p.onnx", deterministic=True)

    assert text.startswith("uqtopusPolicy\n{")
    assert text.count("{") == text.count("}")
    assert "deterministic   yes;" in text


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


def test_params_target_the_template_keys(rich, tmp_path):
    params = controller_params(rich, tmp_path / "p.onnx", ["0__U__controller"])
    assert list(params) == ["0__U__controller"]
    assert rich.hash in params["0__U__controller"]

    several = controller_params(
        rich, tmp_path / "p.onnx", ["0__U__controller", "constant__fvOptions__controller"]
    )
    assert len(several) == 2
    assert len(set(several.values())) == 1      # the same block in both files
