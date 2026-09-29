"""
Tests for the running statistics, the buffer fill and the PPO subclass.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest
import torch as th

from uqtopus.rl import Normalization, export_random_policy
from uqtopus.rl.algos import PPO
from uqtopus.rl.buffer import build_rollout_buffer, rollout_statistics
from uqtopus.rl.export import RunningStatistics

from conftest import N_STEPS, make_runner

# ---------------------------------------------------------------------------
# running statistics
# ---------------------------------------------------------------------------

def test_running_statistics_match_a_direct_computation():
    stats = RunningStatistics(4)
    # the first iteration must run on raw observations, not on a guess
    assert np.allclose(stats.snapshot().mean, 0.0)
    assert np.allclose(stats.snapshot().std, 1.0)

    data = np.random.default_rng(0).normal(3.0, 2.0, (500, 4))
    for chunk in np.array_split(data, 7):        # arrives in uneven batches
        stats.update(chunk)
    snapshot = stats.snapshot()

    assert np.allclose(snapshot.mean, data.mean(axis=0), atol=1e-9)
    assert np.allclose(snapshot.std, data.std(axis=0), atol=1e-6)


def test_snapshots_do_not_move_when_the_estimate_does():
    stats = RunningStatistics(2)
    stats.update(np.zeros((10, 2)))
    frozen = stats.snapshot()

    stats.update(np.full((10, 2), 100.0))

    assert np.allclose(frozen.mean, 0.0)
    assert not np.allclose(stats.snapshot().mean, frozen.mean)


# ---------------------------------------------------------------------------
# buffer
# ---------------------------------------------------------------------------

def test_buffer_uses_the_artifacts_snapshot(runner, spec, tmp_path):
    normalization = Normalization(
        mean=np.full(spec.obs_dim, 5.0), std=np.full(spec.obs_dim, 2.0)
    )
    artifact = export_random_policy(
        spec, tmp_path / "p.onnx", seed=0, normalization=normalization
    )
    rollout = runner.collect(artifact, n_episodes=1)

    normalized = rollout.artifact.normalization.apply(rollout.observations)
    assert np.allclose(normalized, (rollout.observations - 5.0) / 2.0)


def test_buffer_is_full_and_carries_the_episode_boundaries(runner, spec, tmp_path):
    model = PPO("MlpPolicy", runner, n_episodes=3, export_dir=tmp_path / "pol")
    rollout = runner.collect(model.export_current_policy(tmp_path / "p.onnx"), n_episodes=3)

    buffer = build_rollout_buffer(rollout, model.policy, gamma=0.99, gae_lambda=0.95)

    assert buffer.full
    # the last row of each episode only supplies the state to bootstrap from
    assert buffer.buffer_size == rollout.n_steps - 3 == 3 * (N_STEPS - 1)
    assert buffer.observations.shape == (buffer.buffer_size, 1, spec.obs_dim)
    assert np.all(np.isfinite(buffer.advantages))
    assert np.all(np.isfinite(buffer.returns))
    assert np.flatnonzero(buffer.episode_starts.ravel()).tolist() == [
        0, N_STEPS - 1, 2 * (N_STEPS - 1)
    ]


def test_log_probs_are_recomputed_from_the_stored_actions(runner, tmp_path):
    """The solver never writes a log-probability; it is recovered here."""
    model = PPO("MlpPolicy", runner, n_episodes=1, export_dir=tmp_path / "pol")
    rollout = runner.collect(model.export_current_policy(tmp_path / "p.onnx"), n_episodes=1)
    buffer = build_rollout_buffer(rollout, model.policy, gamma=0.99, gae_lambda=0.95)

    obs = th.as_tensor(rollout.artifact.normalization.apply(rollout.observations), dtype=th.float32)
    act = th.as_tensor(rollout.actions, dtype=th.float32)
    with th.no_grad():
        _, expected, _ = model.policy.evaluate_actions(obs, act)

    assert np.allclose(buffer.log_probs.ravel(), expected.numpy().ravel()[:-1], atol=1e-6)


@pytest.fixture
def exact_critic(simulator, spec, tmp_path):
    """
    Three episodes with a reward of 1 at every step, and a model whose critic
    returns the exact discounted value of that reward from any state.
    """
    runner = make_runner(simulator, spec, reward_fn=lambda ds: np.ones(ds.sizes["time"]))
    model = PPO("MlpPolicy", runner, export_dir=tmp_path / "pol")
    with th.no_grad():
        model.policy.value_net.weight.zero_()
        model.policy.value_net.bias.fill_(1.0 / (1.0 - model.gamma))
    rollout = runner.collect(model.export_current_policy(tmp_path / "p.onnx"), n_episodes=3)
    return model, rollout


def test_a_truncated_episode_is_bootstrapped_from_its_last_row(exact_critic):
    """With the exact value function, every advantage is zero."""
    model, rollout = exact_critic
    buffer = build_rollout_buffer(rollout, model.policy, gamma=model.gamma, gae_lambda=0.95)

    assert np.allclose(buffer.advantages, 0.0, atol=1e-3)


def test_a_terminal_end_charges_the_lost_future_to_the_last_step(exact_critic):
    model, rollout = exact_critic
    buffer = build_rollout_buffer(
        rollout, model.policy, gamma=model.gamma, gae_lambda=0.95, truncated=False
    )

    last = np.cumsum(rollout.lengths) - 1
    assert buffer.buffer_size == rollout.n_steps
    assert np.allclose(buffer.advantages.ravel()[last], 1.0 - 1.0 / (1.0 - model.gamma), atol=1e-3)


def test_a_diverged_episode_ends_terminal(exact_critic):
    model, rollout = exact_critic
    rollout.episodes[1].attrs["diverged"] = True
    buffer = build_rollout_buffer(rollout, model.policy, gamma=model.gamma, gae_lambda=0.95)

    first, diverged, third = np.split(buffer.advantages.ravel(), [N_STEPS - 1, 2 * N_STEPS - 1])
    assert buffer.buffer_size == rollout.n_steps - 2
    assert np.allclose(first, 0.0, atol=1e-3)
    assert np.allclose(third, 0.0, atol=1e-3)
    assert np.isclose(diverged[-1], 1.0 - 1.0 / (1.0 - model.gamma), atol=1e-3)


def test_statistics_summarize_the_batch(runner, artifact):
    rollout = runner.collect(artifact, n_episodes=2)

    stats = rollout_statistics(rollout)
    assert stats["rollout/episodes"] == 2.0
    assert stats["rollout/ep_len_mean"] == float(N_STEPS)
    assert stats["rollout/diverged"] == 0.0


# ---------------------------------------------------------------------------
# the PPO subclass
# ---------------------------------------------------------------------------

def test_the_actor_stays_on_the_cpu(runner, tmp_path):
    assert PPO("MlpPolicy", runner, export_dir=tmp_path / "pol").device.type == "cpu"


@pytest.mark.parametrize("log_std", [-7.0, 3.0])
def test_the_graph_draws_from_the_distribution_the_update_evaluates(
    runner, spec, tmp_path, log_std
):
    """A unit draw moves the action by the standard deviation of the policy, narrow or wide."""
    model = PPO("MlpPolicy", runner, export_dir=tmp_path / "pol")
    with th.no_grad():
        model.policy.log_std.fill_(log_std)
    artifact = model.export_current_policy(tmp_path / "p.onnx")
    session = ort.InferenceSession(str(artifact.path), providers=["CPUExecutionProvider"])

    obs = np.random.default_rng(0).normal(size=(1, spec.obs_dim))
    unit = np.ones((1, spec.act_dim), np.float32)
    (at_zero,) = session.run(None, {"observation": obs, "noise": 0 * unit})
    (at_one,) = session.run(None, {"observation": obs, "noise": unit})
    with th.no_grad():
        distribution = model.policy.get_distribution(
            th.as_tensor(obs, dtype=th.float32)
        ).distribution

    assert np.allclose(at_zero, distribution.mean.numpy(), atol=1e-6)
    assert np.allclose(at_one - at_zero, distribution.stddev.numpy(), rtol=1e-4)


def test_a_full_training_loop_runs(runner, tmp_path):
    """Three iterations end to end, checking what each one must leave behind."""
    model = PPO(
        "MlpPolicy",
        runner,
        n_episodes=2,
        export_dir=tmp_path / "policies",
        batch_size=8,
        n_epochs=4,
        seed=0,
    )
    model.learn(total_timesteps=3 * 2 * N_STEPS)

    assert len(model.rollouts) >= 3
    assert len(model.artifacts) == len(model.rollouts)
    assert model.num_timesteps >= 3 * 2 * N_STEPS
    assert model.returns_history.shape == (len(model.rollouts),)

    files = sorted((tmp_path / "policies").glob("policy_iter*.onnx"))
    assert len(files) == len(model.artifacts)

    first, second, last = model.artifacts[0], model.artifacts[1], model.artifacts[-1]
    assert first.path.read_bytes() != last.path.read_bytes(), "training did not move the policy"

    # the first graph saw no observations at all
    assert np.allclose(first.normalization.mean, 0.0)
    assert np.allclose(first.normalization.std, 1.0)
    # the second saw exactly the first rollout
    assert np.allclose(
        second.normalization.mean, model.rollouts[0].observations.mean(axis=0), atol=1e-9
    )

    final = model.export_current_policy(tmp_path / "trained.onnx")
    assert final.path.exists()
    assert final.spec.hash == runner.spec.hash


