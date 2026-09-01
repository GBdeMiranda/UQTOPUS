"""
Tests for the running statistics, the buffer fill and the PPO subclass.
"""

from __future__ import annotations

import numpy as np
import pytest
import stable_baselines3 as sb3
import torch as th

from uqtopus.rl import (
    ActionSpec,
    Normalization,
    RunningStatistics,
    build_rollout_buffer,
    export_random_policy,
    normalized_observations,
    rollout_statistics,
)
from uqtopus.rl.algos import PPO

from conftest import N_STEPS, make_runner, make_spec

_SEEN_KWARGS: dict = {}
_ORIGINAL_PPO_INIT = sb3.PPO.__init__


def _record_ppo_kwargs(self, policy, env, **kwargs):
    """Stand-in for stable_baselines3.PPO.__init__ that keeps what it was given."""
    _SEEN_KWARGS.clear()
    _SEEN_KWARGS.update(kwargs)
    _ORIGINAL_PPO_INIT(self, policy, env, **kwargs)


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

    assert np.allclose(
        normalized_observations(rollout), (rollout.observations - 5.0) / 2.0
    )


def test_buffer_is_full_and_carries_the_episode_boundaries(runner, spec, tmp_path):
    model = PPO("MlpPolicy", runner, n_episodes=3, export_dir=tmp_path / "pol")
    rollout = runner.collect(model._export(0), n_episodes=3)

    buffer = build_rollout_buffer(rollout, model.policy, gamma=0.99, gae_lambda=0.95)

    assert buffer.full
    assert buffer.buffer_size == rollout.n_steps == 3 * N_STEPS
    assert buffer.observations.shape == (rollout.n_steps, 1, spec.obs_dim)
    assert np.all(np.isfinite(buffer.advantages))
    assert np.all(np.isfinite(buffer.returns))
    assert np.flatnonzero(buffer.episode_starts.ravel()).tolist() == [
        0, N_STEPS, 2 * N_STEPS
    ]


def test_log_probs_are_recomputed_from_the_stored_actions(runner, tmp_path):
    """The solver never writes a log-probability; it is recovered here."""
    model = PPO("MlpPolicy", runner, n_episodes=1, export_dir=tmp_path / "pol")
    rollout = runner.collect(model._export(0), n_episodes=1)
    buffer = build_rollout_buffer(rollout, model.policy, gamma=0.99, gae_lambda=0.95)

    obs = th.as_tensor(normalized_observations(rollout), dtype=th.float32)
    act = th.as_tensor(rollout.actions, dtype=th.float32)
    with th.no_grad():
        _, expected, _ = model.policy.evaluate_actions(obs, act)

    assert np.allclose(buffer.log_probs.ravel(), expected.numpy().ravel(), atol=1e-6)


def test_statistics_summarize_the_batch(runner, tmp_path):
    model = PPO("MlpPolicy", runner, n_episodes=2, export_dir=tmp_path / "pol")
    rollout = runner.collect(model._export(0), n_episodes=2)

    stats = rollout_statistics(rollout)
    assert stats["rollout/episodes"] == 2.0
    assert stats["rollout/ep_len_mean"] == float(N_STEPS)
    assert stats["rollout/diverged"] == 0.0


# ---------------------------------------------------------------------------
# the PPO subclass
# ---------------------------------------------------------------------------

def test_the_actor_stays_on_the_cpu(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(sb3.PPO, "__init__", _record_ppo_kwargs)

    PPO("MlpPolicy", runner, export_dir=tmp_path / "pol")
    assert _SEEN_KWARGS["device"] == "cpu"

    PPO("MlpPolicy", runner, export_dir=tmp_path / "pol", device="auto")
    assert _SEEN_KWARGS["device"] == "auto"


def test_a_beta_spec_is_refused(simulator):
    beta = make_spec(
        action=ActionSpec(
            name="Q", targets="jet1", low=-0.1, high=0.1, distribution="beta"
        )
    )
    with pytest.raises(ValueError, match="Gaussian"):
        PPO("MlpPolicy", make_runner(simulator, beta))


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

    # every iteration left a verifiable policy and a manifest behind
    files = sorted((tmp_path / "policies").glob("policy_iter*.onnx"))
    assert len(files) == len(model.artifacts)
    assert all(a.verify_file() for a in model.artifacts)
    assert all(a.manifest_path.exists() for a in model.artifacts)

    first, second, last = model.artifacts[0], model.artifacts[1], model.artifacts[-1]
    assert first.file_sha256 != last.file_sha256, "training did not move the policy"

    # the first graph saw no observations at all
    assert np.allclose(first.normalization.mean, 0.0)
    assert np.allclose(first.normalization.std, 1.0)
    # the second saw exactly the first rollout
    assert np.allclose(
        second.normalization.mean, model.rollouts[0].observations.mean(axis=0), atol=1e-9
    )

    final = model.export_current_policy(tmp_path / "trained.onnx")
    assert final.path.exists()
    assert final.spec_hash == runner.spec.hash
