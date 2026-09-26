"""
Rollout Buffer Filling

Transfers a Rollout into the stable-baselines3 buffer PPO trains from, normalized
with the snapshot frozen into the graph that produced it.
"""

from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3.common.buffers import RolloutBuffer

from .runner import Rollout


def build_rollout_buffer(
    rollout: Rollout,
    policy,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> RolloutBuffer:
    """
    Build a filled RolloutBuffer, sized to the rollout, from collected episodes.

    Parameters:
        rollout (Rollout): experience from one frozen policy.
        policy: the stable-baselines3 policy that produced it, with the weights
            of the exported artifact.
        gamma, gae_lambda (float): discount and GAE parameters.

    Returns:
        RolloutBuffer, full, with advantages and returns computed.
    """
    device = policy.device
    observations = rollout.artifact.normalization.apply(rollout.observations).astype(np.float32)
    actions = rollout.actions.astype(np.float32)
    rewards = rollout.rewards.astype(np.float32)
    episode_starts = rollout.episode_starts.astype(np.float32)

    with th.no_grad():
        values, log_probs, _ = policy.evaluate_actions(
            th.as_tensor(observations, device=device),
            th.as_tensor(actions, device=device),
        )
    values = values.reshape(-1).cpu().numpy()
    log_probs = log_probs.reshape(-1).cpu().numpy()

    buffer = RolloutBuffer(
        buffer_size=rollout.n_steps,
        observation_space=policy.observation_space,
        action_space=policy.action_space,
        device=device,
        gamma=gamma,
        gae_lambda=gae_lambda,
        n_envs=1,
    )
    for i in range(rollout.n_steps):
        buffer.add(
            obs=observations[i].reshape(1, -1),
            action=actions[i].reshape(1, -1),
            reward=rewards[i : i + 1],
            episode_start=episode_starts[i : i + 1],
            value=th.as_tensor(values[i].reshape(1, 1), device=device),
            log_prob=th.as_tensor(log_probs[i].reshape(1), device=device),
        )

    # The observation following the last action is never logged, so the tail of
    # each episode is treated as terminal rather than bootstrapped.
    buffer.compute_returns_and_advantage(
        last_values=th.zeros((1, 1), device=device),
        dones=np.ones(1, dtype=np.float32),
    )
    return buffer


def rollout_statistics(rollout: Rollout) -> dict[str, float]:
    """Per-iteration summary of episode lengths, returns, divergences and bounds."""
    returns = rollout.returns
    return {
        "rollout/ep_rew_mean": float(returns.mean()),
        "rollout/ep_rew_min": float(returns.min()),
        "rollout/ep_rew_max": float(returns.max()),
        "rollout/ep_len_mean": float(np.mean(rollout.lengths)),
        "rollout/episodes": float(len(rollout.episodes)),
        "rollout/diverged": float(sum(ds.attrs["diverged"] for ds in rollout.episodes)),
        "rollout/failures": float(len(rollout.failures)),
        "rollout/fraction_at_bounds": rollout.fraction_at_bounds,
    }
