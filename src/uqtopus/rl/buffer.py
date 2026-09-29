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
    truncated: bool = True,
) -> RolloutBuffer:
    """
    Build a filled RolloutBuffer, sized to the rollout, from collected episodes.

    Parameters:
        rollout (Rollout): experience from one frozen policy.
        policy: the stable-baselines3 policy that produced it, with the weights
            of the exported artifact.
        gamma, gae_lambda (float): discount and GAE parameters.
        truncated (bool): True when the end of a run cuts the task short rather
            than reaching a terminal state. The last row of each episode that
            ran to the end then only supplies the state its return is
            bootstrapped from. A diverged episode always ends terminal.

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

    keep = np.ones(rollout.n_steps, dtype=bool)
    if truncated:
        ends = np.cumsum(rollout.lengths) - 1
        for end, length, episode in zip(ends, rollout.lengths, rollout.episodes):
            if episode.attrs["diverged"]:
                continue
            keep[end] = False
            if length > 1:
                rewards[end - 1] += gamma * values[end]

    buffer = RolloutBuffer(
        buffer_size=int(keep.sum()),
        observation_space=policy.observation_space,
        action_space=policy.action_space,
        device=device,
        gamma=gamma,
        gae_lambda=gae_lambda,
        n_envs=1,
    )
    for i in np.flatnonzero(keep):
        buffer.add(
            obs=observations[i].reshape(1, -1),
            action=actions[i].reshape(1, -1),
            reward=rewards[i : i + 1],
            episode_start=episode_starts[i : i + 1],
            value=th.as_tensor(values[i].reshape(1, 1), device=device),
            log_prob=th.as_tensor(log_probs[i].reshape(1), device=device),
        )

    # Every episode ends terminal in the buffer
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
