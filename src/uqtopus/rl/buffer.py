"""
Rollout Buffer Filling

Transfers a Rollout into the stable-baselines3 buffer PPO trains from, normalized
with the snapshot frozen into the graph that produced it.
"""

from __future__ import annotations

import logging

import numpy as np
import torch as th
from stable_baselines3.common.buffers import RolloutBuffer

from .runner import Rollout

logger = logging.getLogger(__name__)


def normalized_observations(rollout: Rollout) -> np.ndarray:
    """The rollout's observations, normalized as the exported graph did."""
    return rollout.artifact.normalization.apply(rollout.observations)


def build_rollout_buffer(
    rollout: Rollout,
    policy,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    device: str | th.device | None = None,
) -> RolloutBuffer:
    """
    Build a filled RolloutBuffer from collected episodes.

    The buffer is sized to the rollout, since episodes vary in length.

    Parameters:
        rollout (Rollout): experience from one frozen policy.
        policy: the stable-baselines3 policy that produced it. Call before
            train(), while its weights still match the artifact.
        gamma, gae_lambda (float): discount and GAE parameters.
        device (str, torch.device or None): None uses the policy's device.

    Returns:
        RolloutBuffer, full, with advantages and returns already computed.
    """
    if not rollout.episodes:
        raise ValueError("the rollout holds no episodes")

    device = device or policy.device
    observations = normalized_observations(rollout).astype(np.float32)
    actions = rollout.actions.astype(np.float32)
    rewards = rollout.rewards.astype(np.float32)
    episode_starts = rollout.episode_starts

    n_steps = rollout.n_steps

    with th.no_grad():
        values, log_probs, _ = policy.evaluate_actions(
            th.as_tensor(observations, device=device),
            th.as_tensor(actions, device=device),
        )
    values = values.reshape(-1).cpu().numpy()
    log_probs = log_probs.reshape(-1).cpu().numpy()

    buffer = RolloutBuffer(
        buffer_size=n_steps,
        observation_space=policy.observation_space,
        action_space=policy.action_space,
        device=device,
        gamma=gamma,
        gae_lambda=gae_lambda,
        n_envs=1,
    )

    for i in range(n_steps):
        buffer.add(
            obs=observations[i].reshape(1, -1),
            action=actions[i].reshape(1, -1),
            reward=rewards[i : i + 1],
            episode_start=episode_starts[i : i + 1].astype(np.float32),
            value=th.as_tensor(values[i].reshape(1, 1), device=device),
            log_prob=th.as_tensor(log_probs[i].reshape(1), device=device),
        )

    # The observation following the last action is never logged, so the tail of
    # each episode is treated as terminal rather than bootstrapped.
    buffer.compute_returns_and_advantage(
        last_values=th.zeros((1, 1), device=device),
        dones=np.ones(1, dtype=np.float32),
    )

    if not buffer.full:
        raise RuntimeError(
            f"the buffer holds {buffer.pos} of {n_steps} steps; "
            "stable-baselines3 will refuse to train from it"
        )
    return buffer


def rollout_statistics(rollout: Rollout) -> dict[str, float]:
    """Per-iteration summary of episode lengths, returns and divergences."""
    lengths = rollout.lengths
    returns = rollout.returns
    diverged = sum(bool(ds.attrs.get("diverged")) for ds in rollout.episodes)
    return {
        "rollout/ep_rew_mean": float(returns.mean()),
        "rollout/ep_rew_min": float(returns.min()),
        "rollout/ep_rew_max": float(returns.max()),
        "rollout/ep_len_mean": float(np.mean(lengths)),
        "rollout/episodes": float(len(rollout.episodes)),
        "rollout/diverged": float(diverged),
        "rollout/failures": float(len(rollout.failures)),
    }
