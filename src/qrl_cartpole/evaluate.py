"""Greedy evaluation of a saved agent checkpoint."""
from __future__ import annotations
import os
from typing import Any

import numpy as np
import torch
import gymnasium as gym

from qrl_cartpole.agents import build_agent

# Keys that live in agent YAML configs but belong to the training loop, not the agent.
_TRAINING_KEYS = {
    "total_timesteps", "learning_starts", "train_frequency",
    "checkpoint_frequency", "buffer_size", "batch_size",
    "start_epsilon", "end_epsilon", "exploration_fraction",
}


def evaluate(
    checkpoint_path: str,
    env_cfg: dict[str, Any],
    agent_cfg_raw: dict[str, Any],
    n_episodes: int = 10,
    record_video: bool = False,
    output_dir: str = "eval_output",
    seed: int = 0,
) -> list[float]:
    """Run greedy evaluation and return per-episode returns.

    Args:
        checkpoint_path: Path to a .pt checkpoint file.
        env_cfg: Dict with at least "env_id" key.
        agent_cfg_raw: Full agent YAML dict; training-only keys are stripped.
        n_episodes: Number of greedy episodes.
        record_video: Save video to output_dir if True.
        output_dir: Directory for video files.
        seed: Base seed for env.reset(seed=seed + episode_index).

    Returns:
        List of total returns, one per episode.
    """
    os.makedirs(output_dir, exist_ok=True)

    agent_type: str = agent_cfg_raw.get("agent_type", "dqn")
    agent_cfg = {
        k: v for k, v in agent_cfg_raw.items()
        if k not in _TRAINING_KEYS and k != "agent_type"
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    render_mode = "rgb_array" if record_video else None

    env = gym.make(env_cfg["env_id"], render_mode=render_mode)
    if record_video:
        env = gym.wrappers.RecordVideo(
            env, output_dir,
            episode_trigger=lambda _ep: True,
            name_prefix="eval",
        )

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(env.action_space.n)

    agent = build_agent(agent_type, obs_dim, action_dim, device, agent_cfg)
    agent.load(checkpoint_path)

    returns: list[float] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total, done = 0.0, False
        while not done:
            action = int(agent.select_action(obs[np.newaxis], epsilon=0.0)[0])
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = terminated or truncated
        returns.append(total)

    env.close()
    return returns
