"""Example 2 — add a new classical agent without modifying library code.

Steps:
  1. Subclass BaseAgent and implement the four required methods:
       select_action, update, save, load
     (on_step is optional — override only if you have a target network or
     similar step-based schedule).
  2. Register the class in AGENT_REGISTRY under a new key.
  3. Use the class directly or via build_agent("linear_q", ...).

This example implements LinearQAgent: a single linear layer Q-network
(obs → Q-values, no hidden layers). It is intentionally minimal to keep
the example readable — the pattern is what matters, not the architecture.

Run:
    python examples/02_new_classical_agent.py
"""
from __future__ import annotations
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from qrl_cartpole import BaseAgent, AGENT_REGISTRY, Trainer
from qrl_cartpole.utils.replay_buffer import ReplayBufferSamples


class LinearQAgent(BaseAgent):
    """Minimal Q-agent: single linear layer, no hidden layers."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        **_kwargs,
    ) -> None:
        super().__init__(obs_dim, action_dim, device)
        self.gamma = gamma
        self.q = nn.Linear(obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.q.parameters(), lr=learning_rate)

    def select_action(self, obs: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        if random.random() < epsilon:
            return np.array([random.randint(0, self.action_dim - 1) for _ in range(len(obs))])
        with torch.no_grad():
            q = self.q(torch.tensor(obs, dtype=torch.float32, device=self.device))
            return torch.argmax(q, dim=1).cpu().numpy()

    def update(self, batch: ReplayBufferSamples) -> dict[str, float]:
        with torch.no_grad():
            target_max, _ = self.q(batch.next_observations).max(dim=1)
            td_target = (
                batch.rewards.flatten()
                + self.gamma * target_max * (1.0 - batch.dones.flatten())
            )
        q_vals = self.q(batch.observations).gather(1, batch.actions).squeeze()
        loss = F.mse_loss(td_target, q_vals)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"td_loss": loss.item(), "q_values": q_vals.mean().item()}

    def save(self, path: str) -> None:
        torch.save(
            {"q": self.q.state_dict(), "optimizer": self.optimizer.state_dict()},
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.q.load_state_dict(ckpt["q"])
        self.optimizer.load_state_dict(ckpt["optimizer"])


# Register under a new key — Trainer and build_agent can now use "linear_q".
AGENT_REGISTRY["linear_q"] = LinearQAgent


if __name__ == "__main__":
    device = torch.device("cpu")
    agent = LinearQAgent(obs_dim=4, action_dim=2, device=device)
    trainer = Trainer(
        agent=agent,
        env_cfg={"env_id": "CartPole-v1", "num_envs": 1, "capture_video": False},
        training_cfg={
            "total_timesteps": 500,
            "learning_starts": 50,
            "train_frequency": 5,
            "checkpoint_frequency": 250,
            "buffer_size": 1000,
            "batch_size": 32,
            "start_epsilon": 1.0,
            "end_epsilon": 0.1,
            "exploration_fraction": 0.5,
        },
        run_name="example_linear_q",
        seed=42,
    )
    trainer.train()
    print("done")
