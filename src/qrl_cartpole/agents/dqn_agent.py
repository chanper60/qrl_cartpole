from __future__ import annotations
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base_agent import BaseAgent


class QNetwork(nn.Module):
    """Fully-connected Q-value network with configurable hidden layers."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int]) -> None:
        super().__init__()
        dims = [obs_dim, *hidden_dims, action_dim]
        layers: list[nn.Module] = []
        for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_d, out_d))
            if i < len(hidden_dims):
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DQNAgent(BaseAgent):
    """Deep Q-Network with experience replay and a periodically-synced target network.

    All hyperparameters are constructor arguments so configs drive behaviour
    without any code change.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        learning_rate: float = 2.5e-4,
        gamma: float = 0.99,
        tau: float = 1.0,
        target_network_frequency: int = 500,
        hidden_dims: List[int] | None = None,
        **_kwargs,
    ) -> None:
        super().__init__(obs_dim, action_dim, device)
        hidden_dims = hidden_dims or [120, 84]
        self.gamma = gamma
        self.tau = tau
        self.target_network_frequency = target_network_frequency

        self.q_network = QNetwork(obs_dim, action_dim, hidden_dims).to(device)
        self.target_network = QNetwork(obs_dim, action_dim, hidden_dims).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        for p in self.target_network.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

    def select_action(self, obs: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        if random.random() < epsilon:
            return np.array([random.randint(0, self.action_dim - 1) for _ in range(len(obs))])
        with torch.no_grad():
            q = self.q_network(torch.tensor(obs, dtype=torch.float32, device=self.device))
            return torch.argmax(q, dim=1).cpu().numpy()

    def update(self, batch) -> dict[str, float]:
        with torch.no_grad():
            target_max, _ = self.target_network(batch.next_observations).max(dim=1)
            td_target = batch.rewards.flatten() + self.gamma * target_max * (1.0 - batch.dones.flatten())

        old_val = self.q_network(batch.observations).gather(1, batch.actions).squeeze()
        loss = F.mse_loss(td_target, old_val)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"td_loss": loss.item(), "q_values": old_val.mean().item()}

    def on_step(self, global_step: int) -> None:
        if global_step % self.target_network_frequency == 0:
            for tp, qp in zip(self.target_network.parameters(), self.q_network.parameters()):
                tp.data.copy_(self.tau * qp.data + (1.0 - self.tau) * tp.data)

    def save(self, path: str) -> None:
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.q_network.load_state_dict(ckpt["q_network"])
        self.target_network.load_state_dict(ckpt["target_network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
