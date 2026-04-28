from __future__ import annotations
from typing import NamedTuple

import numpy as np
import torch


class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor


class ReplayBuffer:
    """Fixed-capacity circular replay buffer.

    Overwrites the oldest transition once capacity is exceeded.
    """

    def __init__(self, capacity: int, obs_shape: tuple, device: torch.device) -> None:
        self.capacity = capacity
        self.device = device
        self._pos = 0
        self._full = False

        self._obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions = np.zeros((capacity, 1), dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: int | np.integer,
        reward: float,
        done: float,
    ) -> None:
        """Store one transition, overwriting the oldest if the buffer is full."""
        self._obs[self._pos] = obs
        self._next_obs[self._pos] = next_obs
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._dones[self._pos] = done
        self._pos = (self._pos + 1) % self.capacity
        self._full = self._full or self._pos == 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            ReplayBufferSamples with tensors on self.device.
        """
        size = self.capacity if self._full else self._pos
        idx = np.random.randint(0, size, size=batch_size)
        return ReplayBufferSamples(
            observations=torch.tensor(self._obs[idx], device=self.device),
            actions=torch.tensor(self._actions[idx], device=self.device),
            next_observations=torch.tensor(self._next_obs[idx], device=self.device),
            dones=torch.tensor(self._dones[idx], device=self.device),
            rewards=torch.tensor(self._rewards[idx], device=self.device),
        )

    def __len__(self) -> int:
        return self.capacity if self._full else self._pos
