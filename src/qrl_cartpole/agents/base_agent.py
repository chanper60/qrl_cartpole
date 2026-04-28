from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
import torch


class BaseAgent(ABC):
    """Abstract interface every RL agent must implement.

    Trainer only depends on this interface, so any agent can be dropped in
    without modifying the training loop.
    """

    def __init__(self, obs_dim: int, action_dim: int, device: torch.device) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

    @abstractmethod
    def select_action(self, obs: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        """Return an integer action for each observation in the batch.

        Args:
            obs: shape (batch, obs_dim)
            epsilon: probability of a uniform-random action (ε-greedy)

        Returns:
            Integer action array, shape (batch,)
        """
        ...

    @abstractmethod
    def update(self, batch) -> dict[str, float]:
        """Run one gradient step on a sampled replay batch.

        Args:
            batch: ReplayBufferSamples namedtuple

        Returns:
            Dict of scalar metric names → values for logging.
        """
        ...

    def on_step(self, global_step: int) -> None:
        """Called once per env step after learning_starts.

        Override to implement step-based schedules (e.g. target-net sync).
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist all agent state (weights + optimizer) to path."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore all agent state from path."""
        ...
