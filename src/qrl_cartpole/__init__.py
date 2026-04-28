"""qrl_cartpole — hybrid quantum-classical DQN agents for CartPole-v1.

Public API
----------
>>> from qrl_cartpole import DQNAgent, QuantumDQNAgent, Trainer, build_agent
>>> from qrl_cartpole import AGENT_REGISTRY, ReplayBuffer, evaluate
"""

from qrl_cartpole.agents import AGENT_REGISTRY, BaseAgent, DQNAgent, build_agent
from qrl_cartpole.agents.qdqn_skolik import QuantumDQNAgent
from qrl_cartpole.evaluate import evaluate
from qrl_cartpole.training.trainer import Trainer
from qrl_cartpole.utils.replay_buffer import ReplayBuffer, ReplayBufferSamples

__all__ = [
    "BaseAgent",
    "DQNAgent",
    "QuantumDQNAgent",
    "AGENT_REGISTRY",
    "build_agent",
    "evaluate",
    "Trainer",
    "ReplayBuffer",
    "ReplayBufferSamples",
]
