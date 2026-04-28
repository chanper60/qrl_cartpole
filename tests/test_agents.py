"""Tests for agent implementations and the build_agent registry.

What breaks when these tests fail
----------------------------------
test_build_agent_unknown_type_raises
    Unknown agent_type strings succeed silently instead of failing fast.
    A config typo would cause a delayed, cryptic error during training.

test_dqn_update_returns_expected_metric_keys
    update() omits "td_loss" or "q_values". Trainer's TensorBoard loop
    logs nothing and divergence is invisible.

test_quantum_agent_update_metric_keys
    Same as above for the quantum agent.
"""
import numpy as np
import pytest
import torch

from qrl_cartpole import DQNAgent, QuantumDQNAgent, build_agent
from qrl_cartpole.utils.replay_buffer import ReplayBuffer

CPU = torch.device("cpu")
OBS_DIM, ACTION_DIM = 4, 2


def _dqn() -> DQNAgent:
    return DQNAgent(obs_dim=OBS_DIM, action_dim=ACTION_DIM, device=CPU, hidden_dims=[8])


def _qdqn() -> QuantumDQNAgent:
    return QuantumDQNAgent(obs_dim=OBS_DIM, action_dim=ACTION_DIM, device=CPU,
                           n_qubits=4, n_layers=1)


def _fake_batch(n: int = 8):
    buf = ReplayBuffer(capacity=n, obs_shape=(OBS_DIM,), device=CPU)
    for _ in range(n):
        buf.add(np.zeros(OBS_DIM, dtype=np.float32),
                np.zeros(OBS_DIM, dtype=np.float32),
                action=0, reward=1.0, done=0.0)
    return buf.sample(n)


def test_build_agent_unknown_type_raises() -> None:
    """build_agent must raise ValueError with a message naming the bad key."""
    with pytest.raises(ValueError, match="Unknown agent type"):
        build_agent("not_a_real_agent", OBS_DIM, ACTION_DIM, CPU, {})


def test_dqn_update_returns_expected_metric_keys() -> None:
    """update() must return a dict containing 'td_loss' and 'q_values'."""
    agent = _dqn()
    metrics = agent.update(_fake_batch())
    assert "td_loss" in metrics
    assert "q_values" in metrics


def test_quantum_agent_update_metric_keys() -> None:
    """QuantumDQNAgent.update() must return a dict with 'td_loss' and 'q_values'.

    Trainer iterates metrics.items() to write TensorBoard scalars. Missing keys
    mean the quantum agent's loss curves are absent — divergence is invisible.
    """
    agent = _qdqn()
    metrics = agent.update(_fake_batch())
    assert "td_loss" in metrics
    assert "q_values" in metrics