"""QDQN — Exact Skolik et al. 2022 architecture on CartPole-v1.

Circuit (per Skolik Figure 1 + Table 2 best config):
  - 4 qubits, 5 layers, data reuploading ON
  - Encoding : RX on each qubit before every layer (data reuploading)
  - Variational: RY + RZ per qubit
  - Entanglement: CZ daisy-chain ring  (q0→q1→q2→q3→q0)
  - Observables : ⟨Z₀⊗Z₁⟩  and  ⟨Z₂⊗Z₃⟩  (joint ZZ, not individual Z)

Preprocessing  : inputs = atan( w_input * raw_obs )   (trainable w_input per qubit)
Postprocessing : Q = w_output * (1 + ⟨ZZ⟩) / 2       (trainable w_output per action)

Optimizer: RMSprop with three separate learning rates:
  - VQC weights : lr = 0.001
  - w_input     : lr = 0.01
  - w_output    : lr = 0.1

Source: Skolik et al. "Quantum agents in the Gym" (2022) arXiv:2103.15084, Table 2.
Code reference: github.com/qdevpsi3/qrl-dqn-gym (modernised for PennyLane 0.44 + Gymnasium 1.3).
"""
from __future__ import annotations
import random

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

from .base_agent import BaseAgent

N_QUBITS = 4
N_ACTIONS = 2


class QNetwork(nn.Module):
    """VQC Q-network following Skolik et al. 2022 (best configuration)."""

    def __init__(self, n_qubits: int, n_layers: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits

        dev = qml.device("lightning.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="adjoint")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list:
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.RX(inputs[i], wires=i)
                for i in range(n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                for i in range(n_qubits):
                    qml.CZ(wires=[i, (i + 1) % n_qubits])
            return [
                qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
                qml.expval(qml.PauliZ(2) @ qml.PauliZ(3)),
            ]

        weight_shapes = {"weights": (n_layers, n_qubits, 2)}
        self.qlayer = qml.qnn.TorchLayer(
            circuit,
            weight_shapes,
            init_method=lambda t: nn.init.uniform_(t, 0, 2 * torch.pi),
        )

        # Trainable input scaling: atan(obs * w_input) uses full [-π/2, π/2] range.
        self.w_input = Parameter(torch.empty(n_qubits))
        nn.init.normal_(self.w_input)

        # Initialise output weights near expected return so Q-values start in range.
        self.w_output = Parameter(torch.empty(N_ACTIONS))
        nn.init.normal_(self.w_output, mean=90.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.atan(x * self.w_input)
        if x.ndim == 1:
            out = self.qlayer(x)
        else:
            out = torch.stack([self.qlayer(x[i]) for i in range(x.shape[0])])
        out = (1.0 + out) / 2.0
        return out * self.w_output


class QuantumDQNAgent(BaseAgent):
    """DQN with Skolik et al. 2022 VQC (data reuploading, ZZ observables).

    Supports variable n_layers; the published best config is n_layers=5.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        n_qubits: int = N_QUBITS,
        n_layers: int = 5,
        learning_rate: float = 0.001,
        learning_rate_input: float = 0.01,
        learning_rate_output: float = 0.1,
        gamma: float = 0.99,
        tau: float = 1.0,
        target_network_frequency: int = 500,
        **_kwargs,
    ) -> None:
        super().__init__(obs_dim, action_dim, device)
        self.gamma = gamma
        self.tau = tau
        self.target_network_frequency = target_network_frequency

        self.q_network = QNetwork(n_qubits, n_layers).to(device)
        self.target_network = QNetwork(n_qubits, n_layers).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        for p in self.target_network.parameters():
            p.requires_grad = False

        self.optimizer = optim.RMSprop(
            [
                {"params": self.q_network.qlayer.parameters(), "lr": learning_rate},
                {"params": [self.q_network.w_input],           "lr": learning_rate_input},
                {"params": [self.q_network.w_output],          "lr": learning_rate_output},
            ],
            lr=learning_rate,
        )

    def select_action(self, obs: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        if random.random() < epsilon:
            return np.array([random.randint(0, self.action_dim - 1) for _ in range(len(obs))])
        with torch.no_grad():
            q = self.q_network(torch.tensor(obs, dtype=torch.float32, device=self.device))
            return torch.argmax(q, dim=1).cpu().numpy()

    def update(self, batch) -> dict[str, float]:
        with torch.no_grad():
            target_max, _ = self.target_network(batch.next_observations).max(dim=1)
            td_target = (
                batch.rewards.flatten()
                + self.gamma * target_max * (1.0 - batch.dones.flatten())
            )
        old_val = self.q_network(batch.observations).gather(1, batch.actions).squeeze()
        loss = F.smooth_l1_loss(td_target, old_val)

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
