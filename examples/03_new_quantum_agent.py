"""Example 3 — add a new quantum ansatz without modifying library code.

Implements HEAAgent: a hardware-efficient ansatz with RY single-qubit
rotations and a CNOT ladder for entanglement. This differs from the Skolik
ansatz (RX+RY+RZ encoding + CZ ring) to illustrate the extension point.

Circuit per layer:
  - RX(arctan(w_input * obs_i)) on each qubit  (data reuploading)
  - RY(θ_i) on each qubit                      (variational)
  - CNOT(i, i+1) ladder                         (entanglement, Clifford)

Observables: ⟨Z_0⟩, ⟨Z_1⟩  (single-qubit, not joint ZZ)

The pattern is identical to adding a classical agent (Example 2):
subclass BaseAgent → implement four methods → add one line to AGENT_REGISTRY.

Run:
    python examples/03_new_quantum_agent.py
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

from qrl_cartpole import BaseAgent, AGENT_REGISTRY, Trainer
from qrl_cartpole.utils.replay_buffer import ReplayBufferSamples


class HEAQNetwork(nn.Module):
    """Hardware-efficient ansatz: RY rotations + CNOT ladder."""

    def __init__(self, n_qubits: int = 4, n_layers: int = 3) -> None:
        super().__init__()
        self.n_qubits = n_qubits

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list:
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.RX(inputs[i], wires=i)           # data reuploading
                for i in range(n_qubits):
                    qml.RY(weights[layer, i], wires=i)   # variational
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])            # CNOT ladder
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

        self.qlayer = qml.qnn.TorchLayer(
            circuit,
            {"weights": (n_layers, n_qubits)},
            init_method=lambda t: nn.init.uniform_(t, 0, 2 * torch.pi),
        )
        self.w_input = Parameter(torch.ones(n_qubits))
        self.w_output = Parameter(torch.full((2,), 50.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.atan(x * self.w_input)
        if x.ndim == 1:
            out = self.qlayer(x)
        else:
            out = torch.stack([self.qlayer(x[i]) for i in range(x.shape[0])])
        return (1.0 + out) / 2.0 * self.w_output


class HEAAgent(BaseAgent):
    """DQN with a hardware-efficient ansatz (RY + CNOT ladder)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        n_qubits: int = 4,
        n_layers: int = 3,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        tau: float = 1.0,
        target_network_frequency: int = 30,
        **_kwargs,
    ) -> None:
        super().__init__(obs_dim, action_dim, device)
        self.gamma = gamma
        self.tau = tau
        self.target_network_frequency = target_network_frequency

        self.q_network = HEAQNetwork(n_qubits, n_layers).to(device)
        self.target_network = HEAQNetwork(n_qubits, n_layers).to(device)
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

    def update(self, batch: ReplayBufferSamples) -> dict[str, float]:
        with torch.no_grad():
            target_max, _ = self.target_network(batch.next_observations).max(dim=1)
            td_target = (
                batch.rewards.flatten()
                + self.gamma * target_max * (1.0 - batch.dones.flatten())
            )
        q_vals = self.q_network(batch.observations).gather(1, batch.actions).squeeze()
        loss = F.smooth_l1_loss(td_target, q_vals)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"td_loss": loss.item(), "q_values": q_vals.mean().item()}

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


# Register under a new key — no changes to library code.
AGENT_REGISTRY["hea"] = HEAAgent


if __name__ == "__main__":
    device = torch.device("cpu")
    agent = HEAAgent(obs_dim=4, action_dim=2, device=device, n_qubits=4, n_layers=3)
    trainer = Trainer(
        agent=agent,
        env_cfg={"env_id": "CartPole-v1", "num_envs": 1, "capture_video": False},
        training_cfg={
            "total_timesteps": 200,        # short demo; quantum is slow per step
            "learning_starts": 50,
            "train_frequency": 10,
            "checkpoint_frequency": 100,
            "buffer_size": 500,
            "batch_size": 4,
            "start_epsilon": 1.0,
            "end_epsilon": 0.1,
            "exploration_fraction": 0.5,
        },
        run_name="example_hea",
        seed=42,
    )
    trainer.train()
    print("done")
