"""Print gate counts for the Skolik ansatz via qml.specs.

Usage:
    python scripts/circuit_specs.py
"""
import numpy as np
import pennylane as qml

from qrl_cartpole.agents.qdqn_skolik import QNetwork


def print_circuit_specs(n_qubits: int = 4, n_layers: int = 5) -> None:
    qnode = QNetwork(n_qubits=n_qubits, n_layers=n_layers).qlayer.qnode

    inputs  = np.zeros(n_qubits)
    weights = np.zeros((n_layers, n_qubits, 2))

    s = qml.specs(qnode)(inputs, weights)
    r = s.resources
    g = dict(r.gate_types)

    rx, ry, rz, cz = g.get("RX", 0), g.get("RY", 0), g.get("RZ", 0), g.get("CZ", 0)
    non_clifford = rx + ry + rz

    t_per_rotation = 3.0 * np.log2(1 / 1e-3)   # Solovay-Kitaev at ε=1e-3
    t_count = non_clifford * t_per_rotation
    t_depth = 3 * n_layers * t_per_rotation     # 3 rotation types per layer, sequential

    print(f"\nSkolik ansatz — {n_layers}-layer, {n_qubits}-qubit")
    print(f"{'─'*40}")
    print(f"  Total gates      : {r.num_gates}")
    print(f"  Circuit depth    : {r.depth}")
    print()
    print(f"  Non-Clifford:")
    print(f"    RX (obs-dep)   : {rx}")
    print(f"    RY (trainable) : {ry}")
    print(f"    RZ (trainable) : {rz}")
    print(f"    Total          : {non_clifford}")
    print()
    print(f"  Clifford:")
    print(f"    CZ             : {cz}")
    print()
    print(f"  T-gate cost (ε = 1e-3):")
    print(f"    T count        : {t_count:.0f}")
    print(f"    T depth        : {t_depth:.0f}")


if __name__ == "__main__":
    print("── Original circuit ─────────────────────────────")
    print_circuit_specs(n_qubits=4, n_layers=5)