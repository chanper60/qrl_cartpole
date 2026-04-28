# qrl-cartpole

Hybrid quantum-classical DQN agents for CartPole-v1.  
Part A–C of the QML Researcher assignment (Terra Quantum).

---

## Repository layout

```
qrl_cartpole/
│
├── src/qrl_cartpole/          ← pip-installable package (the library)
│   ├── agents/
│   │   ├── base_agent.py      # BaseAgent ABC — the only interface Trainer uses
│   │   ├── dqn_agent.py       # classical DQN baseline
│   │   └── qdqn_skolik.py     # Skolik 2022 quantum agent
│   ├── training/
│   │   └── trainer.py         # agent-agnostic training loop
│   ├── utils/
│   │   └── replay_buffer.py   # circular replay buffer
│   └── evaluate.py            # greedy evaluation function
│
├── configs/                   ← experiment configs (data, not code)
│   ├── env.yaml
│   ├── dqn.yaml               # full classical DQN (10 934 params)
│   ├── dqn_tiny.yaml          # parameter-matched baseline (44 params)
│   ├── dqn_100.yaml           # scaling sweep: ~100 params
│   ├── dqn_200.yaml           # scaling sweep: ~200 params
│   ├── dqn_500.yaml           # scaling sweep: ~500 params
│   ├── dqn_1000.yaml          # scaling sweep: ~1000 params
│   ├── qdqn_skolik.yaml       # Skolik 2022, 5-layer (46 params)
│   └── qdqn_2layer.yaml       # ablation: 2-layer (22 params)
│
├── examples/                  ← how-to-extend demos (not installed)
│   ├── 01_modify_existing_config.py
│   ├── 02_new_classical_agent.py
│   ├── 03_new_quantum_agent.py
│   └── configs/
│
├── scripts/                   ← CLI entry points and analysis
│   ├── train.py               # training CLI
│   ├── evaluate.py            # evaluation CLI
│   ├── plot_training_curves.py  # --group dqn|qdqn → training curve figures
│   ├── evaluate_checkpoints.py  # --group dqn|qdqn → stability bar charts
│   ├── t_gate_analysis.py
│   ├── circuit_specs.py
│   └── u3_eval.py
│
├── run_experiments/           ← reproducibility shell scripts
│   ├── run_dqn_experiments.sh
│   ├── run_dqn_tiny_experiments.sh
│   ├── run_qdqn_skolik_experiments.sh
│   └── run_qdqn_2layer_experiments.sh
│
├── tests/
│   └── test_agents.py
│
├── pyproject.toml
├── requirements.txt
└── setup_env.sh               ← one-command environment setup
```

---

## Installation

```bash
bash setup_env.sh          # creates .venv, installs deps + package
source .venv/bin/activate
```

Or if you already have the conda `ml_env` environment:

```bash
conda activate ml_env
pip install -e .
```

Verify:

```bash
python -c "from qrl_cartpole import DQNAgent, QuantumDQNAgent, Trainer; print('ok')"
```

---

## Reproducing figures

All scripts read from `runs/` and write to `results/`. Run from the `qrl_cartpole/` directory.

### Figure 1 — Classical DQN scaling sweep: training curves (2×3 grid)

```bash
python scripts/plot_training_curves.py --group dqn
# → results/dqn_training_curves.png
```

### Figure 2 — Quantum DQN variants: training curves (1×2 grid)

```bash
python scripts/plot_training_curves.py --group qdqn
# → results/qdqn_training_curves.png
```

Optional smoothing flag for both:

```bash
python scripts/plot_training_curves.py --group dqn --smooth 20
```

### Figure 3 — Classical DQN scaling sweep: test stability bar chart

```bash
python scripts/evaluate_checkpoints.py --group dqn
# → results/dqn_stability.png
# → results/dqn_performance.csv
```

### Figure 4 — Quantum DQN variants: test stability bar chart

```bash
python scripts/evaluate_checkpoints.py --group qdqn
# → results/qdqn_stability.png
# → results/qdqn_performance.csv
```

To evaluate at a different checkpoint step or with more episodes:

```bash
python scripts/evaluate_checkpoints.py --group dqn --checkpoint-step 100000 --episodes 20
```

---

## Training

**Single run:**
```bash
python scripts/train.py --agent-config configs/dqn.yaml --seed 1
python scripts/train.py --agent-config configs/qdqn_skolik.yaml --seed 1
```

**All seeds:**
```bash
bash run_experiments/run_dqn_experiments.sh
bash run_experiments/run_dqn_tiny_experiments.sh
bash run_experiments/run_qdqn_skolik_experiments.sh
bash run_experiments/run_qdqn_2layer_experiments.sh
```

Each run writes to `runs/<config>__seed_<N>__<timestamp>/`.

---

## Evaluation

```bash
python scripts/evaluate.py runs/<run>/checkpoints/final.pt
python scripts/evaluate.py runs/<run>/checkpoints/final.pt --episodes 20 --no-video
```

---

## Tests

```bash
pytest tests/ -v
```

Three tests in `test_agents.py`:

| Test | What it checks |
|---|---|
| `test_build_agent_unknown_type_raises` | Unknown `agent_type` raises `ValueError` immediately, not a cryptic error mid-training |
| `test_dqn_update_returns_expected_metric_keys` | `update()` returns a dict with `td_loss` and `q_values` — required by the TensorBoard logging loop |
| `test_quantum_agent_update_metric_keys` | Same contract for `QuantumDQNAgent` |

---

## Using the library in Python

```python
from qrl_cartpole import DQNAgent, QuantumDQNAgent, Trainer, build_agent
from qrl_cartpole import AGENT_REGISTRY, ReplayBuffer, evaluate

# Build an agent directly
agent = DQNAgent(obs_dim=4, action_dim=2, device=device, hidden_dims=[64, 64])

# Or via the registry (same as config-driven training)
agent = build_agent("qdqn_skolik", obs_dim=4, action_dim=2, device=device,
                    agent_cfg={"n_qubits": 4, "n_layers": 5})

# Train
trainer = Trainer(agent=agent, env_cfg=..., training_cfg=..., run_name="my_run", seed=42)
trainer.train()

# Evaluate a checkpoint
returns = evaluate("runs/my_run/checkpoints/final.pt", env_cfg, agent_cfg_raw)
```

---

## Extending the library

**Adding a new agent (classical or quantum):**

1. Subclass `BaseAgent` and implement `select_action`, `update`, `save`, `load`.
2. Register it: `AGENT_REGISTRY["my_agent"] = MyAgent`
3. Create `configs/my_agent.yaml` with `agent_type: my_agent`.
4. Run: `python scripts/train.py --agent-config configs/my_agent.yaml`

See `examples/02_new_classical_agent.py` and `examples/03_new_quantum_agent.py`
for complete worked examples.

**Switching quantum off (classical baseline):**

Change only the config — `agent_type: dqn` vs `agent_type: qdqn_skolik`.  
`Trainer`, `ReplayBuffer`, and `scripts/train.py` are unchanged.

---

## TensorBoard

```bash
tensorboard --logdir runs/
```

| Tag | Description |
|---|---|
| `charts/episodic_return` | return per episode |
| `charts/epsilon` | exploration rate |
| `charts/SPS` | env steps per second |
| `losses/td_loss` | Bellman loss |
| `losses/q_values` | mean Q-value of sampled batch |
