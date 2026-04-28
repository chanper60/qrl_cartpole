"""Example 1 — run an experiment with a modified config (no code changes).

Shows the dqn_tiny pattern: parameter-matched classical baseline with
hidden_dims=[6] (44 params) vs the full DQN (10 934 params). The only
difference from dqn.yaml is one line: hidden_dims.

This demonstrates the design principle: "quantum off" or any architecture
change is a config change, not a code change.

Run:
    python examples/01_modify_existing_config.py
"""
import torch

from qrl_cartpole import DQNAgent, Trainer

device = torch.device("cpu")

# dqn_tiny: Linear(4→6) + ReLU + Linear(6→2) = 44 params
# vs dqn:   Linear(4→120) + ReLU + Linear(120→84) + ReLU + Linear(84→2) = 10 934 params
agent = DQNAgent(
    obs_dim=4,
    action_dim=2,
    device=device,
    hidden_dims=[6],        # ← only this line differs from dqn.yaml
    learning_rate=2.5e-4,
    gamma=0.99,
    tau=1.0,
    target_network_frequency=500,
)

trainer = Trainer(
    agent=agent,
    env_cfg={"env_id": "CartPole-v1", "num_envs": 1, "capture_video": False},
    training_cfg={
        "total_timesteps": 500,       # short demo; use 200_000 for real training
        "learning_starts": 100,
        "train_frequency": 10,
        "checkpoint_frequency": 250,
        "buffer_size": 1000,
        "batch_size": 32,
        "start_epsilon": 1.0,
        "end_epsilon": 0.01,
        "exploration_fraction": 0.5,
    },
    run_name="example_dqn_tiny",
    seed=42,
)
trainer.train()
print("checkpoint saved to runs/example_dqn_tiny/checkpoints/")
