"""Entry point for training.

Usage:
    python train.py
    python train.py --seed 42
    python train.py --agent-config configs/dqn.yaml --seed 3
    python train.py --env-config configs/env.yaml --agent-config configs/qdqn_skolik.yaml
"""
from __future__ import annotations
import argparse
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import yaml
import gymnasium as gym

from qrl_cartpole import build_agent, Trainer


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


_TRAINING_KEYS = {
    "total_timesteps", "learning_starts", "train_frequency",
    "checkpoint_frequency", "buffer_size", "batch_size",
    "start_epsilon", "end_epsilon", "exploration_fraction",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DQN agent on CartPole-v1.")
    parser.add_argument("--env-config",   default="configs/env.yaml")
    parser.add_argument("--agent-config", default="configs/dqn.yaml")
    parser.add_argument("--seed",         type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Override total_timesteps from config")
    parser.add_argument("--resume-from",  type=str, default=None,
                        help="Path to .pt checkpoint to resume from")
    args = parser.parse_args()

    env_cfg = load_yaml(args.env_config)
    raw     = load_yaml(args.agent_config)

    agent_type: str = raw.pop("agent_type")
    training_cfg = {k: raw.pop(k) for k in list(raw) if k in _TRAINING_KEYS}
    agent_cfg    = raw

    if args.total_timesteps is not None:
        training_cfg["total_timesteps"] = args.total_timesteps

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  seed={args.seed}  agent={agent_type}")

    probe = gym.make(env_cfg["env_id"])
    obs_dim    = int(np.prod(probe.observation_space.shape))
    action_dim = int(probe.action_space.n)
    probe.close()

    agent = build_agent(agent_type, obs_dim, action_dim, device, agent_cfg)

    resume_step = 0
    if args.resume_from:
        agent.load(args.resume_from)
        basename = os.path.basename(args.resume_from)
        if basename.startswith("step_"):
            try:
                resume_step = int(basename.replace("step_", "").replace(".pt", ""))
            except ValueError:
                pass
        print(f"resumed from {args.resume_from}  (start_step={resume_step})")

    config_stem = os.path.splitext(os.path.basename(args.agent_config))[0]
    run_name    = f"{config_stem}__seed_{args.seed}__{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"

    trainer = Trainer(
        agent=agent,
        env_cfg=env_cfg,
        training_cfg=training_cfg,
        run_name=run_name,
        seed=args.seed,
    )
    trainer.train(start_step=resume_step)


if __name__ == "__main__":
    main()
