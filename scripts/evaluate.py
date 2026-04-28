"""CLI for evaluating a saved checkpoint.

Usage:
    python evaluate.py runs/<run>/checkpoints/final.pt
    python evaluate.py runs/<run>/checkpoints/final.pt --episodes 20
    python evaluate.py runs/<run>/checkpoints/final.pt --no-video
"""
from __future__ import annotations
import argparse

import numpy as np
import yaml

from qrl_cartpole import evaluate


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint.")
    parser.add_argument("checkpoint",      help="Path to .pt checkpoint file")
    parser.add_argument("--env-config",    default="configs/env.yaml")
    parser.add_argument("--agent-config",  default="configs/dqn.yaml")
    parser.add_argument("--episodes",      type=int, default=10)
    parser.add_argument("--no-video",      action="store_true")
    parser.add_argument("--output-dir",    default="eval_output")
    parser.add_argument("--seed",          type=int, default=0)
    args = parser.parse_args()

    returns = evaluate(
        checkpoint_path=args.checkpoint,
        env_cfg=_load_yaml(args.env_config),
        agent_cfg_raw=_load_yaml(args.agent_config),
        n_episodes=args.episodes,
        record_video=not args.no_video,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    arr = np.array(returns)
    print("─" * 45)
    for i, r in enumerate(returns):
        print(f"  episode {i + 1:>3}  return = {r:>7.1f}")
    print("─" * 45)
    print(f"mean={arr.mean():.1f}  std={arr.std():.1f}  "
          f"min={arr.min():.1f}  max={arr.max():.1f}")
    if not args.no_video:
        print(f"videos saved → {args.output_dir}/")


if __name__ == "__main__":
    main()
