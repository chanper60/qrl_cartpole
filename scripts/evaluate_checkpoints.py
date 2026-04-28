"""Evaluate checkpoints and produce test-stability bar charts.

Two groups are supported:
  dqn   — 6 bars: dqn_tiny, dqn_100, dqn_200, dqn_500, dqn_1000, dqn
  qdqn  — 2 bars: qdqn_skolik, qdqn_2layer

Outputs (written to --out-dir):
  <group>_performance.csv   — per-seed numbers
  <group>_stability.png     — bar chart (cross-seed mean ± std)

Usage (from qrl_cartpole/ directory):
    python scripts/evaluate_checkpoints.py --group dqn
    python scripts/evaluate_checkpoints.py --group qdqn
    python scripts/evaluate_checkpoints.py --group dqn --checkpoint-step 150000 --episodes 10
"""
from __future__ import annotations
import argparse
import csv
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import yaml

from qrl_cartpole import evaluate


GROUPS: dict[str, list[str]] = {
    "dqn":  ["dqn_tiny", "dqn_100", "dqn_200", "dqn_500", "dqn_1000", "dqn"],
    "qdqn": ["qdqn_skolik", "qdqn_2layer"],
}

LABELS: dict[str, str] = {
    "dqn_tiny":    "DQN-tiny\n(44 params)",
    "dqn_100":     "DQN-100\n(100 params)",
    "dqn_200":     "DQN-200\n(198 params)",
    "dqn_500":     "DQN-500\n(499 params)",
    "dqn_1000":    "DQN-1000\n(996 params)",
    "dqn":         "DQN-full\n(10 934 params)",
    "qdqn_skolik": "QDQN Skolik\n5-layer (46 params)",
    "qdqn_2layer": "QDQN\n2-layer (22 params)",
}

COLORS: dict[str, str] = {
    "dqn_tiny":    "#ff7f0e",
    "dqn_100":     "#1f77b4",
    "dqn_200":     "#17becf",
    "dqn_500":     "#2ca02c",
    "dqn_1000":    "#9467bd",
    "dqn":         "#8c564b",
    "qdqn_skolik": "#2ca02c",
    "qdqn_2layer": "#d62728",
}

TITLES: dict[str, str] = {
    "dqn":  "Test stability — Classical DQN scaling sweep",
    "qdqn": "Test stability — Quantum DQN variants",
}


# ── Run discovery ─────────────────────────────────────────────────────────────

def parse_run_name(name: str) -> tuple[str, str] | None:
    m = re.match(r"^(.+)__seed_(\d+)__(.+)$", name)
    return (m.group(1), m.group(2)) if m else None


def find_checkpoints(runs_dir: str, ckpt_step: int) -> dict[str, dict[str, str]]:
    """Return {config_stem: {seed: checkpoint_path}}.

    Prefers the exact step checkpoint; falls back to final.pt if missing.
    When a seed has multiple run dirs, picks the latest by sorted name.
    """
    ckpt_file = f"step_{ckpt_step:07d}.pt"

    dirs_by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    for name in sorted(os.listdir(runs_dir)):
        path = os.path.join(runs_dir, name)
        if not os.path.isdir(path):
            continue
        parsed = parse_run_name(name)
        if parsed:
            dirs_by_key[parsed].append(path)

    result: dict[str, dict[str, str]] = defaultdict(dict)
    for (stem, seed), dir_list in dirs_by_key.items():
        found = False
        for d in reversed(dir_list):
            ckpt_path = os.path.join(d, "checkpoints", ckpt_file)
            if os.path.exists(ckpt_path):
                result[stem][seed] = ckpt_path
                found = True
                break
        if not found:
            for d in reversed(dir_list):
                final_path = os.path.join(d, "checkpoints", "final.pt")
                if os.path.exists(final_path):
                    result[stem][seed] = final_path
                    break
    return dict(result)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group",           choices=["dqn", "qdqn"], default="dqn",
                        help="Experiment group to evaluate (default: dqn)")
    parser.add_argument("--runs-dir",        default="runs/")
    parser.add_argument("--configs-dir",     default="configs/")
    parser.add_argument("--env-config",      default="configs/env.yaml")
    parser.add_argument("--out-dir",         default="results/")
    parser.add_argument("--checkpoint-step", type=int, default=150_000)
    parser.add_argument("--episodes",        type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.env_config) as f:
        env_cfg = yaml.safe_load(f)

    checkpoints = find_checkpoints(args.runs_dir, args.checkpoint_step)
    experiments  = [e for e in GROUPS[args.group] if e in checkpoints]

    if not experiments:
        print(f"No checkpoints found for group '{args.group}'. "
              f"Expected prefixes: {GROUPS[args.group]}")
        return

    rows: list[dict] = []

    for exp in experiments:
        print(f"\n{'='*55}")
        print(f"  {exp}  ({len(checkpoints[exp])} seeds)")
        print(f"{'='*55}")

        seed_means: list[float] = []

        for seed, ckpt_path in sorted(checkpoints[exp].items()):
            cfg_path = os.path.join(args.configs_dir, f"{exp}.yaml")
            with open(cfg_path) as f:
                agent_cfg = yaml.safe_load(f)

            returns = evaluate(
                checkpoint_path=ckpt_path,
                env_cfg=env_cfg,
                agent_cfg_raw=agent_cfg,
                n_episodes=args.episodes,
                record_video=False,
                output_dir=os.path.join(args.out_dir, "eval_videos"),
                seed=42,
            )

            ep_mean = float(np.mean(returns))
            ep_std  = float(np.std(returns))
            seed_means.append(ep_mean)

            rows.append({
                "experiment": exp,
                "seed":       seed,
                "ep_mean":    round(ep_mean, 2),
                "ep_std":     round(ep_std,  2),
                "ep_min":     round(float(np.min(returns)), 1),
                "ep_max":     round(float(np.max(returns)), 1),
                "n_episodes": len(returns),
            })
            print(f"  seed {seed:>5}  →  {ep_mean:>6.1f} ± {ep_std:>5.1f}  "
                  f"(min {np.min(returns):.0f}, max {np.max(returns):.0f})")

        cs_mean = float(np.mean(seed_means))
        cs_std  = float(np.std(seed_means))
        print(f"  {'cross-seed':>10}  →  {cs_mean:>6.1f} ± {cs_std:>5.1f}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, f"{args.group}_performance.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "seed",
                                               "ep_mean", "ep_std",
                                               "ep_min", "ep_max", "n_episodes"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved → {csv_path}")

    # ── Cross-seed summary table ───────────────────────────────────────────────
    by_exp: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_exp[r["experiment"]].append(r["ep_mean"])

    print(f"\n{'Experiment':<14}  {'cross-seed mean':>16}  {'cross-seed std':>14}  {'n seeds':>7}")
    print("-" * 58)
    for exp in experiments:
        vals = by_exp[exp]
        print(f"{exp:<14}  {np.mean(vals):>16.1f}  {np.std(vals):>14.1f}  {len(vals):>7}")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    cs_means = [float(np.mean(by_exp[e])) for e in experiments]
    cs_stds  = [float(np.std(by_exp[e]))  for e in experiments]
    colors   = [COLORS[e] for e in experiments]
    x        = np.arange(len(experiments))

    fig_w = max(8, len(experiments) * 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    ax.bar(x, cs_means, yerr=cs_stds, capsize=7,
           color=colors, alpha=0.82, width=0.55,
           error_kw={"linewidth": 1.8, "ecolor": "black"})

    for xi, (m, s) in enumerate(zip(cs_means, cs_stds)):
        ax.text(xi, m + s + 8, f"{m:.0f}±{s:.0f}", ha="center",
                fontsize=9, fontweight="bold")

    ax.axhline(450, color="gray",  linestyle="--", linewidth=1,
               alpha=0.7, label="threshold (450)")
    ax.axhline(500, color="green", linestyle=":",  linewidth=1,
               alpha=0.7, label="max return (500)")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(e, e) for e in experiments], fontsize=9.5)
    ax.set_ylabel("Mean return over 10 greedy episodes")
    ax.set_title(
        f"{TITLES[args.group]} — step {args.checkpoint_step // 1000}k\n"
        "Bar height = cross-seed mean,  error bar = cross-seed std"
    )
    ax.set_ylim(0, 570)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, f"{args.group}_stability.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"saved → {fig_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
