"""Plot mean ± std training curves across seeds.

Two groups are supported:
  dqn   — 6 panels: dqn_tiny, dqn_100, dqn_200, dqn_500, dqn_1000, dqn (2×3 grid)
  qdqn  — 2 panels: qdqn_skolik, qdqn_2layer (1×2 grid)

Usage:
    python scripts/plot_training_curves.py --group dqn
    python scripts/plot_training_curves.py --group qdqn
    python scripts/plot_training_curves.py --group dqn --smooth 20
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

GROUPS: dict[str, list[dict]] = {
    "dqn": [
        {"prefix": "dqn_tiny",  "label": "DQN-tiny  (44 params)",   "color": "#ff7f0e"},
        {"prefix": "dqn_100",   "label": "DQN-100   (100 params)",   "color": "#1f77b4"},
        {"prefix": "dqn_200",   "label": "DQN-200   (198 params)",   "color": "#17becf"},
        {"prefix": "dqn_500",   "label": "DQN-500   (499 params)",   "color": "#2ca02c"},
        {"prefix": "dqn_1000",  "label": "DQN-1000  (996 params)",   "color": "#9467bd"},
        {"prefix": "dqn",       "label": "DQN-full  (10 934 params)", "color": "#8c564b"},
    ],
    "qdqn": [
        {"prefix": "qdqn_skolik", "label": "QDQN Skolik 5-layer (46 params)", "color": "#2ca02c"},
        {"prefix": "qdqn_2layer", "label": "QDQN 2-layer (22 params)",        "color": "#d62728"},
    ],
}

TITLES = {
    "dqn":  "CartPole-v1 — Classical DQN scaling sweep (mean ± 1 std across seeds)",
    "qdqn": "CartPole-v1 — Quantum DQN variants (mean ± 1 std across seeds)",
}

LAYOUTS = {
    "dqn":  (2, 3),
    "qdqn": (1, 2),
}

DEFAULT_OUTPUTS = {
    "dqn":  "results/dqn_training_curves.png",
    "qdqn": "results/qdqn_training_curves.png",
}

TAG = "charts/episodic_return"
DEFAULT_SMOOTH = 15


def _load_scalars(event_dir: str) -> tuple[np.ndarray, np.ndarray]:
    ea = EventAccumulator(event_dir, size_guidance={TAG: 0})
    ea.Reload()
    if TAG not in ea.Tags().get("scalars", []):
        return np.array([]), np.array([])
    events = ea.Scalars(TAG)
    steps  = np.array([e.step  for e in events], dtype=np.float64)
    values = np.array([e.value for e in events], dtype=np.float64)
    return steps, values


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    pad = np.full(window - 1, values[0])
    return np.convolve(np.concatenate([pad, values]), kernel, mode="valid")


def _collect_experiment(
    runs_dir: Path,
    prefix: str,
    grid_points: int = 500,
    smooth: int = DEFAULT_SMOOTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    run_dirs = sorted(
        d for d in runs_dir.iterdir()
        if d.is_dir() and d.name.startswith(prefix + "__seed_")
    )
    if not run_dirs:
        return None

    seed_curves: list[tuple[np.ndarray, np.ndarray]] = []
    for d in run_dirs:
        steps, values = _load_scalars(str(d))
        if len(steps) < 2:
            print(f"  [skip] {d.name} — fewer than 2 data points")
            continue
        values = _smooth(values, smooth)
        seed_curves.append((steps, values))

    if not seed_curves:
        return None

    max_step = min(s[-1] for s, _ in seed_curves)
    grid = np.linspace(0, max_step, grid_points)
    interp = np.stack([np.interp(grid, s, v) for s, v in seed_curves], axis=0)
    return grid, interp.mean(axis=0), interp.std(axis=0), len(seed_curves)


def _format_steps(x: float, _pos: object) -> str:
    return f"{x/1e3:.0f}k" if x >= 1_000 else f"{x:.0f}"


def plot(
    group: str = "dqn",
    runs_dir: str = "runs",
    output: str | None = None,
    smooth: int = DEFAULT_SMOOTH,
) -> None:
    if group not in GROUPS:
        raise ValueError(f"Unknown group '{group}'. Choose from: {list(GROUPS)}")

    runs_path = Path(runs_dir)
    if not runs_path.exists():
        raise FileNotFoundError(f"runs directory not found: {runs_dir}")

    experiments = GROUPS[group]
    nrows, ncols = LAYOUTS[group]
    fig_w = ncols * 6
    fig_h = nrows * 4

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                             sharex=False, sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    fig.suptitle(TITLES[group], fontsize=13, y=1.01)

    for ax, exp in zip(axes_flat, experiments):
        result = _collect_experiment(runs_path, exp["prefix"], smooth=smooth)

        if result is None:
            print(f"[skip] no runs found for prefix '{exp['prefix']}'")
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="grey", fontsize=12)
            ax.set_title(exp["label"], fontsize=10)
            continue

        steps, mean, std, n_seeds = result
        color = exp["color"]

        ax.plot(steps, mean, color=color, linewidth=2.0)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.22,
                        label=f"±1 std  (n={n_seeds})")
        ax.axhline(500, color="black", linewidth=0.8, linestyle="--", alpha=0.35)

        ax.set_title(exp["label"], fontsize=10, pad=6)
        ax.set_xlabel("Environment steps", fontsize=9)
        ax.set_ylabel("Episodic return", fontsize=9)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_format_steps))
        ax.set_ylim(0, 520)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, framealpha=0.8)

    # hide any unused axes (e.g. if group has fewer panels than grid cells)
    for ax in axes_flat[len(experiments):]:
        ax.set_visible(False)

    fig.tight_layout()

    out_path = Path(output or DEFAULT_OUTPUTS[group])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved → {out_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from TensorBoard events.")
    parser.add_argument("--group",    choices=["dqn", "qdqn"], default="dqn",
                        help="Which experiment group to plot (default: dqn)")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output",   default=None,
                        help="Override output path (default: results/<group>_training_curves.png)")
    parser.add_argument("--smooth",   type=int, default=DEFAULT_SMOOTH)
    args = parser.parse_args()
    plot(group=args.group, runs_dir=args.runs_dir, output=args.output, smooth=args.smooth)


if __name__ == "__main__":
    main()
