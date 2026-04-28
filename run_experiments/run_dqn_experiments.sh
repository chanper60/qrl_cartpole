#!/usr/bin/env bash
# Run DQN on CartPole-v1 with 5 fixed seeds sequentially.
# Seeds are hardcoded here so results are fully reproducible.
set -euo pipefail
cd "$(dirname "$0")/.."   # always run from project root, wherever this script is called from

ENV_CONFIG="configs/env.yaml"
AGENT_CONFIG="configs/dqn.yaml"

SEEDS=(2574 8805 3545 5181 8071)

echo "════════════════════════════════════════════════"
echo "  DQN CartPole — ${#SEEDS[@]} seeds: ${SEEDS[*]}"
echo "════════════════════════════════════════════════"

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "──────────────────────────────────────────────"
    echo "  seed = ${seed}"
    echo "──────────────────────────────────────────────"
    python scripts/train.py \
        --env-config   "${ENV_CONFIG}"   \
        --agent-config "${AGENT_CONFIG}" \
        --seed         "${seed}"
done

echo ""
echo "════════════════════════════════════════════════"
echo "  All runs complete."
echo "  Launch TensorBoard:"
echo "    tensorboard --logdir runs/"
echo "════════════════════════════════════════════════"
