#!/usr/bin/env bash
# Run Skolik 2022 exact agent — 5 seeds × 500k timesteps
set -euo pipefail
cd "$(dirname "$0")/.."   # always run from project root, wherever this script is called from

ENV_CONFIG="configs/env.yaml"
AGENT_CONFIG="configs/qdqn_2layer.yaml"

SEEDS=(2574 3545 5181)

for seed in "${SEEDS[@]}"; do
    echo "=== qdqn_skolik  seed=${seed} ==="
    python scripts/train.py \
        --env-config   "${ENV_CONFIG}"   \
        --agent-config "${AGENT_CONFIG}" \
        --seed         "${seed}"
done
