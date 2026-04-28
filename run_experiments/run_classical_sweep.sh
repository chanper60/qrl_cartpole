#!/usr/bin/env bash
# Run the full classical parameter-scaling sweep: 100, 200, 500, 1000 params × 5 seeds.
set -euo pipefail
cd "$(dirname "$0")"

echo "════════════════════════════════════════════════"
echo "  Classical scaling sweep (4 configs × 5 seeds)"
echo "════════════════════════════════════════════════"

bash run_dqn_100_experiments.sh
bash run_dqn_200_experiments.sh
bash run_dqn_500_experiments.sh
bash run_dqn_1000_experiments.sh

echo ""
echo "════════════════════════════════════════════════"
echo "  Full sweep complete."
echo "  Launch TensorBoard:"
echo "    tensorboard --logdir runs/"
echo "════════════════════════════════════════════════"
