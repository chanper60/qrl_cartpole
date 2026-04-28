#!/usr/bin/env bash
# Create a Python virtual environment and install all dependencies.
# Run once from the qrl_cartpole/ directory:
#   bash setup_env.sh

set -e

VENV_DIR=".venv"

echo "Creating virtual environment in $VENV_DIR ..."
python3 -m venv "$VENV_DIR"

echo "Activating ..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Installing dependencies from requirements.txt ..."
pip install -r requirements.txt

echo "Installing qrl-cartpole package in editable mode ..."
pip install -e .

echo ""
echo "Done. Activate the environment with:"
echo "  source $VENV_DIR/bin/activate"
