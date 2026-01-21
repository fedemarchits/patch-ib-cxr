#!/bin/bash
# train.sh

# $1: Config file name (e.g., model_A_baseline.yaml)
# $2: Optional flags (e.g., --dry_run)

export WANDB_API_KEY=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

echo "Running Training inside Docker..."
echo "Config: configs/${1}"
echo "Flags: ${2}"

# Pass the second argument ($2) to the python script
python3 train.py --config configs/${1} ${2}