#!/bin/bash
# train.sh

# $1: Config file name (e.g., model_A_baseline.yaml)
# $2: Optional flags (e.g., --dry_run)

echo "Running Training inside Docker..."
echo "Config: configs/${1}"
echo "Flags: ${2}"

# Pass the second argument ($2) to the python script
python train.py --config configs/${1} ${2}