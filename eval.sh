#!/bin/bash
# eval.sh

echo "Running Evaluation inside Docker..."

# $1 is the config file name
# $2 is the checkpoint path
python evaluate.py --config configs/${1} --checkpoint ${2}
