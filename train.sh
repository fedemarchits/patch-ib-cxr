#!/bin/bash
# train.sh

# 1. Login to WandB (Optional: pass key as env var)
# wandb login $WANDB_API_KEY 

# 2. Run the code
# $1 is the config file passed from run_docker.sh
echo "Running Training inside Docker..."
python train.py --config configs/${1}
