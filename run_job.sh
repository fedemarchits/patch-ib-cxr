#!/bin/bash

#SBATCH --job-name=med_align       # Job name
#SBATCH --output=logs/job_%j.out   # Standard output log (checks print statements here)
#SBATCH --error=logs/job_%j.err    # Error log
#SBATCH --partition=all_usr        # ⚠️ CHECK THE GUIDE: Usually 'all_usr' or 'gpu'
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --cpus-per-task=8          # Request 8 CPUs
#SBATCH --mem=32G                  # Request 32GB RAM
#SBATCH --time=24:00:00            # Max time limit (24 hours)

# ------------------------------------------------------------------------------
# ⚠️ IMPORTANT: SET YOUR W&B API KEY HERE
# You can get your key from https://wandb.ai/settings
# ------------------------------------------------------------------------------
WANDB_API_KEY="YOUR_API_KEY_HERE"
export WANDB_API_KEY

# 1. Load Environment (Check the guide for specific modules!)
# Usually you activate your conda env here
source ~/anaconda3/etc/profile.d/conda.sh 
conda activate my_thesis_env

# 2. Debugging Info
echo "Running on host: $(hostname)"
echo "GPU info: $(nvidia-smi)"

# 3. Run Python
python main.py --config configs/model_a_baseline.yaml