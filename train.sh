#!/bin/bash
# train.sh

CONFIG=${1:-model_A_baseline.yaml}
shift 

echo "🚀 Starting Training..."
python3 train.py --config "configs/$CONFIG" "$@"

# This part runs AUTOMATICALLY after train.py finishes
echo "🏁 Training Finished. Starting Evaluation..."
python3 evaluate.py --config "configs/$CONFIG" --checkpoint "logs/best_model.pt"