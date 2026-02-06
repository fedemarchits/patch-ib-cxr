#!/bin/bash
# eval.sh - Run evaluation with visualizations inside Docker
# Usage: ./eval.sh <config_name> <checkpoint_path> [--num_vis_samples N]
# Examples:
#   ./eval.sh model_c_combined.yaml logs/model_c/best_model.pt
#   ./eval.sh model_b_combined.yaml logs/model_b/best_model.pt --num_vis_samples 20

set -e  # Exit on error

CONFIG=${1}
CHECKPOINT=${2}
shift 2 2>/dev/null || true

# Derive output dir from checkpoint path
OUTPUT_DIR=$(dirname "$CHECKPOINT")

echo "Running Evaluation inside Docker..."
echo "  Config: configs/$CONFIG"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output: $OUTPUT_DIR"

python3 evaluate.py \
    --config "configs/$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --visualize \
    --num_vis_samples 10 \
    "$@"

echo "Done! Results saved to:"
echo "  - $OUTPUT_DIR/eval_results.txt"
echo "  - $OUTPUT_DIR/visualizations/"
