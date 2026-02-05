#!/bin/bash
# train.sh
# Usage: ./train.sh <config_name> [--output_dir <dir>] [extra args...]
# Examples:
#   ./train.sh model_a_combined.yaml
#   ./train.sh model_b_combined.yaml --output_dir logs/model_b
#   ./train.sh model_c_combined.yaml --output_dir logs/model_c

CONFIG=${1:-model_a_combined.yaml}
shift

# Extract output_dir from args or use default based on config name
OUTPUT_DIR="logs"
for arg in "$@"; do
    if [[ "$prev_arg" == "--output_dir" ]]; then
        OUTPUT_DIR="$arg"
    fi
    prev_arg="$arg"
done

# If output_dir not specified, derive from config name
if [[ "$OUTPUT_DIR" == "logs" ]]; then
    # Extract model name from config (e.g., model_a_combined.yaml -> model_a)
    MODEL_NAME=$(echo "$CONFIG" | sed 's/.yaml//' | sed 's/_combined//' | sed 's/_patchib//')
    OUTPUT_DIR="logs/$MODEL_NAME"
fi

echo "=============================================="
echo "  Config: $CONFIG"
echo "  Output: $OUTPUT_DIR"
echo "=============================================="

# 1. Training
echo "🚀 Starting Training..."
python3 train.py --config "configs/$CONFIG" --output_dir "$OUTPUT_DIR" "$@"

# 2. Evaluation with Visualizations
echo "🏁 Training Finished. Starting Evaluation with Visualizations..."
python3 evaluate.py \
    --config "configs/$CONFIG" \
    --checkpoint "$OUTPUT_DIR/best_model.pt" \
    --output_dir "$OUTPUT_DIR" \
    --visualize \
    --num_vis_samples 10

echo "=============================================="
echo "  Done! Results saved to:"
echo "    - $OUTPUT_DIR/eval_results.txt"
echo "    - $OUTPUT_DIR/visualizations/"
echo "=============================================="