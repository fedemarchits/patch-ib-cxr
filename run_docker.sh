#!/bin/bash
# run_docker.sh
# Usage:
#   ./run_docker.sh train.sh <config> [args...]     # Training
#   ./run_docker.sh eval.sh <config> <checkpoint>   # Evaluation

export WANDB_API_KEY="YOUR_API_KEY_HERE"
export WANDB_HTTP_TIMEOUT=120

PHYS_DIR=$(pwd)
LLM_CACHE_DIR="/llms"
DOCKER_INTERNAL_CACHE_DIR="/llms"

# First argument is the script to run (default: train.sh)
SCRIPT=${1:-train.sh}
shift

chmod +x "$PHYS_DIR/$SCRIPT"

docker run \
    -v "$PHYS_DIR":/workspace \
    -v /datasets:/datasets \
    -v "$LLM_CACHE_DIR":"$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_HOME="$DOCKER_INTERNAL_CACHE_DIR" \
    -e WANDB_API_KEY="$WANDB_API_KEY" \
    -e WANDB_HTTP_TIMEOUT="$WANDB_HTTP_TIMEOUT" \
    --rm \
    --memory="40g" \
    --shm-size="8g" \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    patch_ib_img2:flops \
    "/workspace/$SCRIPT" "$@"