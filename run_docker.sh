#!/bin/bash
# run_docker.sh

PHYS_DIR=$(pwd)
LLM_CACHE_DIR="/llms"
DOCKER_INTERNAL_CACHE_DIR="/llms"

# Pass the WANDB key from the host to the container
export WANDB_API_KEY="YOUR_ACTUAL_API_KEY_HERE"

docker run \
    -v "$PHYS_DIR":/workspace \
    -v "$LLM_CACHE_DIR":"$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_HOME="$DOCKER_INTERNAL_CACHE_DIR" \
    -e WANDB_API_KEY="$WANDB_API_KEY" \
    --rm \
    --memory="30g" \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    patch_ib_img \
    "/workspace/train.sh" \
    "${1}"
