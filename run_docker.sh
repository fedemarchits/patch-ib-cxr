#!/bin/bash
# run_docker.sh

export WANDB_API_KEY="YOUR_WANDB_API"

PHYS_DIR=$(pwd)
LLM_CACHE_DIR="/llms"
DOCKER_INTERNAL_CACHE_DIR="/llms"

docker run \
    -v "$PHYS_DIR":/workspace \
    -v /datasets:/datasets \
    -v "$LLM_CACHE_DIR":"$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_HOME="$DOCKER_INTERNAL_CACHE_DIR" \
    -e WANDB_API_KEY="$WANDB_API_KEY" \
    --rm \
    --memory="30g" \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    patch_ib_img \
    "/workspace/train.sh" \
    "${1}" "$2"

