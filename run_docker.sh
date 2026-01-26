#!/bin/bash
# run_docker.sh

export WANDB_API_KEY="YOUR_API_KEY_HERE"
export WANDB_HTTP_TIMEOUT=120

PHYS_DIR=$(pwd)
LLM_CACHE_DIR="/llms"
DOCKER_INTERNAL_CACHE_DIR="/llms"

chmod +x "$PHYS_DIR/train.sh"

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
    patch_ib_img \
    "/workspace/train.sh" "$@"