#!/bin/bash
# sbatch_script.sh

# Submit Model A (Baseline)
# Requesting RTX 3090 as discussed
sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 --job-name="patch-ib-cxr" -w faretra,moro232 run_docker.sh "model_A_baseline.yaml"
