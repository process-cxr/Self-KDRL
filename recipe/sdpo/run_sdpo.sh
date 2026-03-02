#!/bin/bash
# SDPO Training Script
# This script launches SDPO training using the recipe pattern

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Set PYTHONPATH to include the project root
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Default configuration
EXPERIMENT_NAME="${1:-sdpo_$(date +%Y%m%d_%H%M%S)}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
DATA_PATH="${DATA_PATH:-datasets/tooluse}"

# Run SDPO training
python -m recipe.sdpo.main_sdpo \
    --config-name sdpo_trainer \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    data.train_files="${DATA_PATH}/train.parquet" \
    data.val_files="${DATA_PATH}/test.parquet" \
    trainer.project_name="SDPO" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    "${@:2}"
