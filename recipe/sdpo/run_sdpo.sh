#!/usr/bin/env bash
# SDPO Training Script
# This script launches SDPO training using Ray job submit
#
# Usage:
#   1. Edit the configuration below as needed
#   2. Run: bash run_sdpo.sh
#
# Prerequisites:
#   1. Ray cluster running (edit RAY_ADDRESS below)
#   2. Data preprocessed to parquet format (edit DATA_PATH below)

set -xeuo pipefail

# =============================================================================
# USER CONFIGURATION - Edit these values
# =============================================================================

# Experiment name suffix
SUFFIX="local_sdpo"

# Hardware settings
NNODES=1
N_GPUS_PER_NODE=1

# Model and Data paths
MODEL_PATH="/home/work/cxr/models/Qwen3-4B"
DATA_PATH="/home/work/cxr/Self-KDRL/datasets/tooluse"
TRAIN_FILE="${DATA_PATH}/train.parquet"
TEST_FILE="${DATA_PATH}/test.parquet"

# Hyperparameters
TRAIN_BATCH_SIZE=32
ROLLOUT_N=8
LR=1e-5
ALPHA=0.5
DISTILLATION_TOPK=100

# Performance settings
SP_SIZE=1
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=8192

# Ray settings
RAY_ADDRESS="http://localhost:8265"

# =============================================================================
# INTERNAL CONFIGURATION - Usually no need to edit
# =============================================================================

# Get script directory (works from any location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKING_DIR="${SCRIPT_DIR}"
RUNTIME_ENV="${WORKING_DIR}/config/runtime_env.yaml"

# Get project root directory (two levels up from script)
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Update PYTHONPATH in runtime_env.yaml
sed -i "s|__PROJECT_ROOT__|${PROJECT_ROOT}|g" "${RUNTIME_ENV}"

PROJECT_NAME="SDPO"

# =============================================================================
# EXPERIMENT NAMING
# =============================================================================

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="SDPO-train${TRAIN_BATCH_SIZE}-alpha${ALPHA}-rollout${ROLLOUT_N}-lr${LR}-${MODEL_NAME}-${SUFFIX}"

# =============================================================================
# PRINT CONFIG
# =============================================================================

echo "============================================================================"
echo "SDPO Training (Recipe Mode)"
echo "============================================================================"
echo "Project:        ${PROJECT_NAME}"
echo "Experiment:     ${EXP_NAME}"
echo "Ray Address:    ${RAY_ADDRESS}"
echo "Working Dir:    ${WORKING_DIR}"
echo "Model:          ${MODEL_PATH}"
echo "Train Data:     ${TRAIN_FILE}"
echo "Test Data:      ${TEST_FILE}"
echo "Nodes:          ${NNODES}"
echo "GPUs/Node:      ${N_GPUS_PER_NODE}"
echo "Rollout N:      ${ROLLOUT_N}"
echo "Alpha (JSD):    ${ALPHA}"
echo "============================================================================"

# =============================================================================
# SUBMIT RAY JOB
# =============================================================================

ray job submit --address="${RAY_ADDRESS}" \
    --no-wait --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m main_sdpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.self_distillation.alpha=${ALPHA} \
    actor_rollout_ref.actor.self_distillation.distillation_topk=${DISTILLATION_TOPK} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    algorithm.rollout_correction.rollout_is=token \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}"

echo "============================================================================"
echo "Job submitted: ${EXP_NAME}"
echo "============================================================================"
