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
N_GPUS_PER_NODE=8

# Model and Data paths
MODEL_PATH="/home/work/cxr/models/Qwen3-8B"
DATA_PATH="/home/work/cxr/Self-KDRL/datasets/sciknoweval/chemistry"
TRAIN_FILE="${DATA_PATH}/train.parquet"
TEST_FILE="${DATA_PATH}/test.parquet"
# Checkpoint directory
CHECKPOINTS_DIR="/home/work/cxr/Self-KDRL/checkpoints"

# Hyperparameters
TRAIN_BATCH_SIZE=32
ROLLOUT_N=8
LR=1e-5
ALPHA=0.5
DISTILLATION_TOPK=100
DONTS_REPROMPT_ON_SELF_SUCCESS=true
INCLUDE_ENVIRONMENT_FEEDBACK=false  # Include environment feedback in reprompt

# Performance settings
SP_SIZE=1
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=8192

# Ray settings
RAY_ADDRESS="http://localhost:8265"

# Wandb settings
# Use native wandb environment variables for proxy:
LOGGER_BACKENDS="['console, wandb']"
WANDB_PROXY='http://10.251.113.32:3128'

# Checkpoint settings
SAVE_FREQ=-1  # Save checkpoint every N iterations (-1 to disable)
MAX_ACTOR_CKPT_TO_KEEP=1  # Maximum number of actor checkpoints to keep
TOTAL_EPOCHS=30  # Total training epochs
TEST_FREQ=5  # Test/eval frequency

# Experiment name settings for wandb
MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
PROJECT_NAME="SDPO"
EXP_NAME="SDPO-train${TRAIN_BATCH_SIZE}-alpha${ALPHA}-rollout${ROLLOUT_N}-lr${LR}-dross${DONTS_REPROMPT_ON_SELF_SUCCESS}-${MODEL_NAME}-${SUFFIX}"

# =============================================================================
# INTERNAL CONFIGURATION - Auto-generated paths (usually no need to edit)
# =============================================================================

# Get script directory (works from any location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKING_DIR="${SCRIPT_DIR}"

# Get project root directory (two levels up from script)
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Get verl config path (relative to project root)
VERL_CONFIG_PATH="${PROJECT_ROOT}/verl/trainer/config"

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
echo "Checkpoint Dir: ${WORKING_DIR}/checkpoints/${PROJECT_NAME}/${EXP_NAME}"
echo "Model:          ${MODEL_PATH}"
echo "Train Data:     ${TRAIN_FILE}"
echo "Test Data:      ${TEST_FILE}"
echo "Nodes:          ${NNODES}"
echo "GPUs/Node:      ${N_GPUS_PER_NODE}"
echo "Rollout N:      ${ROLLOUT_N}"
echo "Alpha (JSD):    ${ALPHA}"
echo "Total Epochs:   ${TOTAL_EPOCHS}"
echo "Save Freq:      ${SAVE_FREQ}"
echo "Test Freq:      ${TEST_FREQ}"
echo "============================================================================"

# =============================================================================
# SUBMIT RAY JOB
# =============================================================================

ray job submit --address="${RAY_ADDRESS}" \
    --no-wait --runtime-env="${WORKING_DIR}/config/runtime_env.yaml" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m main_sdpo \
    hydra.searchpath="['file://${VERL_CONFIG_PATH}']" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.self_distillation.alpha=${ALPHA} \
    actor_rollout_ref.actor.self_distillation.distillation_topk=${DISTILLATION_TOPK} \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=${INCLUDE_ENVIRONMENT_FEEDBACK} \
    algorithm.rollout_correction.rollout_is=token \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.logger=${LOGGER_BACKENDS} \
    +trainer.wandb_proxy=${WANDB_PROXY} \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.default_local_dir="${CHECKPOINTS_DIR}/${PROJECT_NAME}/${EXP_NAME}"

echo "============================================================================"
echo "Job submitted: ${EXP_NAME}"
echo "============================================================================"
