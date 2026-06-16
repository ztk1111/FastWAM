#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${1:?Usage: bash scripts/train_subgoal_idm_zero1.sh <nproc_per_node> [hydra_overrides...]}"
shift

TASK="${TASK:-libero_idm_2cam224_1e-4}"
BIDIRECTIONAL_ENABLED="${BIDIRECTIONAL_ENABLED:-false}"
LAMBDA_BACKWARD="${LAMBDA_BACKWARD:-1.0}"
DIRECTION_TOKEN="${DIRECTION_TOKEN:-false}"
DELTA_ACTION_DIM_MASK="${DELTA_ACTION_DIM_MASK:-null}"
EVAL_EVERY="${EVAL_EVERY:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MOT_CHECKPOINT_MIXED_ATTN="${MOT_CHECKPOINT_MIXED_ATTN:-true}"
WANDB_NAME="${WANDB_NAME:-${TASK}_idm}"

EXTRA_ARGS=("$@")
BASE_ARGS=(
  "task=${TASK}"
  "batch_size=${BATCH_SIZE}"
  "model.mot_checkpoint_mixed_attn=${MOT_CHECKPOINT_MIXED_ATTN}"
  "model.bidirectional.enabled=${BIDIRECTIONAL_ENABLED}"
  "model.bidirectional.lambda_backward=${LAMBDA_BACKWARD}"
  "model.bidirectional.direction_token=${DIRECTION_TOKEN}"
  "model.bidirectional.delta_action_dim_mask=${DELTA_ACTION_DIM_MASK}"
  "model.goal_token.enabled=false"
  "wandb.name=${WANDB_NAME}"
)

echo "[bidirectional_idm] task=${TASK} enabled=${BIDIRECTIONAL_ENABLED} lambda_backward=${LAMBDA_BACKWARD} batch_size=${BATCH_SIZE} mot_ckpt=${MOT_CHECKPOINT_MIXED_ATTN} delta_mask=${DELTA_ACTION_DIM_MASK}"

exec bash scripts/train_zero1.sh "${NPROC_PER_NODE}" "${BASE_ARGS[@]}" "${EXTRA_ARGS[@]}"
