#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${1:?Usage: bash scripts/train_subgoal_idm_zero2.sh <nproc_per_node> [hydra_overrides...]}"
shift

TASK="${TASK:-libero_idm_2cam224_1e-4}"
NUM_SUBGOAL_LATENTS="${NUM_SUBGOAL_LATENTS:-5}"
SUBGOAL_STRATEGY="${SUBGOAL_STRATEGY:-uniform_endpoints}"
INCLUDE_FIRST_LATENT="${INCLUDE_FIRST_LATENT:-true}"
SUBGOAL_TOKEN_DROPOUT="${SUBGOAL_TOKEN_DROPOUT:-0.05}"
GOAL_TOKEN_DROPOUT="${GOAL_TOKEN_DROPOUT:-0.10}"
GOAL_TOKEN_ENABLED="${GOAL_TOKEN_ENABLED:-false}"
GOAL_TOKEN_CHECKPOINT="${GOAL_TOKEN_CHECKPOINT:-}"
WANDB_NAME="${WANDB_NAME:-${TASK}_subgoal_k${NUM_SUBGOAL_LATENTS}}"

EXTRA_ARGS=("$@")
BASE_ARGS=(
  "task=${TASK}"
  "model.subgoal_latent.enabled=true"
  "model.subgoal_latent.num_subgoal_latents=${NUM_SUBGOAL_LATENTS}"
  "model.subgoal_latent.selection_strategy=${SUBGOAL_STRATEGY}"
  "model.subgoal_latent.include_first_latent=${INCLUDE_FIRST_LATENT}"
  "model.subgoal_latent.subgoal_token_dropout=${SUBGOAL_TOKEN_DROPOUT}"
  "model.subgoal_latent.goal_token_dropout=${GOAL_TOKEN_DROPOUT}"
  "wandb.name=${WANDB_NAME}"
)

if [[ "${GOAL_TOKEN_ENABLED}" == "true" ]]; then
  if [[ -z "${GOAL_TOKEN_CHECKPOINT}" ]]; then
    echo "Error: GOAL_TOKEN_ENABLED=true requires GOAL_TOKEN_CHECKPOINT=/path/to/best.pt" >&2
    exit 1
  fi
  BASE_ARGS+=(
    "model.goal_token.enabled=true"
    "model.goal_token.checkpoint_path=${GOAL_TOKEN_CHECKPOINT}"
  )
fi

echo "[subgoal_idm] task=${TASK} k=${NUM_SUBGOAL_LATENTS} strategy=${SUBGOAL_STRATEGY} include_first=${INCLUDE_FIRST_LATENT} goal_token=${GOAL_TOKEN_ENABLED}"

exec bash scripts/train_zero2.sh "${NPROC_PER_NODE}" "${BASE_ARGS[@]}" "${EXTRA_ARGS[@]}"
