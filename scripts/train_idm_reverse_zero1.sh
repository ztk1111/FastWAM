#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${1:?Usage: bash scripts/train_idm_reverse_zero1.sh <nproc_per_node> [hydra_overrides...]}"
shift

exec bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
  task=libero_idm_2cam224_1e-4 \
  model.loss.enable_reverse_action_loss=true \
  "$@"
