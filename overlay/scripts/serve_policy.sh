#!/usr/bin/env bash
# Offline reconstruction from the delivered policy; no base-model or VQ downloads.
set -euo pipefail
TREX_INFERENCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_PATH="${1:?Usage: serve_policy.sh /absolute/checkpoint [--smoke_only 1]}"
shift
cd "$TREX_INFERENCE_ROOT"
export PYTHONPATH="$TREX_INFERENCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
for required in model.pt config.json training_args.json stats_data.json processor/tokenizer_config.json; do
    [[ -f "$POLICY_PATH/$required" ]] || { echo "Missing deployment input: $POLICY_PATH/$required" >&2; exit 2; }
done
exec "${INFERENCE_PYTHON:-$TREX_INFERENCE_ROOT/.venvs/h100/bin/python}" "$TREX_INFERENCE_ROOT/scripts/test.py" \
    --checkpoint_path "$POLICY_PATH" --strict_checkpoint 1 \
    --action_dim 62 --action_chunk 16 --use_robot_state 0 \
    --use_tactile_vec 1 --use_tactile_deform 1 --image_size 384 288 \
    --cuda 0 --port "${PORT:-5678}" "$@"
