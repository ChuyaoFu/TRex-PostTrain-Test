#!/usr/bin/env bash
# Reusable foreground stage for PPU and H100; the caller provides its train Python.
set -euo pipefail
TREX_IMAGE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TREX_IMAGE_PYTHON="${TREX_IMAGE_PYTHON:-python}"
LEROBOT_ROOT="${LEROBOT_ROOT:-$TREX_IMAGE_REPO/training_data/tong_transfer_eef62}"
IMAGE_ROOT="${IMAGE_ROOT:-$TREX_IMAGE_REPO/training_data/tong_transfer_json}"
PROCESSOR_PATH="${PROCESSOR_PATH:-$TREX_IMAGE_REPO/weights/T-Rex_midtrain_epoch6/processor}"
"$TREX_IMAGE_PYTHON" "$TREX_IMAGE_REPO/utils/export_lerobot_to_json_images.py" \
    --source_root "$LEROBOT_ROOT" --output_root "$IMAGE_ROOT" \
    --workers "${IMAGE_WORKERS:-4}" --min_free_gib "${MIN_IMAGE_FREE_GIB:-140}"
"$TREX_IMAGE_PYTHON" "$TREX_IMAGE_REPO/scripts/verify_json_image_data.py" \
    --lerobot_root "$LEROBOT_ROOT" --image_root "$IMAGE_ROOT" --processor_path "$PROCESSOR_PATH"
