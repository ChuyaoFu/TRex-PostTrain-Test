#!/usr/bin/env bash
# One foreground workflow: environments -> HF -> official FK -> JSON/PNG -> 8 H100.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-train}"
case "$MODE" in train|prepare|dry-run) shift "$(( $# > 0 ? 1 : 0 ))" ;; *) echo 'Usage: posttrain_h100.sh [train|prepare|dry-run] [trainer flags...]'; exit 2 ;; esac
TRAIN_VENV="${TRAIN_VENV:-$ROOT/.venvs/h100}"
DATA_VENV="${DATA_VENV:-$ROOT/.venvs/data}"
RAW_ROOT="${RAW_ROOT:-$ROOT/data/trex_gateway_tong_transfer_sf_norawtac_20260820}"
LEROBOT_ROOT="${LEROBOT_ROOT:-$ROOT/training_data/tong_transfer_eef62}"
DATA_FORMAT="${DATA_FORMAT:-json}"
IMAGE_ROOT="${IMAGE_ROOT:-$ROOT/training_data/tong_transfer_json}"
case "$DATA_FORMAT" in
    json) data_args=(--data_format json --data_path "$IMAGE_ROOT/task.json" --val_split_by_episode 0) ;;
    lerobot) data_args=(--data_format lerobot --lerobot_root "$LEROBOT_ROOT") ;;
    *) echo 'DATA_FORMAT must be json or lerobot'; exit 2 ;;
esac
WEIGHTS_ROOT="${WEIGHTS_ROOT:-$ROOT/weights}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
RUN_NAME="${RUN_NAME:-tong_transfer_h100_$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME=tong_transfer_h100
TRAIN_BSZ="${TRAIN_BSZ:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-1e-4}"
WARMUP_RATES="${WARMUP_RATES:-0}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0}"
N_EPOCHS="${N_EPOCHS:-100}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}"
SAVE_STEPS="${SAVE_STEPS:-0}"
VAL_FREQ="${VAL_FREQ:-500}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-30}"
MAX_CKPTS="${MAX_CKPTS:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MASTER_PORT="${MASTER_PORT:-29521}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WANDB_MODE="${WANDB_MODE:-online}" WANDB_PROJECT="${WANDB_PROJECT:-trex-posttrain}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$ROOT/.cache/torch_extensions_h100}"
export DS_BUILD_OPS=0
# Only inherited PPU environments need sanitizing; use upstream torch in a fresh env.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

cmd=("$TRAIN_VENV/bin/python" -m accelerate.commands.launch
    --config_file "$ROOT/config/sft_h100_8gpu.yaml"
    --num_processes 8 --num_machines 1 --machine_rank 0
    --main_process_ip 127.0.0.1 --main_process_port "$MASTER_PORT"
    "$ROOT/scripts/train.py"
    --model_path "$WEIGHTS_ROOT/Qwen3-VL-2B-Instruct"
    --processor_path "$WEIGHTS_ROOT/T-Rex_midtrain_epoch6/processor"
    --resume_checkpoint "$WEIGHTS_ROOT/T-Rex_midtrain_epoch6" --resume_source midtrain --strict_resume 1
    "${data_args[@]}"
    --n_epochs "$N_EPOCHS" --max_train_steps "$MAX_TRAIN_STEPS"
    --save_freq 50 --save_steps "$SAVE_STEPS" --max_ckpts "$MAX_CKPTS" --skip_checkpoint_save 0
    --action_dim 62 --action_chunk 16 --train_bsz_per_gpu "$TRAIN_BSZ"
    --learning_rate "$LR" --min_lr_ratio "$MIN_LR_RATIO" --warmup_rates "$WARMUP_RATES" --weight_decay 0
    --gradient_accumulation_steps "$GRAD_ACCUM" --max_grad_norm 1
    --output_dir "$OUTPUT_DIR" --log_dir "$LOG_DIR"
    --experiment_name "$EXPERIMENT_NAME" --run_name "$RUN_NAME"
    --use_robot_state 0 --use_tactile_vec 1 --use_tactile_deform 1 --use_tactile_vqvae 1
    --tactile_intermediate_size 1536 --training_stage 2 --tactile_loss_weight 1
    --cascaded_total_steps 10 --cascaded_split_step 6 --cascaded_tactile_dropout 0.1 --cascaded_loss_weight 1
    --use_flare 1 --n_flare_tokens_per_frame 4 --n_flare_steps 8 --flare_loss_weight 0.5
    --flare_frame_stride 4 --flare_layer_index -1 --image_size 384 288
    --val_ratio 0.05 --val_freq "$VAL_FREQ" --max_val_batches "$MAX_VAL_BATCHES"
    --val_uniform_sample 0 --eval_at_start 0 --eval_at_end 0 --eval_seed -1 --seed 42
    --num_workers "$NUM_WORKERS" --val_num_workers 2 "$@")

if [[ "$MODE" == dry-run ]]; then
    printf '8 GPUs; per GPU batch=%s; global batch=%s; epochs=%s; step cap=%s\n' "$TRAIN_BSZ" "$((8 * TRAIN_BSZ * GRAD_ACCUM))" "$N_EPOCHS" "$MAX_TRAIN_STEPS"
    printf '%q ' "${cmd[@]}"; printf '\n'
    exit 0
fi

if [[ "$MODE" == train ]]; then source "$ROOT/scripts/wandb_env.sh"; fi
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/$RUN_NAME.pipeline.log}"
mkdir -p "$(dirname "$LOG_FILE")"
# A subshell under a pipeline preserves errexit for every preparation stage.
(
    set -euo pipefail
    trap 'echo "Pipeline failed (line $LINENO). See: $LOG_FILE" >&2' ERR
    echo "Pipeline started: $(date -Iseconds); root=$ROOT; log=$LOG_FILE"
    if [[ -e "$OUTPUT_DIR/$EXPERIMENT_NAME/$RUN_NAME/metrics.jsonl" ]]; then
        echo 'RUN_NAME already has training metrics; select a new run name.' >&2; exit 2
    fi
    for binary in git curl python3 nvidia-smi; do command -v "$binary" >/dev/null || { echo "Missing $binary"; exit 2; }; done
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    if [[ "${SKIP_SETUP:-0}" != 1 ]]; then
        # Required even when uv is already installed on PATH.
        mkdir -p "$ROOT/.tools"
        # Reused inputs already occupy their allocation; export/output stages
        # independently check their remaining capacity before writing.
        SETUP_FREE_GIB=200
        if [[ -x "$TRAIN_VENV/bin/python" && -x "$DATA_VENV/bin/python" ]]; then SETUP_FREE_GIB=28; fi
        python3 - "$ROOT" "${MIN_FREE_GIB:-$SETUP_FREE_GIB}" <<'PYSPACE'
import shutil, sys
free = shutil.disk_usage(sys.argv[1]).free / 2**30
if free < float(sys.argv[2]):
    raise SystemExit(f'Only {free:.1f} GiB free; need >= {sys.argv[2]} GiB for setup/training. Choose another TREX_WORKDIR.')
PYSPACE
        if command -v uv >/dev/null; then UV="$(command -v uv)";
        elif [[ -x "$ROOT/.tools/uv" ]]; then UV="$ROOT/.tools/uv";
        else
            mkdir -p "$ROOT/.tools"
            curl -LsSf https://astral.sh/uv/0.8.22/install.sh -o "$ROOT/.tools/install-uv.sh"
            UV_INSTALL_DIR="$ROOT/.tools" UV_NO_MODIFY_PATH=1 sh "$ROOT/.tools/install-uv.sh"
            UV="$ROOT/.tools/uv"
        fi
        export UV_PYTHON_INSTALL_DIR="$ROOT/.tools/python"
        export UV_CACHE_DIR="$ROOT/.cache/uv"
        if [[ ! -x "$TRAIN_VENV/bin/python" ]]; then "$UV" venv --python 3.10 --seed "$TRAIN_VENV"; fi
        if [[ ! -x "$DATA_VENV/bin/python" ]]; then "$UV" venv --python 3.10 --seed "$DATA_VENV"; fi
        "$UV" pip install --python "$TRAIN_VENV/bin/python" setuptools==75.8.0 wheel==0.45.1 packaging==25.0 ninja==1.11.1.4
        "$UV" pip install --python "$TRAIN_VENV/bin/python" torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
        printf 'torch==2.6.0+cu124\ntorchvision==0.21.0+cu124\n' > "$ROOT/.tools/h100-torch-constraints.txt"
        "$UV" pip install --python "$TRAIN_VENV/bin/python" -c "$ROOT/.tools/h100-torch-constraints.txt" -r "$ROOT/requirements-h100.lock"
        "$UV" pip install --python "$TRAIN_VENV/bin/python" --no-deps --no-build-isolation deepspeed==0.15.4
        # Only LeRobot's dataset loader is used; its robotics/GUI policy extras conflict with the official recipe.
        "$UV" pip install --python "$TRAIN_VENV/bin/python" --no-deps lerobot==0.4.0
        TREX_DATA_VENV="$DATA_VENV" TREX_DATA_REQUIREMENTS="$ROOT/requirements-data-h100.lock" \
            TREX_PIP_INDEX_URL="${TREX_PIP_INDEX_URL:-https://pypi.org/simple}" bash "$ROOT/scripts/setup_data.sh"
        "$TRAIN_VENV/bin/python" -m pip freeze > "$LOG_DIR/$RUN_NAME.train-environment.txt"
        "$DATA_VENV/bin/python" -m pip freeze > "$LOG_DIR/$RUN_NAME.data-environment.txt"
    fi

    if [[ "$MODE" == train ]]; then
        "$TRAIN_VENV/bin/python" "$ROOT/scripts/wandb_preflight.py" --output "$LOG_DIR/$RUN_NAME.wandb_preflight.json"
    fi
    "$TRAIN_VENV/bin/python" - <<'PYGPU'
import torch, torchvision, transformers, accelerate, deepspeed
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from qwen_vla import Qwen3VLVLAModel
assert torch.__version__ == '2.6.0+cu124', torch.__version__
assert torch.cuda.device_count() == 8, 'Exactly eight visible NVIDIA GPUs are required.'
assert torch.cuda.is_bf16_supported(), 'BF16 GPU support required.'
for i in range(8):
    assert 'H100' in torch.cuda.get_device_name(i), torch.cuda.get_device_name(i)
    assert torch.cuda.get_device_capability(i) == (9, 0)
    x = torch.randn(128, 128, device=f'cuda:{i}', dtype=torch.bfloat16)
    assert torch.isfinite(x @ x).all()
print('8 H100 BF16 matmul / train imports PASS; CUDA runtime:', torch.version.cuda, flush=True)
PYGPU
    if [[ "${SKIP_DOWNLOAD:-0}" != 1 ]]; then
        "$TRAIN_VENV/bin/huggingface-cli" download miniFranka/trex_gateway_tong_transfer_sf_norawtac_20260820 \
            --repo-type dataset --revision 9e20b13d042c708e1546138adda25c13ee6aa4e7 --local-dir "$RAW_ROOT"
        "$TRAIN_VENV/bin/python" "$ROOT/scripts/download_h100_inputs.py" --raw-root "$RAW_ROOT" --weights-root "$WEIGHTS_ROOT"
    else
        "$TRAIN_VENV/bin/python" "$ROOT/scripts/download_h100_inputs.py" --offline --raw-root "$RAW_ROOT" --weights-root "$WEIGHTS_ROOT"
    fi
    if [[ -e "$LEROBOT_ROOT/CONVERSION_IN_PROGRESS" ]]; then
        echo "Incomplete conversion at $LEROBOT_ROOT. Move this directory aside before retrying." >&2; exit 2
    fi
    if [[ ! -e "$LEROBOT_ROOT" ]]; then
        "$DATA_VENV/bin/python" "$ROOT/utils/convert_joint_lerobot_to_trex.py" \
            --source_root "$RAW_ROOT" --output_root "$LEROBOT_ROOT" --video_mode symlink
    fi
    "$DATA_VENV/bin/python" - "$LEROBOT_ROOT" "$RAW_ROOT" <<'PYDATA'
import json, sys
from pathlib import Path
root, raw = map(Path, sys.argv[1:])
p = json.loads((root / 'meta/trex_conversion.json').read_text())
assert (p['episodes'], p['frames']) == (200, 208581), 'Expected the full task dataset.'
assert Path(p['source_root']).resolve() == raw.resolve(), 'Converted data points to a different raw dataset.'
assert p['torso'] == [0.9, 1.57, 0.1] and p['head'] == [0.28, 0.0, 0.0]
assert p['head_crop_box'] == [0, 300, 140, 540]
assert all(f.exists() for f in (root / 'videos').rglob('*.mp4')), 'Broken video links.'
PYDATA
    "$TRAIN_VENV/bin/python" "$ROOT/scripts/check_task_data.py" "$LEROBOT_ROOT"
    "$DATA_VENV/bin/python" "$ROOT/scripts/verify_joint_conversion.py" "$LEROBOT_ROOT"
    "$TRAIN_VENV/bin/python" "$ROOT/scripts/smoke_task_loader.py" "$LEROBOT_ROOT"
    if [[ "$DATA_FORMAT" == json ]]; then
        TREX_IMAGE_PYTHON="$TRAIN_VENV/bin/python" LEROBOT_ROOT="$LEROBOT_ROOT" IMAGE_ROOT="$IMAGE_ROOT" \
            PROCESSOR_PATH="$WEIGHTS_ROOT/T-Rex_midtrain_epoch6/processor" bash "$ROOT/scripts/prepare_json_images.sh"
    fi
    if [[ "$MODE" == prepare ]]; then echo 'Preparation PASS. Run the same command with train to launch.'; exit 0; fi
    "$TRAIN_VENV/bin/python" - "$OUTPUT_DIR" <<'PYOUTPUT'
import shutil, sys
free = shutil.disk_usage(sys.argv[1]).free / 2**30
if free < 28:
    raise SystemExit(f'Output disk only has {free:.1f} GiB free; require 28 GiB for checkpoint save/pruning peaks.')
PYOUTPUT
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    printf 'Launching: global batch=%s; epochs=%s; max_train_steps=%s; LR=%s; warmup=%s\n' "$((8 * TRAIN_BSZ * GRAD_ACCUM))" "$N_EPOCHS" "$MAX_TRAIN_STEPS" "$LR" "$WARMUP_RATES"
    printf '%q ' "${cmd[@]}"; printf '\n'
    "${cmd[@]}"
    "$TRAIN_VENV/bin/python" "$ROOT/scripts/summarize_training.py" "$OUTPUT_DIR/$EXPERIMENT_NAME/$RUN_NAME"
    FINAL_POLICY="$("$TRAIN_VENV/bin/python" - "$OUTPUT_DIR/$EXPERIMENT_NAME/$RUN_NAME" <<'PYFINAL'
from pathlib import Path
import sys
checkpoints = [p for p in Path(sys.argv[1]).glob('checkpoint-*') if (p / 'model.pt').is_file()]
if not checkpoints:
    raise SystemExit('Training completed but no policy checkpoint was saved.')
print(max(checkpoints, key=lambda p: (p / 'model.pt').stat().st_mtime))
PYFINAL
    )"
    "$TRAIN_VENV/bin/python" "$ROOT/scripts/verify_posttrain_checkpoint.py" \
        --release "$WEIGHTS_ROOT/T-Rex_midtrain_epoch6/model.pt" --checkpoint "$FINAL_POLICY"
    INFERENCE_PYTHON="$TRAIN_VENV/bin/python" bash "$ROOT/scripts/serve_policy.sh" "$FINAL_POLICY" --smoke_only 1
    printf '%s\n' "$FINAL_POLICY" > "$OUTPUT_DIR/$EXPERIMENT_NAME/$RUN_NAME/latest_policy.txt"
    echo "Verified policy for deployment: $FINAL_POLICY"
    echo "Training completed: $(date -Iseconds)"
) 2>&1 | tee -a "$LOG_FILE"
