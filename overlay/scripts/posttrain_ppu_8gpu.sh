#!/usr/bin/env bash
# Full task posttrain: eight PPU processes, release architecture and real data.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env_ppu.sh"
cd "$ROOT"

NUM_PROCESSES=8
NUM_MACHINES="${NUM_MACHINES:-1}"
MACHINE_RANK="${MACHINE_RANK:-0}"
MASTER_PORT="${MASTER_PORT:-29521}"
if [[ ! "$NUM_MACHINES" =~ ^[1-8]$ ]] || (( NUM_PROCESSES % NUM_MACHINES != 0 )); then
    echo 'NUM_MACHINES must be 1, 2, 4, or 8, with eight processes in total.' >&2
    exit 2
fi
if [[ ! "$MACHINE_RANK" =~ ^[0-7]$ ]] || (( MACHINE_RANK >= NUM_MACHINES )); then
    echo 'MACHINE_RANK must be in [0, NUM_MACHINES).' >&2
    exit 2
fi
PPUS_PER_NODE=$((NUM_PROCESSES / NUM_MACHINES))
if (( NUM_MACHINES > 1 )) && [[ -z "${MASTER_ADDR:-}" ]]; then
    echo 'Set MASTER_ADDR to rank 0 node IP for multi-node training.' >&2
    exit 2
fi
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
AUTO_DEVICES="$(seq -s, 0 "$((PPUS_PER_NODE - 1))")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$AUTO_DEVICES}"

RUN_NAME="${RUN_NAME:-tong_transfer_ppu_8gpu_$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME=tong_transfer_ppu_8gpu
MODEL_PATH="${MODEL_PATH:-$ROOT/weights/Qwen3-VL-2B-Instruct}"
MIDTRAIN_CHECKPOINT="${MIDTRAIN_CHECKPOINT:-$ROOT/weights/T-Rex_midtrain_epoch6}"
PROCESSOR_PATH="${PROCESSOR_PATH:-$ROOT/weights/T-Rex_midtrain_epoch6/processor}"
LEROBOT_ROOT="${LEROBOT_ROOT:-$ROOT/training_data/tong_transfer_eef62}"
DATA_FORMAT="${DATA_FORMAT:-json}"
# Per-node local cache avoids millions of files and capacity pressure on CPFS.
IMAGE_ROOT="${IMAGE_ROOT:-/tmp/trex_tong_transfer_json_20260820}"
case "$DATA_FORMAT" in
    json) data_args=(--data_format json --data_path "$IMAGE_ROOT/task.json" --val_split_by_episode 1) ;;
    lerobot) data_args=(--data_format lerobot --lerobot_root "$LEROBOT_ROOT") ;;
    *) echo 'DATA_FORMAT must be json or lerobot'; exit 2 ;;
esac
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
TRAIN_BSZ="${TRAIN_BSZ:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-3e-5}"
# Official posttrain length: 100 epochs, no maximum-update cap.
N_EPOCHS="${N_EPOCHS:-100}"
SAVE_STEPS="${SAVE_STEPS:-500}"
VAL_FREQ="${VAL_FREQ:-500}"
MAX_CKPTS="${MAX_CKPTS:-2}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
RUN_DIR="$OUTPUT_DIR/$EXPERIMENT_NAME/$RUN_NAME"

for file in "$MODEL_PATH/config.json" "$MODEL_PATH/model.safetensors" \
            "$MIDTRAIN_CHECKPOINT/model.pt" "$MIDTRAIN_CHECKPOINT/training_args.json" \
            "$PROCESSOR_PATH/tokenizer_config.json" "$LEROBOT_ROOT/meta/info.json"; do
    if [[ ! -f "$file" ]]; then echo "Missing training input: $file" >&2; exit 2; fi
done
if [[ -e "$LEROBOT_ROOT/CONVERSION_IN_PROGRESS" ]]; then
    echo 'Dataset conversion is still in progress.' >&2
    exit 2
fi

cmd=(python -m accelerate.commands.launch
    --config_file "$ROOT/config/sft_ppu_8gpu.yaml"
    --num_processes "$NUM_PROCESSES" --num_machines "$NUM_MACHINES"
    --machine_rank "$MACHINE_RANK" --main_process_ip "$MASTER_ADDR"
    --main_process_port "$MASTER_PORT"
    --deepspeed_multinode_launcher standard
    "$ROOT/scripts/train.py"
    --model_path "$MODEL_PATH" --processor_path "$PROCESSOR_PATH"
    --resume_checkpoint "$MIDTRAIN_CHECKPOINT" --resume_source midtrain --strict_resume 1
    "${data_args[@]}"
    --n_epochs "$N_EPOCHS" --max_train_steps 0
    --save_freq 50 --save_steps "$SAVE_STEPS" --max_ckpts "$MAX_CKPTS" --skip_checkpoint_save 0
    --action_dim 62 --action_chunk 16 --train_bsz_per_gpu "$TRAIN_BSZ"
    --learning_rate "$LR" --min_lr_ratio 0.1 --warmup_rates 0.05 --weight_decay 0
    --gradient_accumulation_steps "$GRAD_ACCUM" --max_grad_norm 1
    --output_dir "$OUTPUT_DIR" --log_dir "$LOG_DIR"
    --experiment_name "$EXPERIMENT_NAME" --run_name "$RUN_NAME"
    --use_robot_state 0 --use_tactile_vec 1 --use_tactile_deform 1 --use_tactile_vqvae 1
    --tactile_intermediate_size 1536 --training_stage 2 --tactile_loss_weight 1
    --cascaded_total_steps 10 --cascaded_split_step 6 --cascaded_tactile_dropout 0.1 --cascaded_loss_weight 1
    --use_flare 1 --n_flare_tokens_per_frame 4 --n_flare_steps 8 --flare_loss_weight 0.5
    --flare_frame_stride 4 --flare_layer_index -1 --image_size 384 288
    --val_ratio 0.05 --val_freq "$VAL_FREQ" --max_val_batches "$MAX_VAL_BATCHES"
    --val_uniform_sample 1 --eval_at_start 1 --eval_seed 1234 --seed 42
    --num_workers "$NUM_WORKERS" --val_num_workers 0
    "$@")
print_config() {
    printf 'PPU processes=%s nodes=%s local_devices=%s global_batch=%s\n' \
        "$NUM_PROCESSES" "$NUM_MACHINES" "$CUDA_VISIBLE_DEVICES" "$((NUM_PROCESSES * TRAIN_BSZ * GRAD_ACCUM))"
    printf 'Epochs=%s; optimizer updates uncapped; LR=%s; run_dir=%s\n' "$N_EPOCHS" "$LR" "$RUN_DIR"
}
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    print_config
    printf '%q ' "${cmd[@]}"; printf '\n'
    exit 0
fi

if (( NUM_MACHINES > 1 )); then
    LOG_FILE="${LOG_FILE:-$LOG_DIR/$RUN_NAME.rank$MACHINE_RANK.log}"
else
    LOG_FILE="${LOG_FILE:-$LOG_DIR/$RUN_NAME.log}"
fi
mkdir -p "$(dirname "$LOG_FILE")"
run_training() {
    print_config
    printf 'Console log: %s\n' "$LOG_FILE"
    if [[ "$DATA_FORMAT" == json ]]; then
        TREX_IMAGE_PYTHON="$(command -v python)" LEROBOT_ROOT="$LEROBOT_ROOT" IMAGE_ROOT="$IMAGE_ROOT" \
            PROCESSOR_PATH="$PROCESSOR_PATH" bash "$ROOT/scripts/prepare_json_images.sh" || return $?
    fi
    if [[ "${PREPARE_ONLY:-0}" == 1 ]]; then
        echo "Data preparation PASS. Image cache: $IMAGE_ROOT"
        return 0
    fi
    python - "$PPUS_PER_NODE" <<'PYDEV'
import sys, torch
expected = int(sys.argv[1]); available = torch.cuda.device_count()
if available != expected:
    raise SystemExit(f'Expected {expected} visible PPU devices on this node, found {available}. '
                     'Check the instance allocation and CUDA_VISIBLE_DEVICES.')
print('Visible PPU devices:', [torch.cuda.get_device_name(i) for i in range(available)], flush=True)
PYDEV
    local device_check_status=$?
    if (( device_check_status != 0 )); then return "$device_check_status"; fi
    if [[ -e "$RUN_DIR/metrics.jsonl" ]]; then
        echo "Run already has metrics: $RUN_DIR. Choose a new RUN_NAME." >&2
        return 2
    fi
    "${cmd[@]}"
}
# The entire training pipeline stays in the foreground. Record stdout/stderr
# while preserving the launcher's status instead of tee's successful status.
set +e
run_training 2>&1 | tee -a "$LOG_FILE"
pipeline_status=("${PIPESTATUS[@]}")
set -e
if (( pipeline_status[0] != 0 )); then
    exit "${pipeline_status[0]}"
fi
exit "${pipeline_status[1]}"
