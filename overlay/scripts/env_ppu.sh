#!/usr/bin/env bash
# Source this file in bash. PPU torch is supplied by the host SDK, not pip.
_TREX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_TREX_NOUNSET=0
case $- in *u*) _TREX_NOUNSET=1; set +u ;; esac
source "${TREX_PPU_SDK:-/usr/local/PPU_SDK}/envsetup.sh" cuda
if [ "$_TREX_NOUNSET" = 1 ]; then set -u; fi
export TREX_PPU_VENV="${TREX_PPU_VENV:-$(dirname "$_TREX_ROOT")/venvs/trex-ppu}"
source "$TREX_PPU_VENV/bin/activate"
export PYTHONPATH="$_TREX_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-online}"
# Isolate PPU JIT extensions from NVIDIA builds.
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$TREX_PPU_VENV/torch_extensions}"
unset _TREX_ROOT _TREX_NOUNSET
