#!/usr/bin/env bash
# Source after dry-run handling. Credentials stay outside code and launch arguments.
trex_configure_wandb() {
    local trace_enabled=0 status=0 key_file
    case $- in *x*) trace_enabled=1; set +x ;; esac
    export WANDB_MODE="${WANDB_MODE:-online}"
    export WANDB_PROJECT="${WANDB_PROJECT:-trex-posttrain}"
    if [[ "$WANDB_MODE" == online ]]; then
        if [[ -z "${WANDB_API_KEY:-}" ]]; then
            key_file="${WANDB_API_KEY_FILE:-$HOME/.config/trex/wandb_api_key}"
            if [[ -r "$key_file" ]]; then
                IFS= read -r WANDB_API_KEY < "$key_file" || true
            fi
        fi
        if [[ -z "${WANDB_API_KEY:-}" || "$WANDB_API_KEY" == *[[:space:]]* ]]; then
            echo 'W&B online requires WANDB_API_KEY or a one-line WANDB_API_KEY_FILE (default: ~/.config/trex/wandb_api_key).' >&2
            status=2
        else
            export WANDB_API_KEY
        fi
    elif [[ "$WANDB_MODE" != offline && "$WANDB_MODE" != disabled ]]; then
        echo 'WANDB_MODE must be online, offline, or disabled.' >&2
        status=2
    fi
    if (( trace_enabled )); then set -x; fi
    return "$status"
}
trex_configure_wandb
