#!/usr/bin/env bash
# Transfer this small directory; all upstream code, data and weights download here.
set -euo pipefail
BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REV=f88e10c61da123c68bf0927cf4860bc97a0381f3
WORK="${TREX_WORKDIR:-$BUNDLE/T-Rex_h100}"
if [[ ! -e "$WORK" ]]; then
    mkdir -p "$WORK"
    printf '%s\n' "$REV" > "$WORK/.trex-handoff-initializing"
    git -C "$WORK" init -q
    git -C "$WORK" remote add origin https://github.com/ZhuoyangLiu2005/T-Rex.git
fi
if [[ -f "$WORK/.trex-handoff-initializing" ]] && [[ "$(cat "$WORK/.trex-handoff-initializing")" == "$REV" ]]; then
    if [[ "$(git -C "$WORK" remote get-url origin)" != https://github.com/ZhuoyangLiu2005/T-Rex.git ]]; then
        echo 'Unexpected source repository in partial handoff checkout.' >&2; exit 2
    fi
    git -C "$WORK" fetch --depth 1 origin "$REV"
    git -C "$WORK" checkout --detach -q FETCH_HEAD
    printf '%s\n' "$REV" > "$WORK/.trex-handoff-base"
    rm "$WORK/.trex-handoff-initializing"
fi
if [[ ! -f "$WORK/.trex-handoff-base" ]] || [[ "$(cat "$WORK/.trex-handoff-base")" != "$REV" ]]; then
    echo "Use a new TREX_WORKDIR; refusing to overwrite an unrelated checkout: $WORK" >&2
    exit 2
fi
if [[ "$(git -C "$WORK" rev-parse HEAD)" != "$REV" ]]; then
    echo 'Handoff checkout HEAD changed. Use a new TREX_WORKDIR.' >&2
    exit 2
fi
cp -a "$BUNDLE/overlay/." "$WORK/"
exec bash "$WORK/scripts/posttrain_h100.sh" "$@"
