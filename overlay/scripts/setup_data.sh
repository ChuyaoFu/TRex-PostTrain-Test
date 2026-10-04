#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${TREX_DATA_VENV:-$(dirname "$ROOT")/venvs/trex-data}"
BASE_PYTHON="${TREX_DATA_BASE_PYTHON:-python3}"
if [ ! -x "$VENV/bin/python" ]; then "$BASE_PYTHON" -m venv "$VENV"; fi
"$VENV/bin/python" -m pip install -r "${TREX_DATA_REQUIREMENTS:-$ROOT/requirements-data.txt}" \
    --index-url "${TREX_PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
URDF_ROOT="$ROOT/dataset_quickstart/third_party/dexmate-urdf"
if [ ! -f "$URDF_ROOT/src/dexmate_urdf/robots/humanoid/vega_1/vega_1.urdf" ]; then
    cp -r "$URDF_ROOT/robots/." "$URDF_ROOT/src/dexmate_urdf/robots/"
fi
if [ ! -f "$URDF_ROOT/src/dexmate_urdf/robots/content.py" ]; then
    PATH="$VENV/bin:$PATH" "$VENV/bin/python" "$URDF_ROOT/scripts/workflows/generate_content.py"
fi
"$VENV/bin/python" -m pip install --no-deps --no-build-isolation -e "$URDF_ROOT"
PYTHONPATH="$ROOT/dataset_quickstart/src:$ROOT" "$VENV/bin/python" - <<'PY'
from trex_dataset_quickstart.robot import build_reduced_bimanual_robot, DEFAULT_TORSO, DEFAULT_HEAD
robot, _, _ = build_reduced_bimanual_robot({'torso': DEFAULT_TORSO, 'head': DEFAULT_HEAD})
assert robot.nq == 14
print('Official Vega-1 FK ready; CPU preprocessing only.')
PY
echo "Ready: $VENV/bin/python $ROOT/utils/convert_joint_lerobot_to_trex.py --help"
