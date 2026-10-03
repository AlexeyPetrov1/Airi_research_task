#!/usr/bin/env bash
set -euo pipefail
# Reuse validated scene inputs, never rerun MolmoMotion or future reconstruction.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(dirname -- "$ROOT")"
PYTHON="${DAS_PYTHON:-$WORKSPACE/.venv-das/bin/python}"
DAS_ROOT="${DAS_ROOT:-$WORKSPACE/third_party/DiffusionAsShader-Wanfun}"
CHECKPOINT="${DAS_CHECKPOINT:-$WORKSPACE/models/Wan2.1-Fun-V1.1-1.3B-Control}"
SCENE="${DAS_SCENE:-$ROOT/runs/berkeley_ur5_molmomotion/cup}"
OUT="$SCENE/das_wanfun"
MODE="${1:-controlled}"
OUTPUT="$OUT/generated_molmomotion_seed42.mp4"
TRACKING=(--tracking_path "$OUT/control_molmomotion_720x480.mp4")
RECEIPT="$OUT/process_exit.json"
WAIT_ARGS=()
if [[ -n "${DAS_WAIT_PID:-}" ]]; then
    WAIT_ARGS=(--wait-pid "$DAS_WAIT_PID")
fi
if [[ "$MODE" == "no-control" ]]; then
    OUTPUT="$OUT/generated_no_control_seed42.mp4"
    TRACKING=()
    RECEIPT="$OUT/process_exit_no_control.json"
elif [[ "$MODE" != "controlled" ]]; then
    echo 'Usage: run_das_cup.sh [controlled|no-control]' >&2
    exit 2
fi
"$PYTHON" "$ROOT/scripts/das_supervise.py" --receipt "$RECEIPT" -- \
    "$PYTHON" -u "$ROOT/scripts/das_generate.py" \
    --image "$OUT/image_t0_720x480.png" \
    "${TRACKING[@]}" \
    --prompt 'Pick up the blue cup and put it into the brown cup.' \
    --checkpoint_path "$CHECKPOINT" --das_root "$DAS_ROOT" \
    --output_path "$OUTPUT" --seed 42 --num_inference_steps 25 "${WAIT_ARGS[@]}"
