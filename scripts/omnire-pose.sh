#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" != 4 ]]; then
    echo 'Usage: bash scripts/omnire-pose.sh RAW_ROOT PROCESSED_ROOT SCENE SMPL_NEUTRAL.pkl' >&2
    exit 2
fi
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw="$(realpath "$1")"
mkdir -p "$2"
processed="$(realpath "$2")"
scene="$3"
smpl="$(realpath "$4")"
source "$repo/.deps/pose-env/activate-research"
cd "$repo/.deps/omnire"
mkdir -p smpl_models
ln -sfn "$smpl" smpl_models/SMPL_NEUTRAL.pkl
# The one-worker converter passes scene IDs directly. Use its normal converter with padded IDs.
if [[ ! -f "$processed/$scene/instances/instances_info.json" ]]; then
    python -m datasets.preprocess --dataset pandaset --data_root "$raw" --target_dir "$processed" \
        --scene_ids "$scene" --workers 2 --process_keys images calib pose objects
fi
python -m datasets.tools.humanpose_process --dataset pandaset --data_root "$processed" --scene_ids "$scene" --fps 10
test -s "$processed/$scene/humanpose/smpl.pkl"
