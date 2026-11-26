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
if [[ ! "$scene" =~ ^[0-9]{3}$ || ! -f "$smpl" ]]; then
    echo 'Supply a three-digit scene ID and the licensed SMPL Neutral model.' >&2
    exit 2
fi
source "$repo/.deps/pose-env/activate-research"
cd "$repo/.deps/omnire"
if ! git apply --reverse --check "$repo/omnire-tracker-memory.patch" 2>/dev/null; then
    git apply "$repo/omnire-tracker-memory.patch"
fi
mkdir -p smpl_models
ln -sfn "$smpl" smpl_models/SMPL_NEUTRAL.pkl
hmr_cache="$(python -c 'from third_party.Humans4D.hmr2.configs import CACHE_DIR_4DHUMANS; print(CACHE_DIR_4DHUMANS)')"
mkdir -p "$hmr_cache/data/smpl"
ln -sfn "$smpl" "$hmr_cache/data/smpl/SMPL_NEUTRAL.pkl"
stage="$(mktemp -d "${TMPDIR:-/tmp}/pandaset-pose-$scene.XXXXXX")"
trap 'rm -rf -- "$stage"' EXIT
# The one-worker converter passes scene IDs directly. Use its normal converter with padded IDs.
python - "$raw" "$stage" "$scene" <<'PY'
import runpy
import sys

from pandaset.annotations import Cuboids

# The pinned devkit scans for .pkl, while the staged PandaSet stores .pkl.gz.
Cuboids._data_file_extension = property(lambda self: "pkl.gz")
raw, stage, scene = sys.argv[1:]
sys.argv[1:] = [
    "--dataset", "pandaset", "--data_root", raw, "--target_dir", stage,
    "--scene_ids", scene, "--workers", "2", "--process_keys", "images", "calib", "pose", "objects",
]
runpy.run_module("datasets.preprocess", run_name="__main__")
PY
# The pinned 4D-Humans loader reads its released Lightning checkpoints with Python config objects.
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python - "$stage" "$scene" <<'PY'
import builtins
import inspect
import runpy
import sys

import numpy as np

# Licensed SMPL pickles can load legacy Chumpy values under the pose runtime.
for name in ("bool", "int", "float", "complex", "object", "str"):
    vars(np).setdefault(name, getattr(builtins, name))
vars(np).setdefault("unicode", str)
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

from research.pose_memory import limit_vitdet_workspace

limit_vitdet_workspace()
stage, scene = sys.argv[1:]
sys.argv[1:] = ["--dataset", "pandaset", "--data_root", stage, "--scene_ids", scene]
runpy.run_module("datasets.tools.humanpose_process", run_name="__main__")
PY
test -s "$stage/$scene/humanpose/smpl.pkl"
mkdir -p "$processed/$scene/instances" "$processed/$scene/humanpose"
for file in instances/instances_info.json humanpose/smpl.pkl; do
    target="$processed/$scene/$file"
    cp "$stage/$scene/$file" "$target.part"
    mv -f "$target.part" "$target"
done
