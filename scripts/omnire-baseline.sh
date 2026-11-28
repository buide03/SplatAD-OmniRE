#!/usr/bin/env bash
# Run research.omnire_baseline in the official OmniRe environment, from the pinned DriveStudio checkout.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo/.deps/omnire-env/activate-research"
cd "$repo/.deps/omnire"
export PYTHONPATH="$repo/.deps/omnire:$repo"
# Pruning copies every Adam moment; fixed-size CUDA segments left 4.4 GB reserved but unusable at 4.5M Gaussians.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec python -m research.omnire_baseline "$@"
