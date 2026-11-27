#!/usr/bin/env bash
# Run research.omnire_baseline in the official OmniRe environment, from the pinned DriveStudio checkout.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo/.deps/omnire-env/activate-research"
cd "$repo/.deps/omnire"
export PYTHONPATH="$repo/.deps/omnire:$repo"
exec python -m research.omnire_baseline "$@"
