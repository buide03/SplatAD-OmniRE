#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo="$(pwd)"
python=python3
deps="$repo/.deps"
profile="${1:-train}"
repair="${2:-}"
mkdir -p "$deps"

pin() {
    "$python" -c 'import json,sys; print(json.load(open("versions.json"))[sys.argv[1]][sys.argv[2]])' "$1" "$2"
}

checkout() {
    local project="$1" path="$2" sha
    sha="$(pin "$project" commit)"
    if [[ ! -d "$path/.git" ]]; then
        git init "$path"
        git -C "$path" remote add origin "$(pin "$project" url)"
    fi
    if ! git -C "$path" cat-file -e "$sha^{commit}" 2>/dev/null; then
        git -C "$path" fetch --depth=1 origin "$sha"
    fi
    if [[ "$(git -C "$path" rev-parse HEAD 2>/dev/null || true)" != "$sha" ]]; then
        git -C "$path" checkout --detach "$sha"
    fi
}

if [[ "$profile" == "source" ]]; then
    checkout neurad_studio "$deps/neurad-studio"
    if ! git -C "$deps/neurad-studio" apply --reverse --check "$repo/splatad.patch" 2>/dev/null; then
        git -C "$deps/neurad-studio" apply "$repo/splatad.patch"
    fi
    exit 0
fi

case "$profile" in
    train) python_version=3.12; venv="$deps/train-env" ;;
    pose) python_version=3.10; venv="$deps/pose-env" ;;
    convert) python_version=3.10; venv="$deps/smpl-convert" ;;
    *) echo "Usage: bash scripts/bootstrap-colab.sh [source|train|pose|convert] [--repair]" >&2; exit 2 ;;
esac
if [[ -n "$repair" && "$repair" != "--repair" ]]; then
    echo "Unknown bootstrap option: $repair" >&2
    exit 2
fi

if ! command -v uv >/dev/null; then
    "$python" -m pip install uv
fi

# Match the PyTorch wheel to the CUDA headers and nvcc used by the JIT kernels.
cuda_home=""
cuda_tag=""
if [[ "$profile" != "convert" ]]; then
    cuda_home="${CUDA_HOME:-/usr/local/cuda}"
    if [[ -z "${CUDA_HOME:-}" ]]; then
        for candidate in /usr/local/cuda-12.8 /usr/local/cuda-12.6; do
            if [[ -x "$candidate/bin/nvcc" ]]; then
                cuda_home="$candidate"
                break
            fi
        done
    fi
    if [[ ! -x "$cuda_home/bin/nvcc" ]]; then
        echo "Set CUDA_HOME to a CUDA 12.8 or 12.6 toolkit that provides nvcc." >&2
        exit 1
    fi
    case "$("$cuda_home/bin/nvcc" --version)" in
        *"release 12.8,"*) cuda_tag=cu128 ;;
        *"release 12.6,"*) cuda_tag=cu126 ;;
        *) echo "This profile supports CUDA 12.6 and 12.8. Set CUDA_HOME." >&2; exit 1 ;;
    esac
    export CUDA_HOME="$cuda_home" PATH="$cuda_home/bin:$PATH"
fi
export MAX_JOBS="${MAX_JOBS:-2}"

install_torch() {
    uv pip install --python "$1" "torch==2.7.1+$cuda_tag" "torchvision==0.22.1+$cuda_tag" \
        --index-url "https://download.pytorch.org/whl/$cuda_tag"
}

activation() {
    local venv="$1"
    {
        printf 'source %q\n' "$venv/bin/activate"
        if [[ -n "$cuda_home" ]]; then
            printf 'export CUDA_HOME=%q\n' "$cuda_home"
            printf 'export PATH=%q:"$PATH"\n' "$cuda_home/bin"
        fi
        printf 'export TORCH_EXTENSIONS_DIR=%q\n' "$venv/torch_extensions"
        printf 'export MAX_JOBS=%q\n' "$MAX_JOBS"
        printf 'unset PYTHONHOME\n'
        printf 'export PYTHONNOUSERSITE=1 PYTHONPATH=%q\n' "$repo"
    } > "$venv/activate-research"
}

py="$venv/bin/python"
health_probe() (
    source "$venv/activate-research" || exit 1
    case "$profile" in
        train) "$py" -c 'import torch, torchvision; from gsplat import rasterization, lidar_rasterization; import research.train; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())' ;;
        pose) "$py" -c 'import torch, torchvision, cv2; from pytorch3d import _C; import phalp; assert torch.cuda.is_available()' ;;
        convert) "$py" -c 'import numpy, scipy; from importlib.util import find_spec; assert find_spec("chumpy") is not None' ;;
    esac
)

signature="$(
    printf '%s\n' "$profile" "$cuda_home" "$cuda_tag"
    sha256sum scripts/bootstrap-colab.sh versions.json splatad.patch omnire-pose.patch
)"
ready="$venv/.research-ready"
if [[ -z "$repair" && -f "$ready" && "$(cat "$ready")" == "$signature" ]] && health_probe; then
    echo "Reusing the validated $profile environment."
    exit 0
fi

# Only this local, repository-owned environment is recreated. uv keeps its download cache.
uv venv --clear --python "$python_version" "$venv"
case "$profile" in
    train)
        bash "$repo/scripts/bootstrap-colab.sh" source
        install_torch "$py"
        uv pip install --python "$py" 'setuptools<70' wheel ninja pip
        printf 'torch==2.7.1+%s\ntorchvision==0.22.1+%s\n' "$cuda_tag" "$cuda_tag" > "$deps/train-overrides.txt"
        cat "$deps/neurad-studio/scripts/requirements-common.txt" >> "$deps/train-overrides.txt"
        printf '\ntorchmetrics==1.8.2\n' >> "$deps/train-overrides.txt"
        BUILD_NO_CUDA=1 uv pip install --python "$py" --no-build-isolation \
            --overrides "$deps/train-overrides.txt" -e "$deps/neurad-studio" joblib pyyaml
        ;;
    pose)
        checkout omnire "$deps/omnire"
        if ! git -C "$deps/omnire" apply --reverse --check "$repo/omnire-pose.patch" 2>/dev/null; then
            git -C "$deps/omnire" apply "$repo/omnire-pose.patch"
        fi
        git -C "$deps/omnire" submodule update --init --recursive third_party/Humans4D
        [[ "$(git -C "$deps/omnire/third_party/Humans4D" rev-parse HEAD)" == "$(pin humans4d commit)" ]]
        install_torch "$py"
        uv pip install --python "$py" 'setuptools<70' wheel ninja pip 'numpy<2'
        printf 'numpy<2\ntorch==2.7.1+%s\ntorchvision==0.22.1+%s\n' "$cuda_tag" "$cuda_tag" > "$deps/pose-constraints.txt"
        uv pip install --python "$py" --no-build-isolation -c "$deps/pose-constraints.txt" \
            "git+$(pin phalp url)@$(pin phalp commit)" \
            "git+$(pin pytorch3d url)@$(pin pytorch3d commit)" \
            "git+https://github.com/ziyc/pandaset-devkit.git@0c10f2b03bc00b9c77950f86e478e271edba2c00#subdirectory=python" \
            "$deps/omnire/third_party/Humans4D[all]" chumpy scipy imageio-ffmpeg omegaconf joblib
        ;;
    convert)
        uv pip install --python "$py" 'setuptools<70' wheel pip
        uv pip install --python "$py" --no-build-isolation "numpy==1.26.4" "scipy==1.12.0" "chumpy==0.70"
        ;;
esac
activation "$venv"
health_probe
printf '%s\n' "$signature" > "$ready.partial"
mv -f "$ready.partial" "$ready"
