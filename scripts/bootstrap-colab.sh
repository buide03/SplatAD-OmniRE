#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo="$(pwd)"
python=python3
deps="$repo/.deps"
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

if [[ "${1:-train}" == "source" ]]; then
    checkout neurad_studio "$deps/neurad-studio"
    if ! git -C "$deps/neurad-studio" apply --reverse --check "$repo/splatad.patch" 2>/dev/null; then
        git -C "$deps/neurad-studio" apply "$repo/splatad.patch"
    fi
    exit 0
fi

if ! command -v uv >/dev/null; then
    "$python" -m pip install uv
fi

# Match the PyTorch wheel to the CUDA headers and nvcc used by the JIT kernels.
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
if [[ -z "${CUDA_HOME:-}" && -x /usr/local/cuda-12.8/bin/nvcc ]]; then
    cuda_home=/usr/local/cuda-12.8
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
export MAX_JOBS="${MAX_JOBS:-2}"

install_torch() {
    uv pip install --python "$1" "torch==2.7.1+$cuda_tag" "torchvision==0.22.1+$cuda_tag" \
        --index-url "https://download.pytorch.org/whl/$cuda_tag"
}

activation() {
    local venv="$1"
    {
        printf 'source %q\n' "$venv/bin/activate"
        printf 'export CUDA_HOME=%q\n' "$cuda_home"
        printf 'export PATH=%q:"$PATH"\n' "$cuda_home/bin"
        printf 'export TORCH_EXTENSIONS_DIR=%q\n' "$venv/torch_extensions"
        printf 'export MAX_JOBS=%q\n' "$MAX_JOBS"
        printf 'export PYTHONPATH=%q${PYTHONPATH:+:"$PYTHONPATH"}\n' "$repo"
    } > "$venv/activate-research"
}

case "${1:-train}" in
    train)
        bash "$0" source
        venv="$deps/train-env"
        uv venv --allow-existing --python 3.12 "$venv"
        py="$venv/bin/python"
        install_torch "$py"
        uv pip install --python "$py" 'setuptools<70' wheel ninja pip
        printf 'torch==2.7.1+%s\ntorchvision==0.22.1+%s\n' "$cuda_tag" "$cuda_tag" > "$deps/train-overrides.txt"
        cat "$deps/neurad-studio/scripts/requirements-common.txt" >> "$deps/train-overrides.txt"
        printf '\ntorchmetrics==1.8.2\n' >> "$deps/train-overrides.txt"
        BUILD_NO_CUDA=1 uv pip install --python "$py" --no-build-isolation \
            --overrides "$deps/train-overrides.txt" -e "$deps/neurad-studio" joblib
        activation "$venv"
        "$py" -c 'import torch; from gsplat import rasterization, lidar_rasterization; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
        ;;
    pose)
        checkout omnire "$deps/omnire"
        git -C "$deps/omnire" submodule update --init --recursive third_party/Humans4D
        [[ "$(git -C "$deps/omnire/third_party/Humans4D" rev-parse HEAD)" == "$(pin humans4d commit)" ]]
        venv="$deps/pose-env"
        uv venv --allow-existing --python 3.10 "$venv"
        py="$venv/bin/python"
        install_torch "$py"
        uv pip install --python "$py" 'setuptools<70' wheel ninja pip 'numpy<2'
        printf 'numpy<2\ntorch==2.7.1+%s\ntorchvision==0.22.1+%s\n' "$cuda_tag" "$cuda_tag" > "$deps/pose-constraints.txt"
        uv pip install --python "$py" --no-build-isolation -c "$deps/pose-constraints.txt" \
            "git+$(pin phalp url)@$(pin phalp commit)" \
            "git+$(pin pytorch3d url)@$(pin pytorch3d commit)" \
            "git+https://github.com/ziyc/pandaset-devkit.git@0c10f2b03bc00b9c77950f86e478e271edba2c00#subdirectory=python" \
            "$deps/omnire/third_party/Humans4D[all]" chumpy scipy imageio-ffmpeg omegaconf joblib
        activation "$venv"
        ;;
    *)
        echo "Usage: bash scripts/bootstrap-colab.sh [source|train|pose]" >&2
        exit 2
        ;;
esac
