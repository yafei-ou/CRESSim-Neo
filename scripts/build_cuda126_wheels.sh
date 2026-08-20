#!/usr/bin/env bash
# Build the managed manylinux CUDA 12.6 wheel lane through cibuildwheel.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="cressim-neo/manylinux-cu126:local"
OUTPUT_DIR="${REPO_ROOT}/dist/cu126"

command -v docker >/dev/null || { echo "docker is required" >&2; exit 1; }
command -v cibuildwheel >/dev/null || { echo "cibuildwheel is required (pip install cibuildwheel)" >&2; exit 1; }

docker build -f "${REPO_ROOT}/docker/manylinux-cu126/Dockerfile" -t "${IMAGE_NAME}" "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

export CIBW_MANYLINUX_X86_64_IMAGE="${IMAGE_NAME}"
export CIBW_ENVIRONMENT_LINUX="CUDA_HOME=/usr/local/cuda-12.6 CUDAToolkit_ROOT=/usr/local/cuda-12.6 CUDACXX=/usr/local/cuda-12.6/bin/nvcc CUDAHOSTCXX=/opt/rh/gcc-toolset-13/root/usr/bin/g++"
if [[ "${CRESSIM_NEO_SKIP_AUDITWHEEL:-0}" == 1 ]]; then
    # Preserve an unrepaired wheel for diagnosing a manylinux policy failure.
    # This is never suitable for publication.
    export CIBW_REPAIR_WHEEL_COMMAND='cp {wheel} {dest_dir}'
fi
cibuildwheel "${REPO_ROOT}/packaging/cuda126" --output-dir "${OUTPUT_DIR}"
