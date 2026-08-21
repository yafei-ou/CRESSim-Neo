#!/usr/bin/env bash
# Build reproducible Linux release wheels for one supported binary lane.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LANE=""

usage() {
    cat <<'EOF'
Usage: scripts/build_release_wheels.sh --lane base|cu126

Builds all configured CPython release wheels in Docker and writes them to
dist/<lane>. CIBW_BUILD may narrow the ABI set for local iteration.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --lane)
            (( $# >= 2 )) || { usage >&2; exit 2; }
            LANE="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
done

case "${LANE}" in
    base)
        IMAGE_NAME="cressim-neo/manylinux-base:local"
        DOCKERFILE="${REPO_ROOT}/docker/manylinux-base/Dockerfile"
        PACKAGE_DIR="${REPO_ROOT}/packaging/base"
        ;;
    cu126)
        IMAGE_NAME="cressim-neo/manylinux-cu126:local"
        DOCKERFILE="${REPO_ROOT}/docker/manylinux-cu126/Dockerfile"
        PACKAGE_DIR="${REPO_ROOT}/packaging/cuda126"
        ;;
    *)
        echo "Unsupported release wheel lane: ${LANE}. Supported lanes: base, cu126." >&2
        exit 2
        ;;
esac

command -v docker >/dev/null || { echo "docker is required" >&2; exit 1; }
command -v cibuildwheel >/dev/null || { echo "cibuildwheel is required (pip install cibuildwheel)" >&2; exit 1; }

OUTPUT_DIR="${REPO_ROOT}/dist/${LANE}"
# Release artifacts must come only from this invocation. The lane is selected
# from the fixed list above, so this removes only dist/base or dist/cu126.
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

docker build -f "${DOCKERFILE}" -t "${IMAGE_NAME}" "${REPO_ROOT}"

export CIBW_MANYLINUX_X86_64_IMAGE="${IMAGE_NAME}"
if [[ "${LANE}" == cu126 ]]; then
    export CIBW_ENVIRONMENT_LINUX="CUDA_HOME=/usr/local/cuda-12.6 CUDAToolkit_ROOT=/usr/local/cuda-12.6 CUDACXX=/usr/local/cuda-12.6/bin/nvcc CUDAHOSTCXX=/opt/rh/gcc-toolset-13/root/usr/bin/g++"
    if [[ "${CRESSIM_NEO_SKIP_AUDITWHEEL:-0}" == 1 ]]; then
        # Preserve an unrepaired wheel for diagnosing a manylinux policy failure.
        # This is never suitable for publication.
        export CIBW_REPAIR_WHEEL_COMMAND='cp {wheel} {dest_dir}'
    fi
else
    unset CIBW_ENVIRONMENT_LINUX
    unset CIBW_REPAIR_WHEEL_COMMAND
fi

cibuildwheel "${PACKAGE_DIR}" --output-dir "${OUTPUT_DIR}"
