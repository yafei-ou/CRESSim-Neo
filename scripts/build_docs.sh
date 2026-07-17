#!/usr/bin/env bash
# Build the documentation site from its dedicated native Python API build.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${REPO_ROOT}/build"

usage() {
    cat <<'EOF'
Usage: scripts/build_docs.sh [--configure-only]

Configures and builds build/linux-docs-api, a Release Python API build with viewer
bindings enabled and optional CUDA/ultrasound features disabled. It then
configures build/docs-site and builds the HTML documentation.
EOF
}

print_command() {
    printf ' +'
    printf ' %q' "$@"
    printf '\n'
}

run_command() {
    print_command "$@"
    "$@"
}

CONFIGURE_ONLY=OFF
for argument in "$@"; do
    case "${argument}" in
        --configure-only) CONFIGURE_ONLY=ON ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

API_BUILD_DIR="${BUILD_ROOT}/linux-docs-api"
API_CACHE_FILE="${API_BUILD_DIR}/CMakeCache.txt"
configure_command=(cmake -S "${REPO_ROOT}" -B "${API_BUILD_DIR}")

if [[ -f "${API_CACHE_FILE}" ]]; then
    echo "Reusing dedicated documentation API build: ${API_BUILD_DIR}"
else
    if command -v ninja >/dev/null 2>&1; then
        configure_command+=(-G Ninja)
    else
        configure_command+=(-G 'Unix Makefiles')
    fi
fi

configure_command+=(
    -DCMAKE_BUILD_TYPE=Release
    -DENGINE_STATIC=OFF
    -DCRESSIM_NEO_BUILD_PYTHON=ON
    -DCRESSIM_NEO_BUILD_VIEWER=ON
    -DCRESSIM_NEO_BUILD_EXAMPLES=OFF
    -DBUILD_TESTING=OFF
    -DCRESSIM_NEO_ENABLE_CLANG_TIDY=OFF
    -DCRESSIM_NEO_ENABLE_CUDA_INTEROP=OFF
    -DCRESSIM_NEO_ENABLE_ULTRASOUND=OFF
)

run_command "${configure_command[@]}"

run_command cmake --build "${API_BUILD_DIR}" --target \
    cressim_neo_python cressim_neo_python_package_files
run_command cmake -S "${REPO_ROOT}/docs" -B "${BUILD_ROOT}/docs-site" \
    "-DCRESSIM_NEO_PYTHON_PACKAGE_DIR=${API_BUILD_DIR}/bin"

if [[ "${CONFIGURE_ONLY}" == OFF ]]; then
    run_command cmake --build "${BUILD_ROOT}/docs-site" --target docs-html
fi
