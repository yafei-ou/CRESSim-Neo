#!/usr/bin/env bash
# Build a local Python wheel through the PEP 517 packaging configuration.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEEL_DIR="${REPO_ROOT}/dist"
WITH_VIEWER=OFF

usage() {
    cat <<'EOF'
Usage: scripts/build_wheel.sh [--viewer] [--output DIRECTORY]

Builds a local Release wheel. --viewer enables the interactive viewer bindings
for this wheel. A viewer-enabled wheel must not be published beside a headless
wheel with the same package version and platform tag.
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

while (( $# > 0 )); do
    case "$1" in
        --viewer)
            WITH_VIEWER=ON
            shift
            ;;
        --output)
            (( $# >= 2 )) || { usage >&2; exit 2; }
            WHEEL_DIR="$2"
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

if [[ "${WITH_VIEWER}" == ON && "${WHEEL_DIR}" == "${REPO_ROOT}/dist" ]]; then
    WHEEL_DIR="${REPO_ROOT}/dist/viewer"
fi

command=(python3 -m pip wheel --no-deps --wheel-dir "${WHEEL_DIR}")
if [[ "${WITH_VIEWER}" == ON ]]; then
    command+=(-Ccmake.define.CRESSIM_NEO_BUILD_VIEWER=ON)
fi
command+=("${REPO_ROOT}")

run_command mkdir -p "${WHEEL_DIR}"
run_command "${command[@]}"
