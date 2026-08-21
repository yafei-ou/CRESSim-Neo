#!/usr/bin/env bash
# Build a local Python wheel through the PEP 517 packaging configuration.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEEL_DIR="${REPO_ROOT}/dist"

usage() {
    cat <<'EOF'
Usage: scripts/build_local_wheel.sh [--output DIRECTORY]

Builds a local Release wheel with interactive viewer bindings.
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

command=(python3 -m pip wheel --no-deps --wheel-dir "${WHEEL_DIR}"
    -Ccmake.define.CRESSIM_NEO_BUILD_VIEWER=ON)
command+=("${REPO_ROOT}")

run_command mkdir -p "${WHEEL_DIR}"
run_command "${command[@]}"
