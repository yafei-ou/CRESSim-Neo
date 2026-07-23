#!/usr/bin/env bash
# Produce a source-level third-party-license inventory for manual review.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_OUTPUT="${REPO_ROOT}/build/compliance/scancode.json"
OUTPUT="${DEFAULT_OUTPUT}"
INCLUDE_LICENSE_TEXT=OFF
INCLUDE_DILIGENT=OFF

usage() {
    cat <<'EOF'
Usage: scripts/scan_third_party.sh [--include-diligent] [--include-license-text] [--output <path>]

Scans this checkout, including initialized submodules and fetched dependency
sources, for license, copyright, email, and URL information.
The report includes only files with findings, so files with no detected
information are omitted.

Use --include-license-text to embed matched license passages in the report.
This substantially increases report size and is best reserved for a targeted
scan of shortlisted third-party directories or files.

DiligentEngine is excluded by default to support an initial dry run. Use
--include-diligent for the complete repository inventory.

The JSON report is a candidate inventory, not a shipping notice: review it
against each supported install/package profile and retain notices only for code,
assets, headers, and binaries that are actually distributed.

ScanCode Toolkit must be available as the `scancode` command. By default, the
report is written to build/compliance/scancode.json.
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

print_command() {
    printf ' +'
    printf ' %q' "$@"
    printf '\n'
}

while (( $# > 0 )); do
    case "$1" in
        --include-diligent)
            INCLUDE_DILIGENT=ON
            shift
            ;;
        --include-license-text)
            INCLUDE_LICENSE_TEXT=ON
            shift
            ;;
        --output)
            (( $# >= 2 )) || die "--output requires a path."
            OUTPUT="$2"
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

command -v scancode >/dev/null 2>&1 || die \
    "ScanCode Toolkit is required. Install it, then make sure 'scancode' is on PATH."
if [[ ! -f "${REPO_ROOT}/extern/DiligentEngine/CMakeLists.txt" ||
      ! -f "${REPO_ROOT}/extern/CRESSim-Ultrasound/CMakeLists.txt" ||
      ! -f "${REPO_ROOT}/extern/pybind11/CMakeLists.txt" ]]; then
    die "Submodules are not initialized. Run: git submodule update --init --recursive"
fi

mkdir -p "$(dirname "${OUTPUT}")"

command=(
    scancode
    --copyright
    --license
)

if [[ "${INCLUDE_LICENSE_TEXT}" == ON ]]; then
    command+=(--license-text)
fi

command+=(
    --email
    --url
    --only-findings
    --json-pp "${OUTPUT}"
    --ignore "*/.git/**"
    --ignore "*/.agents/**"
    --ignore "*/.cache/**"
    --ignore "*/.codex/**"
    --ignore "*/build/**"
    --ignore "*/dist/**"
    --ignore "*/Testing/**"
)

if [[ "${INCLUDE_DILIGENT}" == OFF ]]; then
    command+=(--ignore "extern/DiligentEngine/**")
    command+=(--ignore "*/extern/DiligentEngine/**")
fi

command+=("${REPO_ROOT}")

print_command "${command[@]}"
"${command[@]}"
echo "Wrote candidate third-party inventory: ${OUTPUT}"
