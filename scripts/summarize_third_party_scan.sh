#!/usr/bin/env bash
# Create the compact review dashboard from an existing ScanCode JSON report.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT="${REPO_ROOT}/build/compliance/scancode.json"
REVIEW="${REPO_ROOT}/compliance/third_party_review.json"
OUTPUT="${REPO_ROOT}/build/compliance/third_party_review.md"

usage() {
    cat <<'EOF'
Usage: scripts/summarize_third_party_scan.sh [--report <path>] [--output <path>]

Creates a compact Markdown dashboard and updates the persistent review registry.
Existing include/exclude/pending decisions and notes are preserved. If the
registry does not exist, it is initialized with pending decisions.
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

while (( $# > 0 )); do
    case "$1" in
        --report)
            (( $# >= 2 )) || die "--report requires a path."
            REPORT="$2"
            shift 2
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

command -v python3 >/dev/null 2>&1 || die "python3 is required to summarize the ScanCode report."
[[ -f "${REPORT}" ]] || die "ScanCode report not found: ${REPORT}"

python3 "${REPO_ROOT}/scripts/summarize_scancode_report.py" \
    --report "${REPORT}" \
    --review "${REVIEW}" \
    --output "${OUTPUT}"
