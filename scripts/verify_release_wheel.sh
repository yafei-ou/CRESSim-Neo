#!/usr/bin/env bash
# Validate a repaired Linux release wheel and its selected runtime lane.
set -euo pipefail

LANE=""

usage() {
    cat <<'EOF'
Usage: scripts/verify_release_wheel.sh --lane base|cu126 path/to/wheel.whl
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
        -*)
            usage >&2
            exit 2
            ;;
        *)
            if [[ -n "${WHEEL:-}" ]]; then
                usage >&2
                exit 2
            fi
            WHEEL="$1"
            shift
            ;;
    esac
done

case "${LANE}" in
    base|cu126) ;;
    *)
        echo "Unsupported release wheel lane: ${LANE}. Supported lanes: base, cu126." >&2
        exit 2
        ;;
esac

: "${WHEEL:?Usage: scripts/verify_release_wheel.sh --lane base|cu126 path/to/wheel.whl}"
[[ -f "${WHEEL}" ]] || { echo "Wheel does not exist: ${WHEEL}" >&2; exit 2; }

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT
unzip -q "${WHEEL}" -d "${WORK_DIR}"

if find "${WORK_DIR}" -type f \( -name 'libcudart.so*' -o -name 'libcufft.so*' -o -name 'libcurand.so*' \) | grep -q .; then
    echo "CUDA libraries must not be bundled in a release wheel" >&2
    exit 1
fi

found_cudart=OFF
while IFS= read -r -d '' library; do
    dynamic="$(readelf -d "${library}")"
    if grep -q '/usr/local/cuda\|/opt/conda\|/home/' <<<"${dynamic}"; then
        echo "Build-machine path found in ${library}" >&2
        exit 1
    fi

    case "${LANE}" in
        base)
            if grep -q 'Shared library: \[lib\(cuda\|cudart\|cufft\|curand\)' <<<"${dynamic}"; then
                echo "Base wheel must not request CUDA libraries: ${library}" >&2
                exit 1
            fi
            ;;
        cu126)
            if grep -q 'Shared library: \[libcudart.so.12\]' <<<"${dynamic}"; then
                found_cudart=ON
            fi
            ;;
    esac
done < <(find "${WORK_DIR}" -type f -name '*.so*' -print0)

if [[ "${LANE}" == cu126 && "${found_cudart}" != ON ]]; then
    echo "CUDA 12.6 wheel does not request libcudart.so.12" >&2
    exit 1
fi

if command -v auditwheel >/dev/null; then
    auditwheel_output="$(auditwheel show "${WHEEL}")"
    printf '%s\n' "${auditwheel_output}"
    # auditwheel versions differ in whether the first summary line reports the
    # wheel's existing linux tag or its inferred manylinux tag. Require both
    # the repaired filename tag and the authoritative ABI constraint instead.
    if [[ "$(basename "${WHEEL}")" != *-manylinux_2_34_x86_64.whl ]] || \
       ! grep -Fq 'constrains the platform tag to "manylinux_2_34_x86_64"' <<<"${auditwheel_output}"; then
        echo "Release wheel must be repaired to manylinux_2_34_x86_64" >&2
        exit 1
    fi
fi

echo "Verified ${LANE} release wheel: ${WHEEL}"
