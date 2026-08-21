#!/usr/bin/env bash
# Validate that a managed wheel requests, but does not bundle, CUDA libraries.
set -euo pipefail

WHEEL="${1:?Usage: scripts/verify_cuda126_wheel.sh path/to/wheel.whl}"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

unzip -q "${WHEEL}" -d "${WORK_DIR}"
if find "${WORK_DIR}" -type f \( -name 'libcudart.so*' -o -name 'libcufft.so*' -o -name 'libcurand.so*' \) | grep -q .; then
    echo "CUDA libraries must not be bundled in a managed wheel" >&2
    exit 1
fi

found_cudart=OFF
while IFS= read -r -d '' library; do
    dynamic="$(readelf -d "${library}")"
    if grep -q '/usr/local/cuda\|/opt/conda\|/home/' <<<"${dynamic}"; then
        echo "Build-machine path found in ${library}" >&2
        exit 1
    fi
    if grep -q 'Shared library: \[libcudart.so.12\]' <<<"${dynamic}"; then
        found_cudart=ON
    fi
done < <(find "${WORK_DIR}" -type f -name '*.so*' -print0)

if [[ "${found_cudart}" != ON ]]; then
    echo "No wheel library requests libcudart.so.12" >&2
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
        echo "Managed CUDA wheel must be repaired to manylinux_2_34_x86_64" >&2
        exit 1
    fi
fi
