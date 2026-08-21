#!/usr/bin/env bash
# Create the supported Ubuntu managed environment and install a locally or
# remotely published CRESSim CUDA 12.6 wheel.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-cressim-cu126}"
CUDA_INDEX_URL="${CRESSIM_NEO_CUDA_INDEX_URL:-https://yafei-ou.github.io/CRESSim-Neo/whl/cu126}"

command -v micromamba >/dev/null || { echo "micromamba is required" >&2; exit 1; }
micromamba create -y -n "${ENV_NAME}" -f "${REPO_ROOT}/packaging/cuda126/environment.yml"
micromamba run -n "${ENV_NAME}" python -m pip install \
    --index-url "${CUDA_INDEX_URL}" \
    --extra-index-url https://pypi.org/simple \
    cressim-neo
micromamba run -n "${ENV_NAME}" python -m pip check
echo "Environment '${ENV_NAME}' is ready. Activate it with: micromamba activate ${ENV_NAME}"
