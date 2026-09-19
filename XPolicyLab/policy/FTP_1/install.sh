#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${XPOLICYLAB_ROOT}/.." && pwd)"

FTP1_UPSTREAM_ROOT="${FTP1_UPSTREAM_ROOT:-${POLICY_DIR}/ftp1-policy}"
FTP1_UPSTREAM_URL="${FTP1_UPSTREAM_URL:-https://github.com/michaelyuancb/ftp1-policy.git}"
FTP1_UPSTREAM_COMMIT="${FTP1_UPSTREAM_COMMIT:-89fa681d6c014cce28300946b7526db808e0b1c1}"
FTP1_ENV_PREFIX="${FTP1_ENV_PREFIX:-${WORKSPACE_ROOT}/.venvs/FTP_1}"
CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${WORKSPACE_ROOT}/.cache/uv}"

if [[ ! -x "${CONDA_BIN}" ]]; then
    echo "[FTP_1][ERROR] conda not found: ${CONDA_BIN}" >&2
    exit 1
fi
if [[ ! -x "${UV_BIN}" ]]; then
    echo "[FTP_1][ERROR] uv not found: ${UV_BIN}" >&2
    exit 1
fi
if [[ ! -f "${FTP1_UPSTREAM_ROOT}/pyproject.toml" || ! -f "${FTP1_UPSTREAM_ROOT}/uv.lock" ]]; then
    echo "[FTP_1] Cloning pinned upstream into ${FTP1_UPSTREAM_ROOT}"
    git clone "${FTP1_UPSTREAM_URL}" "${FTP1_UPSTREAM_ROOT}"
    git -C "${FTP1_UPSTREAM_ROOT}" checkout --detach "${FTP1_UPSTREAM_COMMIT}"
fi
if [[ ! -f "${FTP1_UPSTREAM_ROOT}/pyproject.toml" || ! -f "${FTP1_UPSTREAM_ROOT}/uv.lock" ]]; then
    echo "[FTP_1][ERROR] incomplete FTP-1 upstream checkout: ${FTP1_UPSTREAM_ROOT}" >&2
    exit 1
fi
if [[ -d "${FTP1_UPSTREAM_ROOT}/.git" ]]; then
    actual_commit="$(git -C "${FTP1_UPSTREAM_ROOT}" rev-parse HEAD)"
    if [[ "${actual_commit}" != "${FTP1_UPSTREAM_COMMIT}" && "${FTP1_ALLOW_UNPINNED:-0}" != "1" ]]; then
        echo "[FTP_1][ERROR] expected upstream ${FTP1_UPSTREAM_COMMIT}, found ${actual_commit}" >&2
        echo "[FTP_1][ERROR] set FTP1_ALLOW_UNPINNED=1 only for an intentional source upgrade" >&2
        exit 1
    fi
else
    echo "[FTP_1] Using vendored FTP-1 source based on ${FTP1_UPSTREAM_COMMIT}"
fi

if [[ ! -x "${FTP1_ENV_PREFIX}/bin/python" ]]; then
    echo "[FTP_1] Creating Python 3.11 conda environment at ${FTP1_ENV_PREFIX}"
    "${CONDA_BIN}" create -y --prefix "${FTP1_ENV_PREFIX}" python=3.11 pip
fi

CONDA_BASE="$("${CONDA_BIN}" info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${FTP1_ENV_PREFIX}"

echo "[FTP_1] Syncing the official locked environment"
export GIT_LFS_SKIP_SMUDGE=1
export UV_CACHE_DIR
export UV_LINK_MODE=copy
export UV_PROJECT_ENVIRONMENT="${FTP1_ENV_PREFIX}"
"${UV_BIN}" sync \
    --project "${FTP1_UPSTREAM_ROOT}" \
    --python "${FTP1_ENV_PREFIX}/bin/python" \
    --frozen \
    --inexact

"${UV_BIN}" pip install --python "${FTP1_ENV_PREFIX}/bin/python" --no-deps -e "${FTP1_UPSTREAM_ROOT}"
"${UV_BIN}" pip install --python "${FTP1_ENV_PREFIX}/bin/python" --no-deps -e "${XPOLICYLAB_ROOT}"

SITE_PACKAGES="$("${FTP1_ENV_PREFIX}/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
TRANSFORMERS_REPLACE="${FTP1_UPSTREAM_ROOT}/src/openpi/models_pytorch/transformers_replace"
if [[ -d "${TRANSFORMERS_REPLACE}" ]]; then
    cp -r "${TRANSFORMERS_REPLACE}/." "${SITE_PACKAGES}/transformers/"
fi

"${FTP1_ENV_PREFIX}/bin/python" - <<'PY'
import torch
import XPolicyLab
import openpi
from openpi.policies import FTP1InferenceWrapper

print(f"[FTP_1] XPolicyLab import: {XPolicyLab.__file__}")
print(f"[FTP_1] openpi import: {openpi.__file__}")
print(f"[FTP_1] torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")
print(f"[FTP_1] wrapper import: {FTP1InferenceWrapper.__name__}")
PY

echo "[FTP_1] Installation complete"
echo "[FTP_1] Activate with: conda activate ${FTP1_ENV_PREFIX}"
