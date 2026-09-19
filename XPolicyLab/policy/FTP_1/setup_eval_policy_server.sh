#!/usr/bin/env bash
set -euo pipefail

if (( $# < 9 || $# > 10 )); then
    echo "Usage: $0 <bench_name> <task_name> <ckpt_name> <env_cfg_type> ee <seed> <gpu_id> <policy_conda_env> <port> [bind_host]" >&2
    exit 2
fi

bench_name=${1:?bench_name required}
task_name=${2:?task_name required}
ckpt_name=${3:?ckpt_name required}
env_cfg_type=${4:?env_cfg_type required}
action_type=${5:?action_type required}
seed=${6:?seed required}
policy_gpu_id=${7:?policy_gpu_id required}
policy_conda_env=${8:?policy_conda_env required}
policy_server_port=${9:?policy_server_port required}
# Split-machine deployment is the purpose of this entry point. eval.sh passes
# localhost explicitly for same-machine evaluation.
policy_server_host=${10:-"0.0.0.0"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
FTP1_UPSTREAM_ROOT="${FTP1_UPSTREAM_ROOT:-${SCRIPT_DIR}/ftp1-policy}"
WORKSPACE_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
SHARED_ROOT="${XPARK_SHARED_ROOT:-/vepfs-cnbje63de6fae220/xspark_shared}"
CONDA_BIN="${CONDA_BIN:-${SHARED_ROOT}/miniconda3/bin/conda}"
policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${FTP1_DEPLOY_CONFIG:-${SCRIPT_DIR}/deploy.yml}"

if [[ "${action_type}" != "ee" ]]; then
    echo "[SERVER][ERROR] FTP_1 only supports world-coordinate EE control; action_type must be ee." >&2
    exit 2
fi
if [[ ! "${policy_server_port}" =~ ^[0-9]+$ ]] || (( policy_server_port < 1 || policy_server_port > 65535 )); then
    echo "[SERVER][ERROR] invalid port: ${policy_server_port}" >&2
    exit 2
fi
if [[ ! -x "${CONDA_BIN}" ]]; then
    echo "[SERVER][ERROR] conda not found: ${CONDA_BIN}; set CONDA_BIN explicitly." >&2
    exit 1
fi
if [[ ! -f "${yaml_file}" ]]; then
    echo "[SERVER][ERROR] deploy config not found: ${yaml_file}" >&2
    exit 1
fi

echo "[SERVER] policy=${policy_name}, task=${task_name}, port=${policy_server_port}"

CONDA_BASE="$("${CONDA_BIN}" info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

export HF_HOME="${HF_HOME:-${SHARED_ROOT}/cache/huggingface}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${SCRIPT_DIR}/assets/openpi}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SHARED_ROOT}/cache/uv}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python - <<'PY'
import importlib.util

required = ("yaml", "websockets", "msgpack", "msgpack_numpy", "pydantic")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("[SERVER][ERROR] policy environment is missing: " + ", ".join(missing))
PY

exec env \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONPATH="${BENCH_ROOT}:${FTP1_UPSTREAM_ROOT}:${FTP1_UPSTREAM_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    python "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            gpu_id="${policy_gpu_id}" \
            policy_name="${policy_name}" \
            action_type="${action_type}"
