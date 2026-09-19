#!/bin/bash
set -euo pipefail

if (( $# < 10 || $# > 11 )); then
    echo "Usage: $0 <bench_name> <task_name> <ckpt_name> <env_cfg_type> ee <seed> <env_gpu_id> <eval_env_conda_env> <additional_info> <port> [server_ip]" >&2
    exit 2
fi

bench_name=${1:?bench_name required}
task_name=${2:?task_name required}
ckpt_name=${3:?ckpt_name required}
env_cfg_type=${4:?env_cfg_type required}
action_type=${5:?action_type required}
seed=${6:?seed required}
env_gpu_id=${7:?env_gpu_id required}
eval_env_conda_env=${8:?eval_env_conda_env required}
additional_info=${9:?additional_info required}
policy_server_port=${10:?policy_server_port required}
policy_server_ip=${11:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="${XPOLICYLAB_BENCH_ROOT:-$(cd "${XPL_ROOT}/.." && pwd)}"
UTILS_DIR="${XPL_ROOT}/utils"
CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

echo "[CLIENT] policy=${policy_name}, task=${task_name}, server=${policy_server_ip}:${policy_server_port}"

if [[ "${action_type}" != "ee" ]]; then
    echo "[CLIENT][ERROR] FTP_1 only supports world-coordinate EE control; action_type must be ee." >&2
    exit 2
fi
if [[ ! "${policy_server_port}" =~ ^[0-9]+$ ]] || (( policy_server_port < 1 || policy_server_port > 65535 )); then
    echo "[CLIENT][ERROR] invalid port: ${policy_server_port}" >&2
    exit 2
fi
if [[ ! -x "${CONDA_BIN}" ]]; then
    echo "[CLIENT][ERROR] conda not found; set CONDA_BIN to the remote machine's conda executable." >&2
    exit 1
fi
if [[ ! -f "${UTILS_DIR}/setup_env_client.sh" ]]; then
    echo "[CLIENT][ERROR] incomplete XPolicyLab client bundle: ${UTILS_DIR}/setup_env_client.sh is missing." >&2
    exit 1
fi

CONDA_BASE="$("${CONDA_BIN}" info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${eval_env_conda_env}"

python - <<'PY'
import importlib.util

required = ("yaml", "websockets", "msgpack", "msgpack_numpy", "pydantic", "numpy")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("[CLIENT][ERROR] evaluation environment is missing: " + ", ".join(missing))
PY

eval_mode="${EVAL_ENV_TYPE:-sim}"
if [[ "${eval_mode}" == "real" || "${eval_mode}" == "real_world" ]]; then
    real_client="${BENCH_ROOT}/src/task_env/real_env_client.py"
    if [[ ! -f "${real_client}" ]]; then
        echo "[CLIENT][ERROR] real robot entry point not found: ${real_client}" >&2
        echo "[CLIENT][ERROR] set XPOLICYLAB_BENCH_ROOT to the robot evaluation repository containing src/task_env/real_env_client.py." >&2
        exit 1
    fi
    if [[ "${additional_info}" != *"base_cfg="* && -z "${BASE_CFG:-}" ]]; then
        echo "[CLIENT][ERROR] real robot mode requires base_cfg in additional_info or BASE_CFG." >&2
        exit 1
    fi
fi

export PYTHONPATH="${BENCH_ROOT}:${XPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

bash "${UTILS_DIR}/setup_env_client.sh" \
    "${UTILS_DIR}" \
    "${yaml_file}" \
    "${eval_env_conda_env}" \
    "${policy_server_port}" \
    "${bench_name}" \
    "${task_name}" \
    "${env_cfg_type}" \
    "${policy_name}" \
    "${additional_info}" \
    "${BENCH_ROOT}" \
    "${seed}" \
    "${env_gpu_id}" \
    "${policy_server_ip}"
