#!/usr/bin/env bash
set -euo pipefail

# Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
bench_name=${1:?bench_name is required}
ckpt_name=${2:?ckpt_name is required}
env_cfg_type=${3:?env_cfg_type is required}
action_type=${4:?action_type is required}
seed=${5:?seed is required}
gpu_id=${6:?gpu_id is required}

if [[ "${bench_name}" != "Spark0_real_bench_v5" || "${ckpt_name}" != "Moxian" || \
      "${env_cfg_type}" != "tianji_marvin_wuji" || "${action_type}" != "ee" ]]; then
    echo "[FTP_1][ERROR] the current training recipe supports only:" >&2
    echo "[FTP_1][ERROR] Spark0_real_bench_v5 Moxian tianji_marvin_wuji ee" >&2
    exit 1
fi

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "${POLICY_DIR}/../../.." && pwd)"
FTP1_UPSTREAM_ROOT="${FTP1_UPSTREAM_ROOT:-${POLICY_DIR}/ftp1-policy}"
FTP1_PYTHON="${FTP1_PYTHON:-${BENCH_ROOT}/.venvs/FTP_1/bin/python}"
NORM_LAUNCHER="${POLICY_DIR}/run_compute_norm_stats.py"
DATASET_CONFIG="${FTP1_DATASET_CONFIG:-${POLICY_DIR}/data_scripts/dataset_spark0_real_bench_v5_moxian_joint_only.json}"
VALIDATION_REPORT="${FTP1_VALIDATION_REPORT:-${BENCH_ROOT}/data/spark0_real_bench_v5_moxian_joint_only/validation_report.json}"
EXPECTED_SOURCE_ROOT="${FTP1_EXPECTED_SOURCE_ROOT:-/vepfs-cnbje63de6fae220/xiangpc/data/bench_v5}"
PRETRAIN_ROOT="${FTP1_PRETRAIN_ROOT:-${BENCH_ROOT}/pretrain_model}"
ASSETS_BASE_DIR="${FTP1_ASSETS_BASE_DIR:-${POLICY_DIR}/assets}"
NATIVE_CHECKPOINT_BASE="${FTP1_NATIVE_CHECKPOINT_BASE:-${POLICY_DIR}/checkpoints/.ftp1_native}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${POLICY_DIR}/assets/openpi}"

stage="${FTP1_STAGE:-all}"
case "${stage}" in
    norm|train|all) ;;
    *) echo "[FTP_1][ERROR] FTP1_STAGE must be norm, train, or all" >&2; exit 1 ;;
esac

[[ -x "${FTP1_PYTHON}" ]] || { echo "[FTP_1][ERROR] Python not found: ${FTP1_PYTHON}" >&2; exit 1; }
[[ -f "${FTP1_UPSTREAM_ROOT}/scripts/zarr_compute_norm_stats.py" ]] || {
    echo "[FTP_1][ERROR] nested upstream missing: ${FTP1_UPSTREAM_ROOT}" >&2
    exit 1
}
FTP1_UPSTREAM_ROOT="$(cd "${FTP1_UPSTREAM_ROOT}" && pwd -P)"
[[ -f "${NORM_LAUNCHER}" ]] || { echo "[FTP_1][ERROR] normalization launcher missing: ${NORM_LAUNCHER}" >&2; exit 1; }
[[ -f "${DATASET_CONFIG}" ]] || { echo "[FTP_1][ERROR] dataset config missing: ${DATASET_CONFIG}" >&2; exit 1; }

repo_id="${FTP1_REPO_ID:-Spark0_real_bench_v5_Moxian_joint_only}"
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
exp_name="${FTP1_EXP_NAME:-${ckpt_setting}}"
standard_checkpoint_root="${POLICY_DIR}/checkpoints/${exp_name}"
native_run_root="${NATIVE_CHECKPOINT_BASE}/ftp1/${exp_name}"

local_batch_size="${FTP1_LOCAL_BATCH_SIZE:-8}"
num_train_steps="${FTP1_NUM_TRAIN_STEPS:-200000}"
save_interval="${FTP1_SAVE_INTERVAL:-10000}"
val_interval="${FTP1_VAL_INTERVAL:-10000}"
log_interval="${FTP1_LOG_INTERVAL:-25}"
num_workers="${FTP1_NUM_WORKERS:-16}"
val_num_workers="${FTP1_VAL_NUM_WORKERS:-2}"
norm_num_workers="${FTP1_NORM_NUM_WORKERS:-16}"
norm_batch_size="${FTP1_NORM_BATCH_SIZE:-32}"
norm_sample_ratio="${FTP1_NORM_SAMPLE_RATIO:-1.0}"
action_down_sample_steps="${FTP1_ACTION_DOWN_SAMPLE_STEPS:-1}"
used_image_keys="${FTP1_USED_IMAGE_KEYS:-camera_ego_rgb}"
action_horizon="${FTP1_ACTION_HORIZON:-33}"
wandb_enabled="${FTP1_WANDB_ENABLED:-1}"
wandb_project="${FTP1_WANDB_PROJECT:-xpolicylab-0910-real}"
wandb_mode="${FTP1_WANDB_MODE:-online}"
wandb_entity="${FTP1_WANDB_ENTITY:-${WANDB_ENTITY:-}}"

case "${wandb_enabled}" in
    0|1) ;;
    *) echo "[FTP_1][ERROR] FTP1_WANDB_ENABLED must be 0 or 1" >&2; exit 1 ;;
esac
case "${wandb_mode}" in
    online|offline|disabled) ;;
    *) echo "[FTP_1][ERROR] FTP1_WANDB_MODE must be online, offline, or disabled" >&2; exit 1 ;;
esac

IFS=',' read -r -a gpu_list <<< "${gpu_id}"
num_gpus=${#gpu_list[@]}
(( num_gpus > 0 )) || num_gpus=1
nnodes="${FTP1_NNODES:-1}"
node_rank="${FTP1_NODE_RANK:-0}"
master_addr="${FTP1_MASTER_ADDR:-127.0.0.1}"
master_port="${FTP1_MASTER_PORT:-29500}"
(( nnodes >= 1 )) || { echo "[FTP_1][ERROR] FTP1_NNODES must be >= 1" >&2; exit 1; }
(( node_rank >= 0 && node_rank < nnodes )) || {
    echo "[FTP_1][ERROR] FTP1_NODE_RANK must be in [0, ${nnodes})" >&2
    exit 1
}
world_size=$((nnodes * num_gpus))
batch_size=$((local_batch_size * world_size))

pretrain_checkpoint="${FTP1_PRETRAIN_CHECKPOINT:-}"
if [[ -z "${pretrain_checkpoint}" && -d "${PRETRAIN_ROOT}" ]]; then
    pretrain_model_file="$(find "${PRETRAIN_ROOT}" -type f -name model.safetensors -print 2>/dev/null | sort -V | tail -n 1)"
    if [[ -n "${pretrain_model_file}" ]]; then
        pretrain_checkpoint="$(dirname "${pretrain_model_file}")"
    fi
fi

if [[ "${stage}" != "norm" && -z "${pretrain_checkpoint}" && "${FTP1_DRY_RUN:-0}" != "1" ]]; then
    echo "[FTP_1][ERROR] no FTP-1 pretrained checkpoint below ${PRETRAIN_ROOT}" >&2
    echo "[FTP_1][ERROR] place the official weights there or set FTP1_PRETRAIN_CHECKPOINT" >&2
    exit 1
fi

common_args=(
    ftp1
    "--repo_id=${repo_id}"
    "--data.repo-id=${repo_id}"
    "--dataset_config_path=${DATASET_CONFIG}"
    "--checkpoint_base_dir=${NATIVE_CHECKPOINT_BASE}"
    "--assets_base_dir=${ASSETS_BASE_DIR}"
    "--action_down_sample_steps=${action_down_sample_steps}"
    --use_val_dataset
    --create_train_val_split
    --norm_type=zscore
    --independent_norm_mode=all
    --proprioception_pose_rep=abs
    --action_pose_rep=abs
    --proprioception_joint_rep=abs
    --action_joint_rep=absolute
    "--used_image_keys=${used_image_keys}"
    "--model.action_horizon=${action_horizon}"
    "--seed=${seed}"
)

norm_command=(
    "${FTP1_PYTHON}" "${NORM_LAUNCHER}"
    "${common_args[@]}"
    --exp_name=moxian_norm
    --val_ratio=0.01
    "--batch_size=${batch_size}"
    "--norm_sample_ratio=${norm_sample_ratio}"
    "--norm_batch_size=${norm_batch_size}"
    "--norm_num_workers=${norm_num_workers}"
    --no-wandb_enabled
)

train_args=(
    "${common_args[@]}"
    "--exp_name=${exp_name}"
    "--batch_size=${batch_size}"
    "--num_train_steps=${num_train_steps}"
    "--log_interval=${log_interval}"
    "--val_interval=${val_interval}"
    "--save_interval=${save_interval}"
    "--keep_period=${save_interval}"
    "--num_workers=${num_workers}"
    "--val_num_workers=${val_num_workers}"
    --model.state_input_mode=adarms
    --model.tactile_expert_variant=gemma_small
    --lr_schedule.warmup_steps=300
    --lr_schedule.peak_lr=5e-5
    "--lr_schedule.decay_steps=${num_train_steps}"
    --lr_schedule.decay_lr=3e-6
    --optimizer.b1=0.9
    --optimizer.b2=0.95
    --optimizer.eps=1e-8
    --optimizer.weight_decay=1e-10
    --optimizer.clip_gradient_norm=1.5
)
if [[ "${wandb_enabled}" == "1" ]]; then
    train_args+=(--wandb_enabled "--project-name=${wandb_project}")
else
    train_args+=(--no-wandb_enabled)
fi
if [[ "${FTP1_RESUME:-0}" == "1" ]]; then
    train_args+=(--resume)
    echo "[FTP_1] resume=1, skip pytorch_weight_path, load latest checkpoint under ${native_run_root}"
elif [[ -n "${pretrain_checkpoint}" ]]; then
    train_args+=("--pytorch_weight_path=${pretrain_checkpoint}")
fi

if (( nnodes > 1 )); then
    train_command=(
        "${FTP1_PYTHON}" -m torch.distributed.run
        "--nnodes=${nnodes}"
        "--nproc_per_node=${num_gpus}"
        "--node_rank=${node_rank}"
        "--master_addr=${master_addr}"
        "--master_port=${master_port}"
        scripts/zarr_train_ftp1_pytorch.py
        "${train_args[@]}"
    )
elif (( num_gpus > 1 )); then
    train_command=(
        "${FTP1_PYTHON}" -m torch.distributed.run --standalone --nnodes=1
        "--nproc_per_node=${num_gpus}" scripts/zarr_train_ftp1_pytorch.py
        "${train_args[@]}"
    )
else
    train_command=("${FTP1_PYTHON}" scripts/zarr_train_ftp1_pytorch.py "${train_args[@]}")
fi

print_command() {
    printf '  %q' "$@"
    printf '\n'
}

echo "[FTP_1] stage=${stage}, dataset=${DATASET_CONFIG}"
echo "[FTP_1] state/action contract=120D container, EE pose + 40 hand-joint dimensions (arm/head slots omitted), direct HDF5 actions, tactile enabled"
echo "[FTP_1] used_image_keys=${used_image_keys} (camera_ego_rgb=single-view head; all=every present camera)"
echo "[FTP_1] action_horizon=${action_horizon} (deploy execute_horizon=32 after action_start_index=1)"
echo "[FTP_1] pretrained_root=${PRETRAIN_ROOT}"
echo "[FTP_1] pretrained_checkpoint=${pretrain_checkpoint:-NOT_FOUND}"
echo "[FTP_1] standard_checkpoint_root=${standard_checkpoint_root}"
echo "[FTP_1] native_checkpoint_root=${native_run_root}"
echo "[FTP_1] GPUs=${gpu_id}, nnodes=${nnodes}, node_rank=${node_rank}, world_size=${world_size}, global_batch=${batch_size}, local_batch=${local_batch_size}, steps=${num_train_steps}, val_interval=${val_interval}, save_interval=${save_interval}"
echo "[FTP_1] master=${master_addr}:${master_port}"
echo "[FTP_1] wandb_enabled=${wandb_enabled}, wandb_entity=${wandb_entity:-UNSET}, wandb_project=${wandb_project}, wandb_mode=${wandb_mode}"

if [[ "${FTP1_DRY_RUN:-0}" == "1" ]]; then
    if [[ "${stage}" == "norm" || "${stage}" == "all" ]]; then
        echo "[FTP_1][DRY-RUN] normalization command:"
        print_command "${norm_command[@]}"
    fi
    if [[ "${stage}" == "train" || "${stage}" == "all" ]]; then
        echo "[FTP_1][DRY-RUN] training command:"
        print_command "${train_command[@]}"
        [[ -n "${pretrain_checkpoint}" ]] || echo "[FTP_1][DRY-RUN][WARN] pretrain_model is currently empty"
    fi
    echo "[FTP_1][DRY-RUN] no normalization or training was started"
    exit 0
fi

"${FTP1_PYTHON}" - "${VALIDATION_REPORT}" "${DATASET_CONFIG}" "${repo_id}" "${EXPECTED_SOURCE_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
dataset_config_path = Path(sys.argv[2])
expected_repo_id = sys.argv[3]
expected_source_root = Path(sys.argv[4]).resolve()
if not report_path.is_file():
    raise SystemExit(
        f"[FTP_1][ERROR] validated joint-only data is required before norm/train: {report_path}"
    )
report = json.loads(report_path.read_text(encoding="utf-8"))
dataset_config = json.loads(dataset_config_path.read_text(encoding="utf-8"))
enabled_datasets = [
    item
    for item in dataset_config.get("datasets", [])
    if item.get("enabled", True)
]
if len(enabled_datasets) != 1:
    raise SystemExit(
        "[FTP_1][ERROR] joint-only recipe requires exactly one enabled dataset; "
        f"got {len(enabled_datasets)}"
    )
dataset_entry = enabled_datasets[0]
if dataset_entry.get("name") != expected_repo_id:
    raise SystemExit(
        "[FTP_1][ERROR] dataset name/repo_id mismatch: "
        f"{dataset_entry.get('name')!r} != {expected_repo_id!r}"
    )
dataset_root = Path(str(dataset_entry.get("path", ""))).resolve()
report_output_root = Path(str(report.get("output_root", ""))).resolve()
if dataset_root != report_output_root:
    raise SystemExit(
        "[FTP_1][ERROR] dataset config/report output mismatch: "
        f"{dataset_root} != {report_output_root}"
    )
if not dataset_root.is_dir():
    raise SystemExit(f"[FTP_1][ERROR] validated dataset root is missing: {dataset_root}")
report_source_root = Path(str(report.get("source_root", ""))).resolve()
if report_source_root != expected_source_root:
    raise SystemExit(
        "[FTP_1][ERROR] validation report was generated from the wrong source: "
        f"{report_source_root} != {expected_source_root}"
    )
expected = int(report.get("expected_episode_count", -1))
present = int(report.get("present_expected_count", -2))
valid = int(report.get("valid_episode_count", -3))
if expected <= 0 or present != expected or valid != expected:
    raise SystemExit(
        "[FTP_1][ERROR] conversion is incomplete: "
        f"expected={expected}, present={present}, valid={valid}"
    )
for field in (
    "missing_episode_count",
    "extra_episode_count",
    "invalid_episode_count",
    "metadata_missing_count",
    "metadata_contract_error_count",
):
    if int(report.get(field, -1)) != 0:
        raise SystemExit(f"[FTP_1][ERROR] validation report has {field}={report.get(field)!r}")
if report.get("joint_only_120d") is not True:
    raise SystemExit("[FTP_1][ERROR] validation report is not joint_only_120d=true")
if int(report.get("ftp1_width", -1)) != 120:
    raise SystemExit("[FTP_1][ERROR] validation report does not declare FTP-1 width 120")
if int(report.get("active_action_dimensions_per_step", -1)) != 58:
    raise SystemExit("[FTP_1][ERROR] joint-only action mask must have 58 active dimensions")
print(f"[FTP_1] validated joint-only dataset: {valid}/{expected} episodes")
PY

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export OPENPI_DATA_HOME
export FTP1_UPSTREAM_ROOT
export HYDRA_FULL_ERROR=1
if [[ "${wandb_enabled}" == "1" ]]; then
    export USE_SWANLAB=0
    export WANDB_MODE="${wandb_mode}"
    export WANDB_ENTITY="${wandb_entity}"
fi
cd "${FTP1_UPSTREAM_ROOT}"

if [[ "${stage}" == "norm" || "${stage}" == "all" ]]; then
    if (( node_rank != 0 )); then
        echo "[FTP_1] skip normalization on node_rank=${node_rank}"
    else
        "${norm_command[@]}"
    fi
fi

if [[ "${stage}" == "train" || "${stage}" == "all" ]]; then
    mkdir -p "$(dirname "${standard_checkpoint_root}")" "$(dirname "${native_run_root}")"
    if [[ -e "${standard_checkpoint_root}" && ! -L "${standard_checkpoint_root}" ]]; then
        echo "[FTP_1][ERROR] standard checkpoint path exists and is not a symlink: ${standard_checkpoint_root}" >&2
        exit 1
    fi
    ln -sfn "${native_run_root}" "${standard_checkpoint_root}"
    "${train_command[@]}"
fi
