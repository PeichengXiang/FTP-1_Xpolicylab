#!/usr/bin/env bash
set -euo pipefail

# Single-node, eight-GPU launcher for bench_v5 real-robot FTP-1.
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${WORKSPACE_ROOT}"

export FTP1_STAGE="${FTP1_STAGE:-all}"
export FTP1_WANDB_ENABLED=1
export FTP1_WANDB_MODE=online
export FTP1_WANDB_PROJECT="${FTP1_WANDB_PROJECT:-xpolicylab-0910-real}"
export FTP1_WANDB_ENTITY="${FTP1_WANDB_ENTITY:-peichengxiang773-hkust}"
export FTP1_NUM_TRAIN_STEPS="${FTP1_NUM_TRAIN_STEPS:-200000}"
export FTP1_LOCAL_BATCH_SIZE="${FTP1_LOCAL_BATCH_SIZE:-8}"
export FTP1_VAL_INTERVAL="${FTP1_VAL_INTERVAL:-10000}"
export FTP1_SAVE_INTERVAL="${FTP1_SAVE_INTERVAL:-10000}"
export FTP1_EXP_NAME="${FTP1_EXP_NAME:-Spark0_real_bench_v5_Moxian_ee_H20}"

exec bash XPolicyLab/policy/FTP_1/train.sh \
  Spark0_real_bench_v5 Moxian tianji_marvin_wuji ee 42 0,1,2,3,4,5,6,7
