#!/usr/bin/env bash
set -euo pipefail
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "${POLICY_DIR}/../../.." && pwd)"
PYTHON="${FTP1_PYTHON:-${BENCH_ROOT}/.venvs/FTP_1/bin/python}"
SOURCE_ROOT="${FTP1_SOURCE_ROOT:-/vepfs-cnbje63de6fae220/xiangpc/data/bench_v5}"
OUTPUT_ROOT="${FTP1_OUTPUT_ROOT:-${BENCH_ROOT}/data/spark0_real_bench_v5_moxian_joint_only}"
WORKERS="${FTP1_CONVERT_WORKERS:-8}"
CONVERTER="${POLICY_DIR}/data_scripts/convert_spark_moxian_dataset.py"
[[ -f "${CONVERTER}" ]] || { echo "[FTP_1][ERROR] converter missing: ${CONVERTER}" >&2; exit 1; }
if [[ "${FTP1_DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${PYTHON}" "${CONVERTER}" --source-root "${SOURCE_ROOT}" --output-root "${OUTPUT_ROOT}" --workers "${WORKERS}" --joint-only-120d
  printf '\n'; exit 0
fi
exec "${PYTHON}" "${CONVERTER}" --source-root "${SOURCE_ROOT}" --output-root "${OUTPUT_ROOT}" --workers "${WORKERS}" --joint-only-120d "$@"
