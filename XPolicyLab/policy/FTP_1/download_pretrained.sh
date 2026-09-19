#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "${POLICY_DIR}/../../.." && pwd)"
FTP1_PYTHON="${FTP1_PYTHON:-/mnt/xspark-data/conda_envs/FTP_1/bin/python}"
PRETRAIN_ROOT="${FTP1_PRETRAIN_ROOT:-${BENCH_ROOT}/pretrain_model}"
REPO_ID="${FTP1_PRETRAIN_REPO_ID:-MJJJJ1064/ftp1_v0426_50kstep}"
REVISION="${FTP1_PRETRAIN_REVISION:-d6e5b73e473d3e70fb5f53132a2b1c35b5031156}"
MODEL_NAME="${FTP1_PRETRAIN_MODEL_NAME:-ftp1_pretrain_v0426_50kstep}"
FINAL_DIR="${PRETRAIN_ROOT}/${MODEL_NAME}"
STAGING_DIR="${PRETRAIN_ROOT}/.${MODEL_NAME}.download"
MAX_WORKERS="${FTP1_DOWNLOAD_WORKERS:-4}"
EXPECTED_FILE_COUNT=96
EXPECTED_TOTAL_BYTES=9842660591
EXPECTED_LFS_COUNT=31
EXPECTED_MODEL_BYTES=7904019864
EXPECTED_MODEL_SHA256=c787f637368b85898e8aac957264c8e03313fc68a64256a3b3bf549e3600d9fb

[[ -x "${FTP1_PYTHON}" ]] || {
    echo "[FTP_1][ERROR] Python environment not found: ${FTP1_PYTHON}" >&2
    exit 1
}
mkdir -p "${PRETRAIN_ROOT}"
exec 9>"${PRETRAIN_ROOT}/.${MODEL_NAME}.lock"
if ! flock -n 9; then
    echo "[FTP_1][ERROR] another checkpoint download/validation holds the lock" >&2
    exit 1
fi

if [[ -e "${FINAL_DIR}" || -L "${FINAL_DIR}" ]]; then
    echo "[FTP_1] pretrained checkpoint already exists: ${FINAL_DIR}"
    "${FTP1_PYTHON}" - \
        "${FINAL_DIR}" "${REPO_ID}" "${REVISION}" \
        "${EXPECTED_FILE_COUNT}" "${EXPECTED_TOTAL_BYTES}" "${EXPECTED_LFS_COUNT}" \
        "${EXPECTED_MODEL_BYTES}" "${EXPECTED_MODEL_SHA256}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
repo_id, revision = sys.argv[2:4]
expected_file_count = int(sys.argv[4])
expected_total_bytes = int(sys.argv[5])
expected_lfs_count = int(sys.argv[6])
expected_model_bytes = int(sys.argv[7])
expected_model_sha256 = sys.argv[8]
manifest_path = root / "download_manifest.json"
required_files = [
    root / "model.safetensors",
    root / "model_config.json",
    root / "train_config.json",
    root / "tactile_input_config_file.json",
]
required_dirs = [
    root / "hpt_tokenizer",
    root / "normalization",
]
missing = [str(path) for path in required_files if not path.is_file()]
missing += [str(path) for path in required_dirs if not path.is_dir()]
if missing:
    raise SystemExit(f"incomplete pretrained checkpoint; missing: {missing}")
if not manifest_path.is_file():
    raise SystemExit(f"download manifest is missing: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected_manifest = {
    "repo_id": repo_id,
    "revision": revision,
    "resolved_revision": revision,
    "official_file_count": expected_file_count,
    "official_total_bytes": expected_total_bytes,
    "lfs_file_count": expected_lfs_count,
    "lfs_sha256_verified": True,
}
bad = {
    key: (manifest.get(key), expected)
    for key, expected in expected_manifest.items()
    if manifest.get(key) != expected
}
if bad:
    raise SystemExit(f"download manifest does not match the pinned snapshot: {bad}")
model = root / "model.safetensors"
if model.stat().st_size != expected_model_bytes:
    raise SystemExit(
        f"model.safetensors size mismatch: {model.stat().st_size} != {expected_model_bytes}"
    )
digest = hashlib.sha256()
with model.open("rb") as stream:
    while chunk := stream.read(16 * 1024 * 1024):
        digest.update(chunk)
if digest.hexdigest() != expected_model_sha256:
    raise SystemExit("model.safetensors SHA256 mismatch")
print(f"[FTP_1] checkpoint structure verified: {root}")
PY
    exit 0
fi

mkdir -p "${STAGING_DIR}"
unset HF_HUB_DISABLE_XET
export HF_XET_HIGH_PERFORMANCE="${FTP1_XET_HIGH_PERFORMANCE:-1}"

"${FTP1_PYTHON}" - \
    "${REPO_ID}" "${REVISION}" "${STAGING_DIR}" "${MAX_WORKERS}" \
    "${EXPECTED_FILE_COUNT}" "${EXPECTED_TOTAL_BYTES}" "${EXPECTED_LFS_COUNT}" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

from huggingface_hub import HfApi, snapshot_download

(
    repo_id,
    revision,
    staging_arg,
    max_workers_arg,
    expected_file_count_arg,
    expected_total_bytes_arg,
    expected_lfs_count_arg,
) = sys.argv[1:]
staging = Path(staging_arg)
max_workers = int(max_workers_arg)
expected_file_count = int(expected_file_count_arg)
expected_total_bytes = int(expected_total_bytes_arg)
expected_lfs_count = int(expected_lfs_count_arg)

print(f"[FTP_1] downloading {repo_id}@{revision}", flush=True)
snapshot_download(
    repo_id=repo_id,
    revision=revision,
    local_dir=staging,
    max_workers=max_workers,
    resume_download=True,
)

info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
if info.sha != revision:
    raise RuntimeError(f"resolved revision {info.sha} does not match pinned {revision}")

missing: list[str] = []
size_errors: list[str] = []
lfs_files: list[tuple[Path, str]] = []
total_bytes = 0
for sibling in info.siblings:
    path = staging / sibling.rfilename
    if sibling.size is None:
        raise RuntimeError(f"missing size metadata for {sibling.rfilename}")
    expected_size = int(sibling.size)
    total_bytes += expected_size
    if not path.is_file():
        missing.append(sibling.rfilename)
        continue
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        size_errors.append(f"{sibling.rfilename}: {actual_size} != {expected_size}")
    lfs = getattr(sibling, "lfs", None)
    sha256 = getattr(lfs, "sha256", None) if lfs is not None else None
    if sha256:
        lfs_files.append((path, str(sha256)))

if missing or size_errors:
    raise RuntimeError(f"snapshot validation failed; missing={missing}, size_errors={size_errors}")
if len(info.siblings) != expected_file_count:
    raise RuntimeError(
        f"official file count changed: {len(info.siblings)} != {expected_file_count}"
    )
if total_bytes != expected_total_bytes:
    raise RuntimeError(f"official total size changed: {total_bytes} != {expected_total_bytes}")
if len(lfs_files) != expected_lfs_count:
    raise RuntimeError(f"official LFS file count changed: {len(lfs_files)} != {expected_lfs_count}")

if os.environ.get("FTP1_VERIFY_SHA256", "1") == "1":
    for index, (path, expected_sha) in enumerate(lfs_files, start=1):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(16 * 1024 * 1024):
                digest.update(chunk)
        actual_sha = digest.hexdigest()
        if actual_sha != expected_sha:
            raise RuntimeError(f"SHA256 mismatch: {path.name}: {actual_sha} != {expected_sha}")
        print(f"[FTP_1] SHA256 {index}/{len(lfs_files)} OK: {path.relative_to(staging)}", flush=True)

required = [
    staging / "model.safetensors",
    staging / "model_config.json",
    staging / "train_config.json",
    staging / "tactile_input_config_file.json",
    staging / "hpt_tokenizer",
    staging / "normalization",
]
missing_required = [str(path) for path in required if not path.exists()]
if missing_required:
    raise RuntimeError(f"required checkpoint entries missing: {missing_required}")

manifest = {
    "repo_id": repo_id,
    "revision": revision,
    "resolved_revision": info.sha,
    "downloaded_at": datetime.now(timezone.utc).isoformat(),
    "official_file_count": len(info.siblings),
    "official_total_bytes": total_bytes,
    "lfs_sha256_verified": os.environ.get("FTP1_VERIFY_SHA256", "1") == "1",
    "lfs_file_count": len(lfs_files),
}
(staging / "download_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
PY

if [[ -e "${FINAL_DIR}" || -L "${FINAL_DIR}" ]]; then
    echo "[FTP_1][ERROR] final checkpoint path appeared during download: ${FINAL_DIR}" >&2
    exit 1
fi
mv -T -- "${STAGING_DIR}" "${FINAL_DIR}"
echo "[FTP_1] pretrained checkpoint ready: ${FINAL_DIR}"
