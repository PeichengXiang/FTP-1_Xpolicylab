#!/usr/bin/env python3
"""Batch-convert curated row-aligned 30 Hz Spark HDF5 into FTP-1 Zarr.

Each source episode becomes one atomic ``.zarr`` directory directly below the
output root.  A completed Zarr that already satisfies the current independent-
action contract is never overwritten during resume.  Outputs that still treat
next-state as action are stale and replaced.  Workers write into ``.staging``
first, validate the FTP-1 contract, and only then rename the directory into
place.  The command is therefore safe to stop and rerun.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np
import zarr


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from parse_data_spark_moxian import (  # noqa: E402
    FINGERTIP_AREAS,
    PALM_AREAS,
    SENSOR_NAME,
    TACTILE_TYPE,
    WUJIHAND2_FAAS_INDEX,
    convert_episode,
)


CONVERTER_PATH = SCRIPT_DIR / "parse_data_spark_moxian.py"
CONVERTER_SHA256 = hashlib.sha256(CONVERTER_PATH.read_bytes()).hexdigest()
# Existing production Zarrs were generated before the decoder call was routed
# through XPolicyLab's mandatory decode_image_bit helper.  That change is
# pixel-identical for this RGB contract and does not alter any output tensor.
PIXEL_EQUIVALENT_CONVERTER_SHA256 = frozenset(
    {
        CONVERTER_SHA256,
        "48b6d8c09ff686ec74b821d9afa3cb85bf72cbdbbc374c43ee39f2c97273ac30",
    }
)


DEFAULT_SOURCE_ROOT = Path(
    "/personal/xspark_shared/hand_data/hdf5/spark0_real/bench_v5"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/personal/xiangpc/0814_Xpolicylab_bench/FTP-1/data/"
    "spark0_real_bench_v5_moxian_joint_only"
)

TACTILE_SPECS = {
    "fingertip": (5, 4, 4, FINGERTIP_AREAS),
    "palm": (1, 15, 16, PALM_AREAS),
}

POSE_DATA_KEYS = {
    "camera_ego_pose",
    "left_wrist_pose",
    "right_wrist_pose",
}
JOINT_ONLY_FORBIDDEN_KEYS = {
    "camera_ego_pose",
    "left_arm_joints",
    "right_arm_joints",
    "left_arm_joints_action",
    "right_arm_joints_action",
    "supplementary_joints",
}

REQUIRED_INDEPENDENT_ACTION_KEYS = {
    "left_wrist_pose_action",
    "right_wrist_pose_action",
    "left_hand_joints_action",
    "right_hand_joints_action",
}

REQUIRED_BASE_KEYS = {
    "camera_ego_rgb",
    "left_wrist_camera_rgb",
    "right_wrist_camera_rgb",
    "left_hand_joints",
    "right_hand_joints",
    "left_hand_joints_idx",
    "right_hand_joints_idx",
    "sub_task_instruction",
    *REQUIRED_INDEPENDENT_ACTION_KEYS,
}


# Keep this allowlist deliberately narrow.  In particular, ENOENT and ordinary
# KeyError/ValueError failures are not retryable: a missing file, dataset, or
# malformed episode must remain a visible conversion failure.
RETRYABLE_SOURCE_IO_ERRNOS = frozenset(
    value
    for value in (
        getattr(errno, "EIO", None),
        getattr(errno, "ESTALE", None),
        getattr(errno, "ETIMEDOUT", None),
        getattr(errno, "EAGAIN", None),
        getattr(errno, "ECONNRESET", None),
    )
    if value is not None
)
RETRYABLE_SOURCE_IO_MARKERS = (
    "input/output error",
    "nosuchkey",
    "no such key",
)


@dataclass(frozen=True)
class Job:
    source: str
    relative_source: str
    output: str
    task: str
    episode_id: str
    source_size_bytes: int
    source_mtime_ns: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not cleaned:
        raise ValueError(f"cannot construct output name from {value!r}")
    return cleaned


def output_name(source: Path, source_root: Path) -> tuple[str, str, str]:
    relative = source.relative_to(source_root)
    if len(relative.parts) != 2:
        raise ValueError(
            "production source must use the curated layout "
            f"<task>/episode_<digits>.hdf5, got {relative}"
        )
    task = _slug(relative.parts[0])
    match = re.fullmatch(r"episode_(\d+)", source.stem)
    if match is None:
        raise ValueError(
            "standardized episode name must be episode_<digits>.hdf5, got "
            f"{relative}"
        )
    episode_id = match.group(1)
    return f"{task}__episode_{episode_id}.zarr", task, episode_id


def discover_jobs(source_root: Path, output_root: Path) -> list[Job]:
    sources = sorted(p for p in source_root.rglob("*.hdf5") if not p.name.startswith(".") and not p.name.endswith(".partial.hdf5"))
    if not sources:
        raise FileNotFoundError(f"no HDF5 episodes below {source_root}")
    jobs: list[Job] = []
    seen_outputs: set[str] = set()
    for source in sources:
        name, task, episode_id = output_name(source, source_root)
        if name in seen_outputs:
            raise ValueError(f"duplicate output name: {name}")
        seen_outputs.add(name)
        source_stat = source.stat()
        jobs.append(
            Job(
                source=str(source),
                relative_source=str(source.relative_to(source_root)),
                output=str(output_root / name),
                task=task,
                episode_id=episode_id,
                source_size_bytes=int(source_stat.st_size),
                source_mtime_ns=int(source_stat.st_mtime_ns),
            )
        )
    return jobs


def _array_keys(group: Any) -> set[str]:
    return {str(key) for key in group.array_keys()}


def _assert_finite(array: Any, name: str, chunk_rows: int = 512) -> None:
    for start in range(0, int(array.shape[0]), chunk_rows):
        values = np.asarray(array[start : start + chunk_rows])
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values near row {start}")


def validate_episode_zarr(
    path: Path, *, expected_joint_only_120d: bool | None = None
) -> dict[str, Any]:
    root = zarr.open_group(str(path), mode="r")
    if "data" not in root or "meta" not in root:
        raise ValueError("missing /data or /meta")
    data = root["data"]
    joint_only_120d = bool(root.attrs.get("joint_only_120d", False))
    if (
        expected_joint_only_120d is not None
        and joint_only_120d is not expected_joint_only_120d
    ):
        raise ValueError(
            "joint-only contract mismatch: "
            f"expected={expected_joint_only_120d}, actual={joint_only_120d}"
        )
    ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    if ends.ndim != 1 or len(ends) != 1 or int(ends[-1]) <= 0:
        raise ValueError(f"invalid episode endpoints: {ends}")
    frame_count = int(ends[-1])
    if root.attrs.get("input_mode") != "standardized_30hz":
        raise ValueError("output was not made from the standardized 30 Hz input")
    if root.attrs.get("resampling_performed") is not False:
        raise ValueError("output metadata does not forbid resampling")
    if int(root.attrs.get("input_frames", -1)) != frame_count:
        raise ValueError("input_frames must equal the output row count")
    if int(root.attrs.get("output_frames", -1)) != frame_count:
        raise ValueError("output_frames must equal the output row count")
    if int(root.attrs.get("output_episode_count", -1)) != 1:
        raise ValueError("one source HDF5 must produce exactly one Zarr episode")
    standardized_source = dict(root.attrs.get("standardized_source", {}))
    if standardized_source.get("auxiliary_60hz_datasets_read") is not False:
        raise ValueError("metadata does not prove auxiliary 60 Hz datasets were ignored")
    for mapping in (
        "state_row_mapping",
        "action_row_mapping",
        "rgb_row_mapping",
        "tactile_row_mapping",
    ):
        if standardized_source.get(mapping) != "identity":
            raise ValueError(f"standardized source {mapping} is not identity")
    if root.attrs.get("action_source") != "hdf5_action":
        raise ValueError("output action_source must be hdf5_action")
    tactile_preprocessing = dict(root.attrs.get("tactile_pressure_preprocessing", {}))
    if tactile_preprocessing.get("formula") != "direct_copy_of_pressure_component":
        raise ValueError("tactile pressure was not marked as a direct copy")
    if tactile_preprocessing.get("baseline_subtraction_in_ftp1_converter") is not False:
        raise ValueError("FTP-1 converter must not subtract a tactile baseline")
    camera_metadata = dict(root.attrs.get("camera_ego_pose", {}))
    if camera_metadata.get("source_rows_verified_identical") is not True:
        raise ValueError("camera_ego_pose was not verified as one fixed extrinsic")
    if camera_metadata.get("interpolation_performed") is not False:
        raise ValueError("camera_ego_pose must not be interpolated")
    keys = _array_keys(data)
    required = set(REQUIRED_BASE_KEYS)
    required.update({"left_wrist_pose", "right_wrist_pose"})
    required.update(REQUIRED_INDEPENDENT_ACTION_KEYS)
    if not joint_only_120d:
        required.update(
            {
                "camera_ego_pose",
                "left_arm_joints",
                "right_arm_joints",
                "left_arm_joints_action",
                "right_arm_joints_action",
            }
        )
    for side in ("left", "right"):
        for group_name in TACTILE_SPECS:
            required.update(
                {
                    f"{side}_tactile_data_{group_name}",
                    f"{side}_tactile_area_{group_name}",
                    f"{side}_tactile_sensor_{group_name}",
                    f"{side}_tactile_type_{group_name}",
                }
            )
    missing = sorted(required - keys)
    if missing:
        raise ValueError(f"missing data keys: {missing}")
    unexpected_joint_only_keys = (
        sorted(JOINT_ONLY_FORBIDDEN_KEYS & keys) if joint_only_120d else []
    )
    if unexpected_joint_only_keys:
        raise ValueError(
            "joint-only Zarr contains arrays that would activate non-arm/hand slots: "
            f"{unexpected_joint_only_keys}"
        )
    bad_lengths = {
        key: int(data[key].shape[0])
        for key in keys
        if not data[key].shape or int(data[key].shape[0]) != frame_count
    }
    if bad_lengths:
        raise ValueError(f"time-axis mismatch: {bad_lengths}")

    source_meta = dict(root.attrs.get("standardized_source", {}))
    source_fps = float(source_meta.get("frequency_hz") or root.attrs.get("target_fps_hz") or 0.0)
    if source_fps <= 0:
        raise ValueError("output metadata is missing a positive source frequency")
    duration_seconds = float(max(frame_count - 1, 0) / source_fps)

    for key in ("camera_ego_rgb", "left_wrist_camera_rgb", "right_wrist_camera_rgb"):
        array = data[key]
        if tuple(array.shape[1:]) != (224, 224, 3) or np.dtype(array.dtype) != np.dtype(np.uint8):
            raise ValueError(f"{key} has invalid shape/dtype: {array.shape}, {array.dtype}")

    for key in (
        "left_wrist_pose",
        "right_wrist_pose",
        "left_wrist_pose_action",
        "right_wrist_pose_action",
    ):
        if tuple(data[key].shape[1:]) != (6,):
            raise ValueError(f"{key} must be (T, 6), got {data[key].shape}")
        _assert_finite(data[key], key)
    if not joint_only_120d:
        camera_ego_pose = np.asarray(data["camera_ego_pose"][:], dtype=np.float32)
        if not np.array_equal(
            camera_ego_pose, np.repeat(camera_ego_pose[:1], frame_count, axis=0)
        ):
            raise ValueError("camera_ego_pose must repeat one fixed camera extrinsic")

    expected_idx = WUJIHAND2_FAAS_INDEX.astype(np.int32)
    for side in ("left", "right"):
        arm = data.get(f"{side}_arm_joints") if not joint_only_120d else None
        arm_action = data.get(f"{side}_arm_joints_action") if not joint_only_120d else None
        hand = data[f"{side}_hand_joints"]
        hand_action = data[f"{side}_hand_joints_action"]
        if arm is not None and (tuple(arm.shape) != (frame_count, 7) or np.dtype(arm.dtype).kind != "f"):
            raise ValueError(
                f"{side}_arm_joints must be float (T,7), got {arm.shape}, {arm.dtype}"
            )
        if arm_action is not None and (
            tuple(arm_action.shape) != (frame_count, 7) or np.dtype(arm_action.dtype).kind != "f"
        ):
            raise ValueError(
                f"{side}_arm_joints_action must be float (T,7), got "
                f"{arm_action.shape}, {arm_action.dtype}"
            )
        if tuple(hand.shape) != (frame_count, 20) or np.dtype(hand.dtype).kind != "f":
            raise ValueError(
                f"{side}_hand_joints must be float (T,20), got {hand.shape}, {hand.dtype}"
            )
        if tuple(hand_action.shape) != (frame_count, 20) or np.dtype(hand_action.dtype).kind != "f":
            raise ValueError(
                f"{side}_hand_joints_action must be float (T,20), got "
                f"{hand_action.shape}, {hand_action.dtype}"
            )
        if arm is not None:
            _assert_finite(arm, f"{side}_arm_joints")
        if arm_action is not None:
            _assert_finite(arm_action, f"{side}_arm_joints_action")
        _assert_finite(hand, f"{side}_hand_joints")
        _assert_finite(hand_action, f"{side}_hand_joints_action")

        index_array = data[f"{side}_hand_joints_idx"]
        if tuple(index_array.shape) != (frame_count, 20) or np.dtype(index_array.dtype).kind not in "iu":
            raise ValueError(
                f"{side}_hand_joints_idx must be integer (T,20), "
                f"got {index_array.shape}, {index_array.dtype}"
            )
        idx = np.asarray(index_array[0], dtype=np.int32)
        if not np.array_equal(idx, expected_idx):
            raise ValueError(f"{side} hand FAAS index mismatch")
        if len(np.unique(idx)) != len(idx) or np.any((idx < 0) | (idx >= 32)):
            raise ValueError(f"{side} hand FAAS indices are invalid")
        for start in range(0, frame_count, 512):
            index_rows = np.asarray(index_array[start : start + 512], dtype=np.int32)
            if not np.all(index_rows == expected_idx[None, :]):
                raise ValueError(
                    f"{side} hand FAAS indices change near row {start}"
                )
        for group_name, (area_count, height, width, areas) in TACTILE_SPECS.items():
            prefix = f"{side}_tactile"
            tactile = data[f"{prefix}_data_{group_name}"]
            if tuple(tactile.shape[1:]) != (area_count, height, width):
                raise ValueError(
                    f"{prefix}_data_{group_name} has invalid shape {tactile.shape}"
                )
            if np.dtype(tactile.dtype) != np.dtype(np.float32):
                raise ValueError(
                    f"{prefix}_data_{group_name} must preserve float32 pressure"
                )
            _assert_finite(tactile, f"{prefix}_data_{group_name}")
            for start in range(0, frame_count, 512):
                values = np.asarray(tactile[start : start + 512])
                if np.any(values < 0):
                    raise ValueError(f"{prefix}_data_{group_name} contains negative pressure")
            actual_areas = np.asarray(data[f"{prefix}_area_{group_name}"][0], dtype=np.int32)
            if not np.array_equal(actual_areas, areas):
                raise ValueError(f"{prefix}_area_{group_name} mismatch: {actual_areas}")
            sensor = str(np.asarray(data[f"{prefix}_sensor_{group_name}"][0]).item())
            tactile_type = str(np.asarray(data[f"{prefix}_type_{group_name}"][0]).item())
            if sensor != SENSOR_NAME or tactile_type != TACTILE_TYPE:
                raise ValueError(
                    f"{prefix}_{group_name} identity mismatch: {sensor!r}, {tactile_type!r}"
                )

    instruction = str(np.asarray(data["sub_task_instruction"][0]).item())
    return {
        "frames": frame_count,
        "episodes": int(len(ends)),
        "duration_seconds": duration_seconds,
        "instruction": instruction,
        "data_key_count": len(keys),
        "joint_only_120d": joint_only_120d,
        "ftp1_width": 120,
        "active_action_dimensions_per_step": 58 if joint_only_120d else 72,
    }


def _write_metadata(metadata_dir: Path, output_stem: str, metadata: dict[str, Any]) -> Path:
    path = metadata_dir / f"{output_stem}.json"
    _atomic_json(path, metadata)
    return path


def _exception_chain(exc: BaseException) -> Iterable[BaseException]:
    """Yield an exception and its explicit/implicit causes without looping."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_retryable_source_io_error(exc: BaseException) -> bool:
    """Recognize only transient storage errors observed while reading HDF5.

    ``NoSuchKey`` may be exposed as an SDK exception, a structured ClientError,
    or only as text embedded by the filesystem/HDF5 layer.  It is retryable but
    never ignored: the retry loop below is finite and re-raises the final error.
    """
    for current in _exception_chain(exc):
        if isinstance(current, OSError) and current.errno in RETRYABLE_SOURCE_IO_ERRNOS:
            return True

        response = getattr(current, "response", None)
        if isinstance(response, dict):
            error = response.get("Error")
            if isinstance(error, dict) and str(error.get("Code", "")).lower() == "nosuchkey":
                return True

        if type(current).__name__.lower() == "nosuchkey":
            return True
        message = str(current).lower()
        if any(marker in message for marker in RETRYABLE_SOURCE_IO_MARKERS):
            return True
    return False


def _annotate_retry_failure(
    exc: BaseException, attempts: int, attempt_errors: list[dict[str, Any]]
) -> None:
    """Attach JSON-safe retry provenance; BaseException attributes survive ProcessPool pickling."""
    try:
        exc.source_io_attempts = attempts  # type: ignore[attr-defined]
        exc.source_io_attempt_errors = attempt_errors  # type: ignore[attr-defined]
    except Exception:
        # Some third-party exception implementations may reject attributes.  The
        # original failure still propagates and remains fatal for this episode.
        pass


def _run_with_source_io_retries(
    operation: Callable[[], Any],
    *,
    retries: int,
    base_delay_seconds: float,
    before_retry: Callable[[], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Any, int, list[dict[str, Any]]]:
    """Run an HDF5 conversion operation with bounded transient-I/O retries."""
    if retries < 0:
        raise ValueError("source I/O retries must be non-negative")
    if base_delay_seconds < 0:
        raise ValueError("source I/O retry delay must be non-negative")

    attempt_errors: list[dict[str, Any]] = []
    for attempt in range(1, retries + 2):
        try:
            return operation(), attempt, attempt_errors
        except BaseException as exc:
            retryable = _is_retryable_source_io_error(exc)
            will_retry = retryable and attempt <= retries
            event: dict[str, Any] = {
                "attempt": attempt,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "retryable": retryable,
                "will_retry": will_retry,
            }
            if will_retry:
                delay = base_delay_seconds * (2 ** (attempt - 1))
                event["delay_seconds"] = delay
            attempt_errors.append(event)

            if not will_retry:
                _annotate_retry_failure(exc, attempt, attempt_errors)
                raise
            if before_retry is not None:
                before_retry()
            sleep(delay)

    raise AssertionError("unreachable retry loop")


def _zarr_has_independent_hdf5_action(path: Path) -> bool:
    """Return True only for Zarrs that already store HDF5 ``action/*`` separately."""
    try:
        root = zarr.open_group(str(path), mode="r")
    except Exception:
        return False
    if root.attrs.get("action_source") != "hdf5_action":
        return False
    if "data" not in root:
        return False
    keys = _array_keys(root["data"])
    return REQUIRED_INDEPENDENT_ACTION_KEYS.issubset(keys)


def _convert_job(
    job: Job,
    staging_root: str,
    metadata_root: str,
    target_fps: float,
    image_size: int,
    min_segment_frames: int,
    source_io_retries: int,
    source_io_retry_delay_seconds: float,
    joint_only_120d: bool = False,
) -> dict[str, Any]:
    cv2.setNumThreads(1)
    started = time.monotonic()
    source = Path(job.source)
    final = Path(job.output)
    if final.exists() and not _zarr_has_independent_hdf5_action(final):
        shutil.rmtree(final)
    if final.exists():
        validation = validate_episode_zarr(
            final, expected_joint_only_120d=joint_only_120d
        )
        existing_root = zarr.open_group(str(final), mode="r")
        if existing_root.attrs.get("converter_sha256") not in PIXEL_EQUIVALENT_CONVERTER_SHA256:
            raise ValueError(
                f"existing output was made by a different converter version: {final}"
            )
        if int(existing_root.attrs.get("source_size_bytes", -1)) != job.source_size_bytes:
            raise ValueError(f"existing output source size no longer matches: {final}")
        if int(existing_root.attrs.get("source_mtime_ns", -1)) != job.source_mtime_ns:
            raise ValueError(f"existing output source mtime no longer matches: {final}")
        metadata_path = Path(metadata_root) / f"{final.stem}.json"
        if not metadata_path.is_file():
            _write_metadata(Path(metadata_root), final.stem, dict(existing_root.attrs))
        return {
            "status": "existing_valid",
            "source": job.source,
            "relative_source": job.relative_source,
            "output": job.output,
            "task": job.task,
            "episode_id": job.episode_id,
            **validation,
            "source_io_attempts": 0,
            "source_io_attempt_errors": [],
            "elapsed_seconds": time.monotonic() - started,
            "finished_at": _utc_now(),
        }

    staging = Path(staging_root) / (
        f".{final.name}.partial-{os.getpid()}-{uuid.uuid4().hex}.zarr"
    )
    try:
        def convert_once() -> dict[str, Any]:
            return convert_episode(
                source,
                staging,
                target_fps=target_fps,
                image_size=image_size,
                min_segment_frames=min_segment_frames,
                save_preview=False,
                write_sidecar=False,
                joint_only_120d=joint_only_120d,
            )

        def remove_partial_staging() -> None:
            if staging.exists():
                shutil.rmtree(staging)

        metadata, source_io_attempts, source_io_attempt_errors = _run_with_source_io_retries(
            convert_once,
            retries=source_io_retries,
            base_delay_seconds=source_io_retry_delay_seconds,
            before_retry=remove_partial_staging,
        )
        validation = validate_episode_zarr(
            staging, expected_joint_only_120d=joint_only_120d
        )
        if final.exists():
            raise FileExistsError(f"final output appeared during conversion: {final}")

        metadata["output_zarr"] = str(final)
        metadata["relative_source"] = job.relative_source
        metadata["task"] = job.task
        metadata["episode_id"] = job.episode_id
        metadata["converter_sha256"] = CONVERTER_SHA256
        metadata["source_size_bytes"] = job.source_size_bytes
        metadata["source_mtime_ns"] = job.source_mtime_ns
        metadata["source_io_attempts"] = source_io_attempts
        metadata["validation"] = validation
        metadata["converted_at"] = _utc_now()
        writable_root = zarr.open_group(str(staging), mode="a")
        writable_root.attrs.update(metadata)
        os.rename(staging, final)
        metadata_path = _write_metadata(Path(metadata_root), final.stem, metadata)
        return {
            "status": "converted",
            "source": job.source,
            "relative_source": job.relative_source,
            "output": job.output,
            "metadata_json": str(metadata_path),
            "task": job.task,
            "episode_id": job.episode_id,
            **validation,
            "source_io_attempts": source_io_attempts,
            "source_io_attempt_errors": source_io_attempt_errors,
            "elapsed_seconds": time.monotonic() - started,
            "finished_at": _utc_now(),
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _failed_result(job: Job, exc: BaseException) -> dict[str, Any]:
    return {
        "status": "failed",
        "source": job.source,
        "relative_source": job.relative_source,
        "output": job.output,
        "task": job.task,
        "episode_id": job.episode_id,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "source_io_attempts": int(getattr(exc, "source_io_attempts", 1)),
        "source_io_attempt_errors": list(
            getattr(exc, "source_io_attempt_errors", [])
        ),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        "finished_at": _utc_now(),
    }


def _select_jobs(
    jobs: Iterable[Job], tasks: set[str], limit: int | None
) -> list[Job]:
    selected = [job for job in jobs if not tasks or job.task in tasks]
    if limit is not None:
        selected = selected[:limit]
    return selected


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--min-segment-frames", type=int, default=32)
    parser.add_argument(
        "--joint-only-120d",
        action="store_true",
        required=True,
        help=(
            "keep FTP-1 width 120 but omit all pose arrays so only the 54 "
            "arm/hand-joint dimensions are supervised"
        ),
    )
    parser.add_argument(
        "--source-io-retries",
        type=int,
        default=2,
        help="Retries per episode for transient HDF5 EIO/NoSuchKey failures (default: 2)",
    )
    parser.add_argument(
        "--source-io-retry-delay-seconds",
        type=float,
        default=2.0,
        help="Initial retry delay; subsequent delays use exponential backoff (default: 2)",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if not any(np.isclose(args.target_fps, fps, atol=1e-9, rtol=0.0) for fps in (15.0, 30.0)):
        raise ValueError(
            "row-aligned source must not be resampled; --target-fps must be 15 or 30"
        )
    if not 0 <= args.source_io_retries <= 10:
        raise ValueError("--source-io-retries must be between 0 and 10")
    if args.source_io_retry_delay_seconds < 0:
        raise ValueError("--source-io-retry-delay-seconds must be non-negative")
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lock_stream = (output_root / ".conversion.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another conversion process holds {output_root}/.conversion.lock") from exc
    lock_stream.write(f"pid={os.getpid()} started={_utc_now()}\n")
    lock_stream.flush()
    staging_root = output_root / ".staging"
    metadata_root = output_root / "metadata"
    staging_root.mkdir(exist_ok=True)
    metadata_root.mkdir(exist_ok=True)

    all_jobs = discover_jobs(source_root, output_root)
    tasks = {_slug(task) for task in args.task}
    jobs = _select_jobs(all_jobs, tasks, args.limit)
    if not jobs:
        raise ValueError("no jobs selected")
    inventory = {
        "created_at": _utc_now(),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "discovered_episode_count": len(all_jobs),
        "selected_episode_count": len(jobs),
        "conversion_contract": "standardized_30hz_row_identity",
        "target_fps": args.target_fps,
        "image_size": args.image_size,
        "min_segment_frames": args.min_segment_frames,
        "source_io_retries": args.source_io_retries,
        "source_io_retry_delay_seconds": args.source_io_retry_delay_seconds,
        "converter_sha256": CONVERTER_SHA256,
        "joint_only_120d": args.joint_only_120d,
        "ftp1_width": 120,
        "active_action_dimensions_per_step": 58 if args.joint_only_120d else 72,
        "jobs": [job.__dict__ for job in jobs],
    }
    _atomic_json(output_root / "source_manifest.json", inventory)

    log_path = output_root / "conversion_log.jsonl"
    counters = {"converted": 0, "existing_valid": 0, "failed": 0}
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_job = {
                executor.submit(
                    _convert_job,
                    job,
                    str(staging_root),
                    str(metadata_root),
                    args.target_fps,
                    args.image_size,
                    args.min_segment_frames,
                    args.source_io_retries,
                    args.source_io_retry_delay_seconds,
                    args.joint_only_120d,
                ): job
                for job in jobs
            }
            for completed, future in enumerate(
                concurrent.futures.as_completed(future_to_job), start=1
            ):
                job = future_to_job[future]
                try:
                    result = future.result()
                except BaseException as exc:  # preserve every per-episode failure
                    result = _failed_result(job, exc)
                counters[result["status"]] += 1
                line = json.dumps(result, ensure_ascii=False, sort_keys=True)
                log_file.write(line + "\n")
                log_file.flush()
                os.fsync(log_file.fileno())
                print(
                    json.dumps(
                        {
                            "progress": f"{completed}/{len(jobs)}",
                            "status": result["status"],
                            "relative_source": job.relative_source,
                            "frames": result.get("frames"),
                            "source_io_attempts": result.get("source_io_attempts", 1),
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                            "counts": counters,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    summary = {
        "finished_at": _utc_now(),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "selected_episode_count": len(jobs),
        "counts": counters,
        "elapsed_seconds": time.monotonic() - started,
        "conversion_log": str(log_path),
        "joint_only_120d": args.joint_only_120d,
        "ftp1_width": 120,
        "active_action_dimensions_per_step": 58 if args.joint_only_120d else 72,
    }
    _atomic_json(output_root / "conversion_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if counters["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
