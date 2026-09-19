#!/usr/bin/env python3
"""Validate a complete Spark-Moxian FTP-1 conversion and write a QC report."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import zarr

from convert_spark_moxian_dataset import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SOURCE_ROOT,
    Job,
    PIXEL_EQUIVALENT_CONVERTER_SHA256,
    discover_jobs,
    validate_episode_actions_against_source,
    validate_episode_zarr,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.tmp-", text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _validate(job: Job, joint_only_120d: bool) -> dict[str, Any]:
    output = Path(job.output)
    result = validate_episode_zarr(
        output, expected_joint_only_120d=joint_only_120d
    )
    root = zarr.open_group(str(output), mode="r")
    if root.attrs.get("converter_sha256") not in PIXEL_EQUIVALENT_CONVERTER_SHA256:
        raise ValueError("converter SHA256 does not match the installed converter")
    if int(root.attrs.get("source_size_bytes", -1)) != job.source_size_bytes:
        raise ValueError("source size fingerprint does not match")
    if int(root.attrs.get("source_mtime_ns", -1)) != job.source_mtime_ns:
        raise ValueError("source mtime fingerprint does not match")
    action_audit = validate_episode_actions_against_source(
        output,
        Path(job.source),
        expected_joint_only_120d=joint_only_120d,
    )
    return {
        "relative_source": job.relative_source,
        "output": job.output,
        "task": job.task,
        "episode_id": job.episode_id,
        **result,
        **action_audit,
    }


def _read_conversion_metadata(output_root: Path, output_path: Path) -> dict[str, Any] | None:
    path = output_root / "metadata" / f"{output_path.stem}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--joint-only-120d",
        action="store_true",
        required=True,
        help="require EE poses and hand joints only: 58 active dimensions in the 120-D container",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    expected_jobs = discover_jobs(source_root, output_root)
    expected_by_output = {Path(job.output).name: job for job in expected_jobs}
    actual_names = {path.name for path in output_root.glob("*.zarr") if path.is_dir()}
    expected_names = set(expected_by_output)
    missing_names = sorted(expected_names - actual_names)
    extra_names = sorted(actual_names - expected_names)
    valid_jobs = [expected_by_output[name] for name in sorted(expected_names & actual_names)]

    valid_results: list[dict[str, Any]] = []
    invalid_results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {
            executor.submit(_validate, job, args.joint_only_120d): job
            for job in valid_jobs
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(future_to_job), start=1
        ):
            job = future_to_job[future]
            try:
                valid_results.append(future.result())
            except BaseException as exc:
                invalid_results.append(
                    {
                        "relative_source": job.relative_source,
                        "output": job.output,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            if completed % 25 == 0 or completed == len(valid_jobs):
                print(
                    json.dumps(
                        {
                            "validated": completed,
                            "total_present": len(valid_jobs),
                            "invalid_so_far": len(invalid_results),
                        }
                    ),
                    flush=True,
                )

    task_episodes = Counter(result["task"] for result in valid_results)
    task_frames: dict[str, int] = defaultdict(int)
    instruction_counts: Counter[str] = Counter()
    for result in valid_results:
        task_frames[result["task"]] += int(result["frames"])
        instruction_counts[result["instruction"]] += 1

    metadata_missing: list[str] = []
    metadata_contract_errors: list[dict[str, str]] = []
    maxima = {
        "left_tactile_age_max_ms": 0.0,
        "right_tactile_age_max_ms": 0.0,
        "camera_error_max_ms": 0.0,
    }
    for result in valid_results:
        output = Path(result["output"])
        metadata = _read_conversion_metadata(output_root, output)
        if metadata is None:
            metadata_missing.append(output.name)
            continue
        try:
            frame_count = int(result["frames"])
            if metadata.get("input_mode") != "standardized_30hz":
                raise ValueError("input_mode is not standardized_30hz")
            if metadata.get("resampling_performed") is not False:
                raise ValueError("resampling_performed is not false")
            if int(metadata.get("input_frames", -1)) != frame_count:
                raise ValueError("input_frames does not equal output rows")
            if int(metadata.get("output_frames", -1)) != frame_count:
                raise ValueError("output_frames does not equal output rows")
            standardized = dict(metadata.get("standardized_source", {}))
            if standardized.get("auxiliary_60hz_datasets_read") is not False:
                raise ValueError("auxiliary 60 Hz datasets were not explicitly ignored")
            camera = dict(metadata.get("camera_ego_pose", {}))
            if camera.get("source_rows_verified_identical") is not True:
                raise ValueError("fixed camera extrinsic was not verified")
        except (KeyError, TypeError, ValueError) as exc:
            metadata_contract_errors.append(
                {"output": output.name, "error": str(exc)}
            )
            continue
        maxima["left_tactile_age_max_ms"] = max(
            maxima["left_tactile_age_max_ms"],
            float(metadata["left_tactile_age_ms"]["max"]),
        )
        maxima["right_tactile_age_max_ms"] = max(
            maxima["right_tactile_age_max_ms"],
            float(metadata["right_tactile_age_ms"]["max"]),
        )
        for camera_stats in metadata["camera_error_ms"].values():
            maxima["camera_error_max_ms"] = max(
                maxima["camera_error_max_ms"], float(camera_stats["max_abs"])
            )

    active_action_dimensions = sorted({
        int(result["active_action_dimensions_per_step"]) for result in valid_results
    })
    if len(active_action_dimensions) > 1:
        raise ValueError(f"Inconsistent validated action dimensions: {active_action_dimensions}")

    report = {
        "validated_at": _utc_now(),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "expected_episode_count": len(expected_jobs),
        "present_expected_count": len(valid_jobs),
        "valid_episode_count": len(valid_results),
        "missing_episode_count": len(missing_names),
        "extra_episode_count": len(extra_names),
        "invalid_episode_count": len(invalid_results),
        "metadata_missing_count": len(metadata_missing),
        "metadata_contract_error_count": len(metadata_contract_errors),
        "conversion_contract": "standardized_30hz_row_identity",
        "joint_only_120d": args.joint_only_120d,
        "ftp1_width": 120,
        "active_action_dimensions_per_step": active_action_dimensions[0] if active_action_dimensions else None,
        "action_source_contract": "direct_hdf5_action_group",
        "action_source_verified_episode_count": sum(
            result.get("action_source_verified") is True for result in valid_results
        ),
        "model_input_resolution": [224, 224],
        "image_color_order": "RGB",
        "image_channel_transform": "none",
        "total_converted_frames": sum(int(result["frames"]) for result in valid_results),
        "total_converted_hours": sum(float(result["duration_seconds"]) for result in valid_results)
        / 3600.0,
        "episodes_by_task": dict(sorted(task_episodes.items())),
        "frames_by_task": dict(sorted(task_frames.items())),
        "instructions": dict(sorted(instruction_counts.items())),
        "qc_global_maxima": maxima,
        "missing_outputs": missing_names,
        "extra_outputs": extra_names,
        "invalid_outputs": invalid_results,
        "metadata_missing": metadata_missing,
        "metadata_contract_errors": metadata_contract_errors,
    }
    report_path = (
        args.report.resolve()
        if args.report is not None
        else output_root / "validation_report.json"
    )
    _atomic_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if (
        missing_names
        or extra_names
        or invalid_results
        or metadata_missing
        or metadata_contract_errors
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
