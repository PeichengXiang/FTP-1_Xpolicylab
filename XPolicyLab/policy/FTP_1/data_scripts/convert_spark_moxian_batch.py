#!/usr/bin/env python3
"""Deprecated CLI alias for the standardized 30 Hz production batch driver.

This filename previously selected a raw/60 Hz conversion route.  Invoking it
now delegates immediately to ``convert_spark_moxian_dataset.py``, whose input
is restricted to the curated row-aligned 30 Hz HDF5 schema.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import traceback
from typing import Any, Iterable
import uuid


if __name__ == "__main__":
    from convert_spark_moxian_dataset import main as standardized_main

    print(
        "convert_spark_moxian_batch.py is deprecated; using the standardized "
        "30 Hz row-identity batch converter",
        file=sys.stderr,
    )
    standardized_main()
    raise SystemExit(0)


SCHEMA_VERSION = 1
DEFAULT_INPUT_ROOT = Path(
    "/personal/zijian/Spark_0/data/spark0_real_bench"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/personal/xiangpc/0814_Xpolicylab_bench/FTP-1/data/spark_moxian"
)
DEFAULT_CONVERTER = Path(
    "/personal/xiangpc/0814_Xpolicylab_bench/FTP-1/ftp1-policy/"
    "data_processing/parse_data_module/parse_data_spark_moxian.py"
)

EXPECTED_BASE_KEYS = {
    "timestamps",
    "camera_ego_rgb",
    "left_wrist_camera_rgb",
    "right_wrist_camera_rgb",
    "camera_ego_pose",
    "left_wrist_pose",
    "right_wrist_pose",
    "left_arm_joints",
    "right_arm_joints",
    "left_hand_joints",
    "right_hand_joints",
    "left_hand_joints_idx",
    "right_hand_joints_idx",
    "sub_task_instruction",
}
TACTILE_GROUPS = {
    "fingertip": (5, 4, 4),
    "palm": (1, 15, 16),
}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return value or "unnamed"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _signature_matches(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    return all(expected.get(key) == actual.get(key) for key in ("path", "size_bytes", "mtime_ns"))


@dataclasses.dataclass(frozen=True)
class EpisodeSpec:
    input_path: str
    relative_path: str
    id_name: str
    task_name: str
    sensor_name: str
    source_stem: str
    episode_id: str
    task_dir: str
    output_zarr: str
    artifact_dir: str
    source_signature: dict[str, Any]

    @property
    def dataset_name(self) -> str:
        return f"SparkMoxian_{_slug(self.id_name)}_{_slug(self.task_name)}"


def discover_episodes(input_root: Path, output_root: Path) -> list[EpisodeSpec]:
    """Discover the fixed four-component Spark layout in deterministic order."""
    input_root = input_root.resolve()
    # Keep the user-facing mount spelling (normally /personal/...) in manifests
    # and FTP-1 dataset JSON.  The converter itself may canonicalize it to the
    # equivalent /mnt/xspark-data mount internally.
    output_root = output_root.absolute()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)

    def discovery_key(path: Path) -> tuple[Any, ...]:
        relative = path.relative_to(input_root)
        parts = relative.parts
        stem = path.stem
        episode_key: tuple[int, Any] = (0, int(stem)) if stem.isdigit() else (1, stem)
        return (*parts[:-1], *episode_key)

    specs: list[EpisodeSpec] = []
    seen_outputs: set[str] = set()
    for input_path in sorted(input_root.rglob("*.hdf5"), key=discovery_key):
        relative = input_path.relative_to(input_root)
        if len(relative.parts) != 4:
            raise ValueError(
                "expected <id>/<task>/<sensor>/<episode>.hdf5, got "
                f"{relative.as_posix()}"
            )
        id_name, task_name, sensor_name, filename = relative.parts
        source_stem = Path(filename).stem
        relative_text = relative.as_posix()
        short_hash = hashlib.sha1(relative_text.encode("utf-8")).hexdigest()[:10]
        if source_stem.isdigit():
            episode_label = f"{int(source_stem):06d}"
        else:
            episode_label = _slug(source_stem)
        episode_id = f"{_slug(sensor_name)}__{episode_label}__{short_hash}"
        task_dir = output_root / _slug(id_name) / _slug(task_name)
        output_zarr = task_dir / f"{episode_id}.zarr"
        artifact_dir = task_dir / "_artifacts" / episode_id
        output_key = str(output_zarr)
        if output_key in seen_outputs:
            raise AssertionError(f"output collision: {output_zarr}")
        seen_outputs.add(output_key)
        specs.append(
            EpisodeSpec(
                input_path=str(input_path.resolve()),
                relative_path=relative_text,
                id_name=id_name,
                task_name=task_name,
                sensor_name=sensor_name,
                source_stem=source_stem,
                episode_id=episode_id,
                task_dir=str(task_dir),
                output_zarr=output_key,
                artifact_dir=str(artifact_dir),
                source_signature=_source_signature(input_path),
            )
        )
    return specs


def _load_converter(converter_path: Path):
    module_name = f"parse_data_spark_moxian_batch_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, converter_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load converter: {converter_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _expected_tactile_keys() -> set[str]:
    keys: set[str] = set()
    for side in ("left", "right"):
        for group in TACTILE_GROUPS:
            for member in ("data", "area", "sensor", "type"):
                keys.add(f"{side}_tactile_{member}_{group}")
    return keys


def validate_output_zarr(
    converter_module,
    zarr_path: Path,
    *,
    expected_source: dict[str, Any] | None = None,
    expected_converter_sha256: str | None = None,
    require_all_qc_valid: bool = True,
) -> dict[str, Any]:
    """Perform a loader-oriented, bounded validation without reading all RGB pixels."""
    replay = converter_module.ReplayBuffer.create_from_path(str(zarr_path), mode="r")
    if replay.n_episodes != 1:
        raise ValueError(f"expected one episode, got {replay.n_episodes}")
    episode_ends = replay.episode_ends[:]
    if episode_ends.shape != (1,) or int(episode_ends[0]) < 2:
        raise ValueError(f"invalid episode_ends: {episode_ends!r}")
    frame_count = int(episode_ends[0])

    keys = set(replay.data.keys())
    missing = (EXPECTED_BASE_KEYS | _expected_tactile_keys()) - keys
    if missing:
        raise KeyError(f"missing output keys: {sorted(missing)}")
    for key in keys:
        if replay.data[key].shape[0] != frame_count:
            raise ValueError(
                f"{key} time dimension {replay.data[key].shape[0]} != {frame_count}"
            )

    timestamps = replay.data["timestamps"][:]
    if timestamps.dtype.kind != "f" or not bool((timestamps[1:] > timestamps[:-1]).all()):
        raise ValueError("timestamps must be strictly increasing floating point")
    median_dt = float(__import__("numpy").median(__import__("numpy").diff(timestamps)))
    if not (0.032 <= median_dt <= 0.035):
        raise ValueError(f"unexpected timestamp step: {median_dt}")

    for key in ("camera_ego_rgb", "left_wrist_camera_rgb", "right_wrist_camera_rgb"):
        array = replay.data[key]
        if array.dtype.name != "uint8" or len(array.shape) != 4 or array.shape[-1] != 3:
            raise ValueError(f"invalid RGB array {key}: shape={array.shape}, dtype={array.dtype}")
    for key in ("camera_ego_pose", "left_wrist_pose", "right_wrist_pose"):
        if replay.data[key].shape != (frame_count, 6):
            raise ValueError(f"invalid pose array {key}: {replay.data[key].shape}")
    for side in ("left", "right"):
        if replay.data[f"{side}_arm_joints"].shape != (frame_count, 7):
            raise ValueError(f"invalid {side} arm shape")
        if replay.data[f"{side}_hand_joints"].shape != (frame_count, 20):
            raise ValueError(f"invalid {side} hand shape")
        index_array = replay.data[f"{side}_hand_joints_idx"]
        if index_array.shape != (frame_count, 20) or index_array.dtype.kind not in "iu":
            raise ValueError(f"invalid {side} hand index array")
        first_index = index_array[0]
        if int(first_index.min()) < 0 or int(first_index.max()) > 31 or len(set(first_index.tolist())) != 20:
            raise ValueError(f"invalid {side} hand FAAS indices: {first_index.tolist()}")
        if not bool((index_array[-1] == first_index).all()):
            raise ValueError(f"{side} hand FAAS indices change across the episode")

        for group, trailing_shape in TACTILE_GROUPS.items():
            data = replay.data[f"{side}_tactile_data_{group}"]
            area = replay.data[f"{side}_tactile_area_{group}"]
            sensor = replay.data[f"{side}_tactile_sensor_{group}"]
            tactile_type = replay.data[f"{side}_tactile_type_{group}"]
            if data.shape != (frame_count,) + trailing_shape or data.dtype.kind != "f":
                raise ValueError(f"invalid {side}/{group} tactile data: {data.shape}, {data.dtype}")
            if area.shape != (frame_count, trailing_shape[0]) or area.dtype.kind not in "iu":
                raise ValueError(f"invalid {side}/{group} tactile area: {area.shape}, {area.dtype}")
            if sensor.shape != (frame_count,) or sensor.dtype.kind != "U":
                raise ValueError(f"invalid {side}/{group} tactile sensor dtype: {sensor.dtype}")
            if tactile_type.shape != (frame_count,) or tactile_type.dtype.kind != "U":
                raise ValueError(f"invalid {side}/{group} tactile type dtype: {tactile_type.dtype}")
            if str(sensor[0]) != "MoxianTactileGlove460" or str(sensor[-1]) != str(sensor[0]):
                raise ValueError(f"invalid {side}/{group} sensor name")
            if str(tactile_type[0]) != "matrix" or str(tactile_type[-1]) != "matrix":
                raise ValueError(f"invalid {side}/{group} tactile type")
            expected_area = {
                "fingertip": [0, 1, 2, 3, 4],
                "palm": [5],
            }[group]
            if area[0].tolist() != expected_area or area[-1].tolist() != expected_area:
                raise ValueError(f"invalid {side}/{group} function areas")

    instruction = replay.data["sub_task_instruction"]
    if instruction.dtype.kind != "U" or not str(instruction[0]).strip():
        raise ValueError("instruction must be non-empty fixed-width Unicode")

    attrs = dict(replay.root.attrs)
    if require_all_qc_valid:
        for key in ("left_tactile_age_ms", "right_tactile_age_ms"):
            qc = attrs.get(key)
            if not isinstance(qc, dict) or qc.get("all_valid") is not True:
                raise ValueError(f"QC failed or missing for {key}: {qc}")
        camera_qc = attrs.get("camera_error_ms")
        if not isinstance(camera_qc, dict):
            raise ValueError(f"camera QC is missing: {camera_qc}")
        invalid_cameras = [
            key
            for key in ("camera_ego_rgb", "left_wrist_camera_rgb", "right_wrist_camera_rgb")
            if not isinstance(camera_qc.get(key), dict) or camera_qc[key].get("all_valid") is not True
        ]
        if invalid_cameras:
            raise ValueError(f"camera QC failed for: {invalid_cameras}")
    if expected_source is not None:
        stored = attrs.get("batch_source_signature")
        if not isinstance(stored, dict) or not _signature_matches(expected_source, stored):
            raise ValueError(f"source signature mismatch: expected={expected_source}, stored={stored}")
        raw_signature = attrs.get("batch_raw_signature")
        if isinstance(raw_signature, dict) and raw_signature.get("path"):
            raw_path = Path(str(raw_signature["path"]))
            if not raw_path.is_file() or not _signature_matches(raw_signature, _source_signature(raw_path)):
                raise ValueError(f"raw source signature is stale: {raw_signature}")
    if expected_converter_sha256 is not None:
        stored_converter = attrs.get("batch_converter_sha256")
        if stored_converter != expected_converter_sha256:
            raise ValueError(
                f"converter signature mismatch: expected={expected_converter_sha256}, "
                f"stored={stored_converter}"
            )
    return {
        "frame_count": frame_count,
        "duration_seconds": float(timestamps[-1]),
        "key_count": len(keys),
    }


def _json_dump_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl_fsync(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _rewrite_preview_summary(path: Path, metadata: dict[str, Any]) -> None:
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metadata"] = metadata
    _json_dump_atomic(path, payload)


def _run_job(job: dict[str, Any]) -> dict[str, Any]:
    """Worker entry point: convert, validate in staging, then commit atomically."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    spec = EpisodeSpec(**job["episode"])
    converter_path = Path(job["converter_path"])
    converter_sha256 = str(job["converter_sha256"])
    task_dir = Path(spec.task_dir)
    final_zarr = Path(spec.output_zarr)
    final_artifact = Path(spec.artifact_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "_artifacts").mkdir(exist_ok=True)
    staging_root = task_dir / "_staging"
    staging_root.mkdir(exist_ok=True)
    stage = staging_root / f"{spec.episode_id}--{os.getpid()}--{uuid.uuid4().hex}"
    stage.mkdir()
    staged_zarr = stage / "episode.zarr"
    staged_artifact = stage / "artifact"
    staged_preview = staged_artifact / "preview"

    started_at = _utc_now()
    try:
        module = _load_converter(converter_path)
        metadata = module.convert_episode(
            Path(spec.input_path),
            staged_zarr,
            target_fps=float(job["target_fps"]),
            image_size=int(job["image_size"]),
            max_tactile_age_ms=float(job["max_tactile_age_ms"]),
            max_camera_error_ms=job["max_camera_error_ms"],
            preview_dir=staged_preview,
            overwrite=False,
        )
        raw_signature = _source_signature(Path(str(metadata["raw_hdf5"])))

        final_metadata_path = final_artifact / "conversion.json"
        final_preview = final_artifact / "preview"
        canonical_metadata = dict(metadata)
        canonical_metadata.update(
            {
                "output_zarr": str(final_zarr),
                "metadata_json": str(final_metadata_path),
                "preview_png": str(final_preview / "moxian_ftp1_preview.png"),
                "preview_summary": str(final_preview / "preview_summary.json"),
                "batch_schema_version": SCHEMA_VERSION,
                "batch_episode_id": spec.episode_id,
                "batch_relative_input": spec.relative_path,
                "batch_source_signature": spec.source_signature,
                "batch_raw_signature": raw_signature,
                "batch_converter_sha256": converter_sha256,
                "batch_started_at": started_at,
                "batch_finished_at": _utc_now(),
            }
        )
        replay = module.ReplayBuffer.create_from_path(str(staged_zarr), mode="a")
        replay.root.attrs.update(canonical_metadata)
        del replay

        staged_artifact.mkdir(parents=True, exist_ok=True)
        _json_dump_atomic(staged_artifact / "conversion.json", canonical_metadata)
        _rewrite_preview_summary(staged_preview / "preview_summary.json", canonical_metadata)
        raw_sidecar = stage / "episode_conversion.json"
        if raw_sidecar.is_file():
            os.replace(raw_sidecar, staged_artifact / "converter_raw.json")

        validation = validate_output_zarr(
            module,
            staged_zarr,
            expected_source=spec.source_signature,
            expected_converter_sha256=converter_sha256,
            require_all_qc_valid=not bool(job["allow_invalid_qc"]),
        )
        if final_zarr.exists() or final_artifact.exists():
            raise FileExistsError(
                f"final destination appeared during conversion: {final_zarr} / {final_artifact}"
            )

        # Commit side artifacts first.  The immediate final ``*.zarr`` rename is
        # the completion point observed by resume logic and by FTP-1.
        os.replace(staged_artifact, final_artifact)
        os.replace(staged_zarr, final_zarr)
        try:
            stage.rmdir()
        except OSError:
            # Cleanup is not part of the commit contract.  A harmless residual
            # staging directory must not turn an atomically committed Zarr into
            # a reported conversion failure.
            pass
        return {
            "status": "converted",
            "episode": dataclasses.asdict(spec),
            "output_zarr": str(final_zarr),
            "artifact_dir": str(final_artifact),
            "validation": validation,
            "metadata": canonical_metadata,
            "started_at": started_at,
            "finished_at": _utc_now(),
        }
    except BaseException as exc:
        return {
            "status": "failed",
            "episode": dataclasses.asdict(spec),
            "stage_dir": str(stage),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "started_at": started_at,
            "finished_at": _utc_now(),
        }


def _task_entries(specs: Iterable[EpisodeSpec]) -> list[dict[str, Any]]:
    task_map: dict[tuple[str, str], EpisodeSpec] = {}
    for spec in specs:
        task_map.setdefault((spec.id_name, spec.task_name), spec)
    entries = []
    for key in sorted(task_map):
        spec = task_map[key]
        entries.append(
            {
                "name": spec.dataset_name,
                "path": spec.task_dir,
                "norm_stats_domain_name": "SparkMoxianWujiHand2",
                "use_trajectory_ratio": 1.0,
                "enabled": True,
            }
        )
    return entries


def _build_dataset_config(specs: list[EpisodeSpec]) -> dict[str, Any]:
    return {
        "datasets": _task_entries(specs),
        "default_use_trajectory_ratio": 1.0,
        "description": (
            "Spark real-robot WujiHand2 data with Moxian matrix tactile.  Each path "
            "is an FTP-1 dataset parent whose immediate children are one-episode Zarr files."
        ),
    }


def _load_manifest(path: Path, *, input_root: Path, output_root: Path, converter: Path) -> dict[str, Any]:
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema: {payload.get('schema_version')}")
        return payload
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "converter_path": str(converter),
        "episodes": {},
        "runs": [],
    }


def _quarantine(path: Path, task_dir: Path, reason: str) -> Path:
    quarantine_root = task_dir / "_quarantine"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = quarantine_root / f"{path.name}--{stamp}--{_slug(reason)[:48]}--{uuid.uuid4().hex[:8]}"
    os.replace(path, destination)
    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--converter", type=Path, default=DEFAULT_CONVERTER)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task", action="append", default=[], help="Filter by id/task or task name")
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-tactile-age-ms", type=float, default=70.0)
    parser.add_argument("--max-camera-error-ms", type=float, default=None)
    parser.add_argument(
        "--allow-invalid-qc",
        action="store_true",
        help="Land episodes containing camera/tactile frames marked invalid (unsafe for training)",
    )
    parser.add_argument(
        "--quarantine-invalid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Move stale/invalid generated outputs aside and retry (default: true)",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.workers < 1:
        raise ValueError("workers must be >= 1")
    input_root = args.input_root.resolve()
    output_root = args.output_root.absolute()
    converter_path = args.converter.resolve()
    if not converter_path.is_file():
        raise FileNotFoundError(converter_path)
    converter_sha256 = _sha256_file(converter_path)
    all_specs = discover_episodes(input_root, output_root)
    specs = all_specs
    if args.task:
        filters = set(args.task)
        specs = [
            spec
            for spec in specs
            if spec.task_name in filters or f"{spec.id_name}/{spec.task_name}" in filters
        ]
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("limit must be non-negative")
        specs = specs[: args.limit]

    print(
        json.dumps(
            {
                "input_root": str(input_root),
                "output_root": str(output_root),
                "converter": str(converter_path),
                "converter_sha256": converter_sha256,
                "episodes_selected": len(specs),
                "tasks": len(_task_entries(specs)),
                "workers": args.workers,
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.dry_run:
        for spec in specs[:20]:
            print(f"PLAN {spec.relative_path} -> {spec.output_zarr}")
        if len(specs) > 20:
            print(f"... {len(specs) - 20} additional episodes")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".batch.lock"
    lock_stream = lock_path.open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another batch process holds {lock_path}") from exc

    dataset_config_path = output_root / "dataset_spark_moxian.json"
    # Dataset config is always complete even when this invocation converts only
    # a task filter or pilot limit.
    _json_dump_atomic(dataset_config_path, _build_dataset_config(all_specs))
    manifest_path = output_root / "manifest.json"
    failures_path = output_root / "failures.jsonl"
    manifest = _load_manifest(
        manifest_path, input_root=input_root, output_root=output_root, converter=converter_path
    )
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    run_record = {
        "run_id": run_id,
        "started_at": _utc_now(),
        "converter_sha256": converter_sha256,
        "selected": len(specs),
        "workers": args.workers,
    }
    manifest.setdefault("runs", []).append(run_record)

    module = _load_converter(converter_path)
    pending: list[EpisodeSpec] = []
    skipped = 0
    for spec in specs:
        final_zarr = Path(spec.output_zarr)
        artifact_dir = Path(spec.artifact_dir)
        if final_zarr.exists():
            try:
                validation = validate_output_zarr(
                    module,
                    final_zarr,
                    expected_source=spec.source_signature,
                    expected_converter_sha256=converter_sha256,
                    require_all_qc_valid=not args.allow_invalid_qc,
                )
                if not artifact_dir.is_dir():
                    raise ValueError(f"side-artifact directory is missing: {artifact_dir}")
            except Exception as exc:
                if not args.quarantine_invalid:
                    raise RuntimeError(
                        f"existing output is invalid; rerun with --quarantine-invalid: {final_zarr}: {exc}"
                    ) from exc
                moved = [_quarantine(final_zarr, Path(spec.task_dir), "invalid-zarr")]
                if artifact_dir.exists():
                    moved.append(_quarantine(artifact_dir, Path(spec.task_dir), "invalid-artifact"))
                print(f"QUARANTINE {spec.relative_path}: {moved}")
                pending.append(spec)
            else:
                skipped += 1
                manifest["episodes"][spec.relative_path] = {
                    "status": "complete",
                    "episode_id": spec.episode_id,
                    "output_zarr": spec.output_zarr,
                    "artifact_dir": spec.artifact_dir,
                    "source_signature": spec.source_signature,
                    "converter_sha256": converter_sha256,
                    "validation": validation,
                    "last_verified_at": _utc_now(),
                }
        else:
            if artifact_dir.exists():
                if not args.quarantine_invalid:
                    raise RuntimeError(
                        f"orphan artifact exists; rerun with --quarantine-invalid: {artifact_dir}"
                    )
                moved = _quarantine(artifact_dir, Path(spec.task_dir), "orphan-artifact")
                print(f"QUARANTINE orphan artifact {moved}")
            pending.append(spec)

    manifest["updated_at"] = _utc_now()
    _json_dump_atomic(manifest_path, manifest)
    print(f"RESUME verified={skipped}, pending={len(pending)}")

    base_job = {
        "converter_path": str(converter_path),
        "converter_sha256": converter_sha256,
        "target_fps": args.target_fps,
        "image_size": args.image_size,
        "max_tactile_age_ms": args.max_tactile_age_ms,
        "max_camera_error_ms": args.max_camera_error_ms,
        "allow_invalid_qc": args.allow_invalid_qc,
    }
    converted = 0
    failed = 0
    jobs = [{**base_job, "episode": dataclasses.asdict(spec)} for spec in pending]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(_run_job, job): job for job in jobs}
        for future in concurrent.futures.as_completed(future_map):
            try:
                result = future.result()
            except BaseException as exc:
                job = future_map[future]
                result = {
                    "status": "failed",
                    "episode": job["episode"],
                    "stage_dir": None,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "traceback": traceback.format_exc(),
                    "started_at": None,
                    "finished_at": _utc_now(),
                }
            relative_path = result["episode"]["relative_path"]
            if result["status"] == "converted":
                converted += 1
                manifest["episodes"][relative_path] = {
                    "status": "complete",
                    "episode_id": result["episode"]["episode_id"],
                    "output_zarr": result["output_zarr"],
                    "artifact_dir": result["artifact_dir"],
                    "source_signature": result["episode"]["source_signature"],
                    "converter_sha256": converter_sha256,
                    "validation": result["validation"],
                    "metadata": result["metadata"],
                    "completed_at": result["finished_at"],
                }
                print(f"OK {converted + failed}/{len(pending)} {relative_path}")
            else:
                failed += 1
                failure_event = {
                    "run_id": run_id,
                    **result,
                }
                _append_jsonl_fsync(failures_path, failure_event)
                manifest["episodes"][relative_path] = {
                    "status": "failed",
                    "episode_id": result["episode"]["episode_id"],
                    "source_signature": result["episode"]["source_signature"],
                    "converter_sha256": converter_sha256,
                    "exception_type": result["exception_type"],
                    "exception": result["exception"],
                    "stage_dir": result["stage_dir"],
                    "failed_at": result["finished_at"],
                }
                print(
                    f"FAIL {converted + failed}/{len(pending)} {relative_path}: "
                    f"{result['exception_type']}: {result['exception']}",
                    file=sys.stderr,
                )
            manifest["updated_at"] = _utc_now()
            _json_dump_atomic(manifest_path, manifest)

    run_record.update(
        {
            "finished_at": _utc_now(),
            "verified_existing": skipped,
            "converted": converted,
            "failed": failed,
        }
    )
    manifest["updated_at"] = _utc_now()
    _json_dump_atomic(manifest_path, manifest)
    _json_dump_atomic(output_root / "runs" / f"{run_id}.json", run_record)
    print(json.dumps(run_record, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
