"""Convert curated Spark/HumanTouch 30 Hz HDF5 episodes to FTP-1 Zarr.

The production input is the self-contained 30 Hz HDF5 below
``hdf5/spark0_real/bench``.  Its RGB, state, pose, action, tactile, timestamp, and
validity rows are already aligned, so conversion is strictly row preserving:
no nearest-neighbor frame selection, interpolation, clock fit, stride
sampling, or second tactile baseline is allowed.  Supervised actions come
from the independent HDF5 ``action/*`` group, never from the next state.
The public converter and CLI reject historical raw and ``v2_processed`` inputs.

The standardized source already contains the audited HumanTouch 320-taxel
pressure contract.  It is exported without changing point order or values as
two existing FTP-1 matrix groups:

* five fingertips: ``(T, 5, 4, 4)``, areas ``[0, 1, 2, 3, 4]``;
* palm: ``(T, 1, 15, 16)``, area ``[5]``.

Only ``tactile/taxel_pressure_3D/{left,right}_ee[..., 3]`` is read as tactile
pressure.  Auxiliary ``*_60hz`` and region-pressure datasets in the same HDF5
are never read by the production path.

No model class is added or modified.  These tensors instantiate FTP-1's
existing ``MatrixCNNEncoder`` through the normal tactile configuration path.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.spatial.transform import Rotation, Slerp
from XPolicyLab.utils.process_data import decode_image_bit


_SCRIPT_DIR = Path(__file__).resolve().parent


def _find_data_processing_root() -> Path:
    candidates: list[Path] = []
    repo_override = os.environ.get("FTP1_REPO_ROOT")
    if repo_override:
        candidates.append(Path(repo_override) / "data_processing")
    candidates.extend(
        [
            _SCRIPT_DIR.parent,
            _SCRIPT_DIR.parent / "data_processing",
            _SCRIPT_DIR.parent / "ftp1-policy" / "data_processing",
        ]
    )
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "common" / "replay_buffer.py").is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise RuntimeError(
        "cannot locate FTP-1 data_processing/common/replay_buffer.py; "
        f"set FTP1_REPO_ROOT. Searched: {searched}"
    )


_DATA_PROCESSING_ROOT = _find_data_processing_root()
if str(_DATA_PROCESSING_ROOT) not in sys.path:
    sys.path.insert(0, str(_DATA_PROCESSING_ROOT))

from common.replay_buffer import ReplayBuffer


SENSOR_NAME = "MoxianTactileGlove460"
TACTILE_TYPE = "matrix"
FINGERTIP_AREAS = np.asarray([0, 1, 2, 3, 4], dtype=np.int32)
FINGER_MIDDLE_AREAS = np.asarray([6, 7, 8, 9, 10], dtype=np.int32)
PALM_AREAS = np.asarray([5], dtype=np.int32)
PALM_TAXELS = 240
PALM_HEIGHT = 15
PALM_WIDTH = 16
FINGERTIP_TAXELS = 80
FINGERTIP_COUNT = 5
FINGERTIP_HEIGHT = 4
FINGERTIP_WIDTH = 4

# Raw-only rules are intentionally pinned to both current Spark_data vendor
# stages.  Stage 1 validates/loads source WXYZ.  Stage 2 maps that pose into
# the same Spark robot-base world Link7 frame used by processed data.
SPARK_RAW_STAGE1_CONVERTER = (
    "/personal/zijian/Spark_0/Spark-0/Spark_data/src/spark_data/vendor/"
    "spark_real/raw_stream_to_spark_v1.py"
)
SPARK_RAW_STAGE2_CONVERTER = (
    "/personal/zijian/Spark_0/Spark-0/Spark_data/src/spark_data/vendor/"
    "spark_real/to_spark_world_frame.py"
)
RAW_CAMERA_PRESET = "marvin_head_orbbec"
RAW_CAMERA_EGO_POSE_WXYZ = np.asarray(
    [
        -0.011,
        -0.67532997,
        1.43199995,
        0.96592581,
        0.25881910,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)
SPARK_REAL_TO_ROBOT_BASE_R = np.asarray(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
XROBOT_FLANGE_OFFSET = np.asarray(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)
SPARK_REAL_MIDPOINT_XY_AFTER_RZ_M = np.asarray(
    [0.0007933439028746238, 0.28296904976923726], dtype=np.float64
)
SPARK_REAL_MIN_EE_Z_AFTER_RZ_M = -0.2860525620014648
SPARK_SIM_INIT_MIDPOINT_XY_M = np.asarray(
    [-0.0057082895306813375, -0.5929954433751696], dtype=np.float64
)
SPARK_SIM_INIT_EE_Z_M = 0.9041496858775669
ROBOT_BASE_TRANSLATION = np.asarray(
    [
        SPARK_SIM_INIT_MIDPOINT_XY_M[0] - SPARK_REAL_MIDPOINT_XY_AFTER_RZ_M[0],
        SPARK_SIM_INIT_MIDPOINT_XY_M[1] - SPARK_REAL_MIDPOINT_XY_AFTER_RZ_M[1],
        SPARK_SIM_INIT_EE_Z_M - SPARK_REAL_MIN_EE_Z_AFTER_RZ_M,
    ],
    dtype=np.float64,
)
FIX_BASE_OFFSET_WORLD_X_M = {"left": -0.04, "right": 0.04}
RAW_TASK_INSTRUCTIONS = {
    "build_tower": "Build a tower with the blocks.",
    "clean_blackboard": "Wipe the blackboard clean.",
    "clean_blackboard_left": "Wipe the blackboard clean.",
    "clean_blackboard_right": "Wipe the blackboard clean.",
    "insert_test_tubes": "Insert the test tubes into the rack.",
    "pack_shoes_into_box": "Pack the shoes into the box.",
    # The source collection contains this historical directory typo.
    "pask_shoes_into_box": "Pack the shoes into the box.",
    "place_the_phone": "Place the phone in the designated location.",
    "stack_bowls": "Stack the bowls on top of one another.",
    "stand_up_bottle": "Stand the bottle upright.",
    "transfer_water_with_dropper": "Transfer water from one container to the other using the dropper.",
}
RAW_REQUIRED_KEYS = frozenset(
    {
        "cam_head/color",
        "cam_head/timestamp",
        "cam_left_wrist/color",
        "cam_left_wrist/timestamp",
        "cam_right_wrist/color",
        "cam_right_wrist/timestamp",
        "left_arm/eef",
        "left_arm/joint",
        "left_arm/timestamp",
        "right_arm/eef",
        "right_arm/joint",
        "right_arm/timestamp",
        "left_hand/joint",
        "left_hand/timestamp",
        "right_hand/joint",
        "right_hand/timestamp",
        "left_tactile/pressure",
        "left_tactile/timestamp",
        "right_tactile/pressure",
        "right_tactile/timestamp",
    }
)

# Self-contained 30 Hz Spark/HumanTouch files produced for the curated bench
# export.  Every row in these datasets already refers to the same timestamp;
# this mode must never join back to raw clocks or create another sampling grid.
STANDARDIZED_30HZ_REQUIRED_KEYS = frozenset(
    {
        "timestamps",
        "instruction",
        "additional_info/frequency",
        "vision/cam_head/colors",
        "vision/cam_head/extrinsics",
        "vision/cam_left_wrist/colors",
        "vision/cam_right_wrist/colors",
        "state/left_arm_joint_states",
        "state/right_arm_joint_states",
        "state/left_ee_joint_states",
        "state/right_ee_joint_states",
        "state/left_ee_poses",
        "state/right_ee_poses",
        "tactile/taxel_pressure_3D/left_ee",
        "tactile/taxel_pressure_3D/right_ee",
        "validity/cam_head",
        "validity/cam_left_wrist",
        "validity/cam_right_wrist",
        "validity/left_arm_joint_states",
        "validity/right_arm_joint_states",
        "validity/left_ee_joint_states",
        "validity/right_ee_joint_states",
        "validity/left_ee_pose",
        "validity/right_ee_pose",
        "action/left_arm_joint_states",
        "action/right_arm_joint_states",
        "action/left_ee_joint_states",
        "action/right_ee_joint_states",
        "action/left_ee_poses",
        "action/right_ee_poses",
        "validity/action_left_arm_joint_states",
        "validity/action_right_arm_joint_states",
        "validity/action_left_ee_joint_states",
        "validity/action_right_ee_joint_states",
    }
)
STANDARDIZED_30HZ_ACTION_KEYS = frozenset(
    {
        "action/left_arm_joint_states",
        "action/right_arm_joint_states",
        "action/left_ee_joint_states",
        "action/right_ee_joint_states",
        "action/left_ee_poses",
        "action/right_ee_poses",
    }
)
# Row-aligned exports may be 15 Hz (DexTouch-WM / FTP-1 native) or 30 Hz
# (bench_v5).  Both are accepted only when source_fps == target_fps.
ALLOWED_ROW_ALIGNED_FPS = (15.0, 30.0)
_ACTION_STATE_PAIRS = (
    ("action/left_ee_joint_states", "state/left_ee_joint_states"),
    ("action/right_ee_joint_states", "state/right_ee_joint_states"),
    ("action/left_ee_poses", "state/left_ee_poses"),
    ("action/right_ee_poses", "state/right_ee_poses"),
)
STANDARDIZED_EE_POSE_KEYS = frozenset(
    {
        "state/left_ee_poses",
        "state/right_ee_poses",
        "validity/left_ee_pose",
        "validity/right_ee_pose",
    }
)

# Spark stores WujiHand2 joints in Isaac's stage-major order:
# (index, middle, pinky, ring, thumb) for each of four stages.  These are the
# matching FTP-1 FAAS slots, without changing the joint values themselves.
WUJIHAND2_FAAS_INDEX = np.asarray(
    [
        7,
        12,
        22,
        17,
        1,
        6,
        11,
        21,
        16,
        26,
        8,
        13,
        23,
        18,
        2,
        9,
        14,
        24,
        19,
        3,
    ],
    dtype=np.int32,
)

# A full raw-data audit found two left-hand non-taxel carrier slots that form
# long, exact 0/1 plateaus.  The product point-layout table marks both as
# structural padding.  Keep this allowlist narrow so an unseen glove layout or
# arbitrary corruption is not silently discarded.
KNOWN_PADDING_ARTIFACT_FLAT_INDICES = {
    "left": frozenset({102, 282}),
    "right": frozenset(),
}


@dataclass(frozen=True)
class ClockFit:
    offset_seconds: float
    scale: float
    residual_p95_ms: float
    residual_p99_ms: float
    residual_max_ms: float
    retained_samples: int
    total_samples: int


@dataclass(frozen=True)
class TactileStream:
    mapped_time: np.ndarray
    pressure: np.ndarray
    fit: ClockFit
    first_frame_baseline: np.ndarray | None = None
    raw_padding_diagnostics: dict[str, Any] | None = None


def _decode_text(value: Any) -> str:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def _detect_input_mode(
    handle: h5py.File, *, joint_only_120d: bool = False
) -> str:
    """Classify an input without guessing around a partially written file."""
    standardized_markers = (
        "tactile/taxel_pressure_3D/left_ee",
        "tactile/taxel_pressure_3D/right_ee",
        "additional_info/frequency",
    )
    if any(key in handle for key in standardized_markers):
        # EE-hand-only mode still requires the HDF5 world EE poses; only arm
        # joint arrays are omitted from the exported Zarr contract.
        required_keys = STANDARDIZED_30HZ_REQUIRED_KEYS
        missing = sorted(key for key in required_keys if key not in handle)
        if missing:
            raise ValueError(
                "partial standardized 30 Hz HDF5; missing required keys: "
                f"{missing}"
            )
        return "standardized_30hz"
    if "metadata/provenance_json" in handle:
        return "processed"
    missing = sorted(key for key in RAW_REQUIRED_KEYS if key not in handle)
    if not missing:
        return "raw"
    raise ValueError(
        "input is neither a provenance-backed processed HDF5 nor a complete "
        f"spark0_real_bench raw HDF5; missing raw keys: {missing}"
    )


def _normalize_instruction(value: str) -> str:
    normalized = value.strip()
    if not normalized.endswith("."):
        normalized += "."
    return normalized


def _resolve_instruction(
    input_path: Path,
    processed: h5py.File | None,
    override: str | None,
) -> tuple[str, str]:
    if override is not None:
        if not override.strip():
            raise ValueError("instruction override cannot be empty")
        return _normalize_instruction(override), "explicit_override"
    if processed is not None:
        if "instruction" not in processed:
            raise KeyError("processed HDF5 has no instruction")
        return _normalize_instruction(_decode_text(processed["instruction"][()])), "processed_hdf5"

    matches = [part for part in input_path.parts if part in RAW_TASK_INSTRUCTIONS]
    if len(matches) != 1:
        raise ValueError(
            "raw-only input has no embedded instruction; expected exactly one known "
            "task directory in the path or an explicit --instruction, got "
            f"matches={matches} for {input_path}"
        )
    task = matches[0]
    return RAW_TASK_INSTRUCTIONS[task], f"raw_task_directory:{task}"


def _raw_camera_pose_rows(count: int) -> np.ndarray:
    if count < 1:
        raise ValueError("raw camera pose requires at least one row")
    return np.repeat(RAW_CAMERA_EGO_POSE_WXYZ[None], count, axis=0)


def _quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """NumPy-equivalent of Spark_data.alignment.transforms."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    matrix = np.empty(quaternion.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[..., 0, 1] = 2 * (x * y - z * w)
    matrix[..., 0, 2] = 2 * (x * z + y * w)
    matrix[..., 1, 0] = 2 * (x * y + z * w)
    matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[..., 1, 2] = 2 * (y * z - x * w)
    matrix[..., 2, 0] = 2 * (x * z - y * w)
    matrix[..., 2, 1] = 2 * (y * z + x * w)
    matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def _matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Exact algorithm used by Spark_data.alignment.transforms."""
    matrices = np.asarray(matrix, dtype=np.float64)
    flat = matrices.reshape(-1, 3, 3)
    result = np.empty((len(flat), 4), dtype=np.float64)
    for index, rotation in enumerate(flat):
        trace = np.trace(rotation)
        if trace > 0:
            scale = math.sqrt(trace + 1.0) * 2
            result[index] = [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
            continue
        diagonal = np.diag(rotation)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = math.sqrt(
                1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
            ) * 2
            result[index] = [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
            ]
        elif axis == 1:
            scale = math.sqrt(
                1 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
            ) * 2
            result[index] = [
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
            ]
        else:
            scale = math.sqrt(
                1 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
            ) * 2
            result[index] = [
                (rotation[1, 0] - rotation[0, 1]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
            ]
    result /= np.linalg.norm(result, axis=-1, keepdims=True)
    return result.reshape(matrices.shape[:-2] + (4,))


def _ensure_quaternion_continuity(pose: np.ndarray) -> np.ndarray:
    result = np.asarray(pose, dtype=np.float64).copy()
    for index in range(1, len(result)):
        if np.dot(result[index - 1, 3:7], result[index, 3:7]) < 0:
            result[index, 3:7] *= -1
    return result


def _transform_raw_eef_to_spark_world(
    pose: np.ndarray, side: str
) -> np.ndarray:
    """Apply official Stage-2 default Link7 transform to source WXYZ EEF."""
    if side not in FIX_BASE_OFFSET_WORLD_X_M:
        raise ValueError(f"unknown EEF side: {side}")
    output = np.asarray(pose, dtype=np.float64).copy()
    if output.ndim != 2 or output.shape[1] != 7:
        raise ValueError(f"raw EEF must be (T, 7), got {output.shape}")

    output[:, :3] = output[:, :3] @ SPARK_REAL_TO_ROBOT_BASE_R.T
    world_rotation = SPARK_REAL_TO_ROBOT_BASE_R @ _quaternion_wxyz_to_matrix(
        output[:, 3:7]
    )
    output[:, 3:7] = _matrix_to_quaternion_wxyz(world_rotation)
    output = _ensure_quaternion_continuity(output)

    output[:, :3] += ROBOT_BASE_TRANSLATION
    output[:, 0] += FIX_BASE_OFFSET_WORLD_X_M[side]
    flange_rotation = _quaternion_wxyz_to_matrix(output[:, 3:7])
    output[:, 3:7] = _matrix_to_quaternion_wxyz(
        flange_rotation @ XROBOT_FLANGE_OFFSET
    )
    return _ensure_quaternion_continuity(output)


def _load_raw_state_arrays(
    raw: h5py.File, expected_length: int
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load the official row-aligned raw state contract and reject corruption."""
    for stream in ("left_arm", "right_arm", "left_hand", "right_hand"):
        timestamps = np.asarray(raw[f"{stream}/timestamp"][:])
        if timestamps.shape != (expected_length,):
            raise ValueError(
                f"raw {stream}/timestamp must be ({expected_length},), "
                f"got {timestamps.shape}"
            )
        if not np.all(np.isfinite(timestamps)):
            raise ValueError(f"raw {stream}/timestamp contains non-finite values")

    specs = {
        "left_arm_joints": ("left_arm/joint", 7),
        "right_arm_joints": ("right_arm/joint", 7),
        "left_hand_joints": ("left_hand/joint", 20),
        "right_hand_joints": ("right_hand/joint", 20),
        "left_wrist_pose": ("left_arm/eef", 7),
        "right_wrist_pose": ("right_arm/eef", 7),
    }
    arrays: dict[str, np.ndarray] = {}
    source_valid = np.ones(expected_length, dtype=bool)
    for output_key, (source_key, width) in specs.items():
        values = np.asarray(raw[source_key][:])
        if values.shape != (expected_length, width):
            raise ValueError(
                f"raw {source_key} must be ({expected_length}, {width}), got {values.shape}"
            )
        row_valid = np.all(np.isfinite(values), axis=1)
        source_valid &= row_valid
        arrays[output_key] = values

    for output_key in ("left_wrist_pose", "right_wrist_pose"):
        quaternion_norm = np.linalg.norm(arrays[output_key][:, 3:], axis=1)
        source_valid &= np.isfinite(quaternion_norm) & (
            np.abs(quaternion_norm - 1.0) <= 0.1
        )

    if not np.all(source_valid):
        invalid = np.flatnonzero(~source_valid)
        preview = invalid[:10].tolist()
        raise ValueError(
            "raw state violates the upstream finite/WXYZ-unit-quaternion contract "
            f"at {len(invalid)} rows; first indices: {preview}"
        )
    arrays["left_wrist_pose"] = _transform_raw_eef_to_spark_world(
        arrays["left_wrist_pose"], "left"
    )
    arrays["right_wrist_pose"] = _transform_raw_eef_to_spark_world(
        arrays["right_wrist_pose"], "right"
    )
    return arrays, source_valid


def _load_provenance(processed: h5py.File) -> dict[str, Any]:
    if "metadata/provenance_json" not in processed:
        raise KeyError("processed HDF5 has no metadata/provenance_json")
    return json.loads(_decode_text(processed["metadata/provenance_json"][()]))


def _resolve_raw_path(
    processed_path: Path, provenance: dict[str, Any], raw_override: Path | None
) -> Path:
    if raw_override is not None:
        raw_path = raw_override
    else:
        source_path = provenance.get("source_path")
        if not source_path:
            raise KeyError("provenance_json has no source_path; pass --raw-hdf5")
        raw_path = Path(source_path)

    if raw_path.exists():
        return raw_path

    # The same JuiceFS content is referenced by both mount prefixes in this
    # dataset.  Resolve the alternate spelling without guessing a new file.
    raw_text = str(raw_path)
    alternatives: list[Path] = []
    if raw_text.startswith("/mnt/xspark-data/zijian/"):
        alternatives.append(Path(raw_text.replace("/mnt/xspark-data/zijian/", "/personal/zijian/", 1)))
    if raw_text.startswith("/personal/zijian/"):
        alternatives.append(Path(raw_text.replace("/personal/zijian/", "/mnt/xspark-data/zijian/", 1)))
    for candidate in alternatives:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"raw source from {processed_path} does not exist: {raw_path}"
    )


def _relative_ns(values: np.ndarray, origin_ns: int) -> np.ndarray:
    values_i64 = np.asarray(values, dtype=np.int64)
    # Subtract as int64 before converting to float; epoch nanoseconds cannot be
    # represented at millisecond precision in float64.
    return (values_i64 - np.int64(origin_ns)).astype(np.float64) * 1e-9


def _robust_affine_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ClockFit]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 3:
        raise ValueError(f"invalid affine-fit inputs: x={x.shape}, y={y.shape}")
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        raise ValueError("fewer than three finite clock samples")

    offset = 0.0
    scale = 1.0
    residual = np.zeros_like(x)
    for _ in range(3):
        design = np.column_stack([np.ones(int(keep.sum())), x[keep]])
        offset, scale = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        residual = y - (offset + scale * x)
        cutoff = float(np.quantile(np.abs(residual[keep]), 0.99))
        keep = np.isfinite(residual) & (np.abs(residual) <= cutoff + 1e-12)
        if keep.sum() < 3:
            raise ValueError("robust affine clock fit rejected too many samples")

    mapped = offset + scale * x
    abs_ms = np.abs(residual) * 1e3
    fit = ClockFit(
        offset_seconds=float(offset),
        scale=float(scale),
        residual_p95_ms=float(np.quantile(abs_ms, 0.95)),
        residual_p99_ms=float(np.quantile(abs_ms, 0.99)),
        residual_max_ms=float(abs_ms.max()),
        retained_samples=int(keep.sum()),
        total_samples=int(len(x)),
    )
    return mapped, fit


def _deduplicate_tactile(
    raw: h5py.File,
    pressure_source: h5py.Dataset | np.ndarray,
    side: str,
    head_time: np.ndarray,
) -> TactileStream:
    timestamps = np.asarray(raw[f"{side}_tactile/timestamp"][:], dtype=np.int64)
    pressure = np.asarray(pressure_source[:], dtype=np.float32)
    if pressure.shape != (len(head_time), 460):
        raise ValueError(
            f"{side} pressure must be (T, 460), got {pressure.shape} for T={len(head_time)}"
        )
    if len(timestamps) != len(head_time):
        raise ValueError(f"{side} tactile timestamp length mismatch")
    if np.any(np.diff(timestamps) < 0):
        raise ValueError(f"{side} tactile timestamps are non-monotonic")

    raw_grid = pressure.reshape(len(pressure), 23, 20)
    raw_padding_diagnostics = _validate_structural_padding(raw_grid, side)

    starts = np.flatnonzero(np.r_[True, np.diff(timestamps) != 0])
    ends = np.r_[starts[1:], len(timestamps)]
    for start, end in zip(starts, ends, strict=True):
        if not np.all(pressure[start:end] == pressure[start]):
            raise ValueError(
                f"{side} tactile timestamp repeats with different pressure rows at [{start}:{end}]"
            )

    pressure_unique, first_frame_baseline = _subtract_first_frame_baseline(
        pressure[starts]
    )
    tactile_relative = (
        timestamps[starts] - timestamps[starts[0]]
    ).astype(np.float64) * 1e-9
    observed_head_time = np.asarray(
        [np.median(head_time[start:end]) for start, end in zip(starts, ends, strict=True)],
        dtype=np.float64,
    )
    mapped_time, fit = _robust_affine_fit(tactile_relative, observed_head_time)
    if np.any(np.diff(mapped_time) <= 0):
        raise ValueError(f"{side} fitted tactile time is not strictly increasing")
    return TactileStream(
        mapped_time=mapped_time,
        pressure=pressure_unique,
        fit=fit,
        first_frame_baseline=first_frame_baseline,
        raw_padding_diagnostics=raw_padding_diagnostics,
    )


def _subtract_first_frame_baseline(
    pressure: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the HumanTouch per-episode baseline contract in float32."""
    pressure = np.asarray(pressure, dtype=np.float32)
    if pressure.ndim != 2 or len(pressure) == 0:
        raise ValueError(f"pressure must be a non-empty 2D array, got {pressure.shape}")
    baseline = pressure[0].copy()
    corrected = np.maximum(
        pressure - baseline[None], np.float32(0.0)
    ).astype(np.float32, copy=False)
    return corrected, baseline


def _nearest_indices(sample_time: np.ndarray, target_time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sample_time = np.asarray(sample_time, dtype=np.float64)
    target_time = np.asarray(target_time, dtype=np.float64)
    right = np.clip(np.searchsorted(sample_time, target_time, side="left"), 0, len(sample_time) - 1)
    left = np.maximum(right - 1, 0)
    choose_right = np.abs(sample_time[right] - target_time) < np.abs(sample_time[left] - target_time)
    indices = np.where(choose_right, right, left).astype(np.int64)
    return indices, sample_time[indices] - target_time


def _causal_zoh(
    stream: TactileStream, target_time: np.ndarray, max_age_seconds: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_index = np.searchsorted(stream.mapped_time, target_time, side="right") - 1
    safe_index = np.clip(raw_index, 0, len(stream.mapped_time) - 1)
    age = target_time - stream.mapped_time[safe_index]
    valid = (raw_index >= 0) & (age >= -1e-9) & (age <= max_age_seconds)
    return stream.pressure[safe_index].copy(), valid, age, safe_index.astype(np.int64)


def _interpolation_validity(
    sample_time: np.ndarray, source_valid: np.ndarray, target_time: np.ndarray
) -> np.ndarray:
    sample_time = np.asarray(sample_time, dtype=np.float64)
    source_valid = np.asarray(source_valid, dtype=bool)
    if source_valid.shape != sample_time.shape:
        raise ValueError(
            f"source validity {source_valid.shape} does not match time {sample_time.shape}"
        )
    right = np.clip(np.searchsorted(sample_time, target_time, side="left"), 1, len(sample_time) - 1)
    left = right - 1
    return source_valid[left] & source_valid[right]


def _interp_vector(sample_time: np.ndarray, values: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] != len(sample_time):
        raise ValueError(f"vector interpolation shape mismatch: {values.shape}")
    result = np.empty((len(target_time), values.shape[1]), dtype=np.float32)
    for column in range(values.shape[1]):
        result[:, column] = np.interp(target_time, sample_time, values[:, column])
    return result


def _interp_pose_wxyz(sample_time: np.ndarray, pose: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (len(sample_time), 7):
        raise ValueError(f"pose interpolation expects (T, 7), got {pose.shape}")
    position = _interp_vector(sample_time, pose[:, :3], target_time)
    quat_wxyz = pose[:, 3:]
    quat_norm = np.linalg.norm(quat_wxyz, axis=1)
    if np.any(~np.isfinite(quat_norm)) or np.any(quat_norm < 1e-8):
        raise ValueError("pose contains an invalid quaternion")
    quat_wxyz = quat_wxyz / quat_norm[:, None]
    rotations = Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]])
    rotvec = Slerp(sample_time, rotations)(target_time).as_rotvec().astype(np.float32)
    return np.concatenate([position, rotvec], axis=1).astype(np.float32)


def _decode_jpeg_sequence(dataset: h5py.Dataset, indices: np.ndarray, image_size: int) -> np.ndarray:
    # Spark0 real camera JPEGs were produced by passing RGB arrays directly to
    # cv2.imencode.  The standardized ``vision/*/colors`` files retain those
    # selected JPEG payloads byte-for-byte and label them ``true_rgb``.  Thus,
    # for both raw and standardized inputs, the repository's official
    # decode_image_bit helper returns the RGB tensor expected by FTP-1.  Never
    # add a BGR/RGB conversion after that helper.
    output = np.empty((len(indices), image_size, image_size, 3), dtype=np.uint8)
    row_identity = np.array_equal(
        np.asarray(indices, dtype=np.int64),
        np.arange(len(dataset), dtype=np.int64),
    )
    # Curated 30 Hz input always takes every row exactly once.  Reading the
    # variable-length JPEG dataset in one HDF5 request avoids thousands of
    # tiny JuiceFS/HDF5 lookups while preserving byte payloads and row order.
    row_payloads = dataset[:] if row_identity else None
    cached_index = -1
    cached_image: np.ndarray | None = None
    for out_index, source_index in enumerate(indices):
        source_index_i = int(source_index)
        if source_index_i != cached_index:
            payload = bytes(
                row_payloads[out_index]
                if row_payloads is not None
                else dataset[source_index_i]
            )
            decoded = decode_image_bit(payload)
            decoded = cv2.resize(decoded, (image_size, image_size), interpolation=cv2.INTER_AREA)
            cached_image = decoded
            cached_index = source_index_i
        output[out_index] = cached_image
    return output


def _pose_wxyz_to_xyz_rotvec_rows(pose: np.ndarray, name: str) -> np.ndarray:
    """Change representation only; do not interpolate or reorder 30 Hz rows."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(f"{name} must be (T, 7), got {pose.shape}")
    if not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} contains non-finite values")
    quaternion = pose[:, 3:7]
    norm = np.linalg.norm(quaternion, axis=1)
    if np.any(norm < 1e-8) or np.any(np.abs(norm - 1.0) > 0.1):
        raise ValueError(f"{name} contains a non-unit WXYZ quaternion")
    quaternion = quaternion / norm[:, None]
    rotvec = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]]).as_rotvec()
    return np.concatenate([pose[:, :3], rotvec], axis=1).astype(np.float32)


def _load_fixed_camera_ego_pose(
    handle: h5py.File, frame_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reuse the one fixed head-camera extrinsic for every FTP-1 row.

    The curated HDF5 stores the fixed ``vision/cam_head/extrinsics`` value as
    ``(T, 7)`` only to keep every dataset row aligned.  Treating those rows as
    a moving camera would be misleading, so require exact equality and convert
    the single WXYZ pose to FTP-1's xyz+rotation-vector representation once.
    """
    source = np.asarray(handle["vision/cam_head/extrinsics"][:], dtype=np.float64)
    if source.shape == (7,):
        fixed_wxyz = source
    elif source.shape == (frame_count, 7):
        fixed_wxyz = source[0]
        if not np.array_equal(source, np.repeat(source[:1], frame_count, axis=0)):
            raise ValueError(
                "vision/cam_head/extrinsics must contain one fixed camera extrinsic"
            )
    else:
        raise ValueError(
            "vision/cam_head/extrinsics must be (7,) or "
            f"({frame_count}, 7), got {source.shape}"
        )
    converted = _pose_wxyz_to_xyz_rotvec_rows(
        fixed_wxyz[None], "vision/cam_head/extrinsics"
    )[0]
    return np.repeat(converted[None], frame_count, axis=0), fixed_wxyz.copy()


def _read_standardized_vector(
    handle: h5py.File, key: str, frame_count: int, width: int
) -> np.ndarray:
    values = np.asarray(handle[key][:], dtype=np.float32)
    if values.shape != (frame_count, width):
        raise ValueError(
            f"standardized {key} must be ({frame_count}, {width}), got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"standardized {key} contains non-finite values")
    return values


def _is_allowed_row_aligned_fps(fps: float) -> bool:
    return any(
        math.isclose(float(fps), allowed, rel_tol=0.0, abs_tol=1e-9)
        for allowed in ALLOWED_ROW_ALIGNED_FPS
    )


def _assert_action_datasets_are_independent(
    handle: h5py.File, frame_count: int
) -> None:
    """Require distinct source datasets for state and action.

    The production contract is provenance based: supervision is read directly
    from the HDF5 ``action/*`` datasets and is never synthesized by shifting a
    ``state/*`` array.  Some Spark recordings legitimately contain EE command
    values that are numerically equal to the following observed EE pose, so
    value equality alone cannot establish (or refute) provenance.  We therefore
    reject an HDF5 hard-link alias while allowing two independent datasets to
    contain equal values.
    """
    for action_key, state_key in _ACTION_STATE_PAIRS:
        action = handle[action_key]
        state = handle[state_key]
        if action.shape[0] != frame_count or state.shape[0] != frame_count:
            raise ValueError(
                f"{action_key}/{state_key} length must equal timestamps {frame_count}"
            )
        action_id = getattr(action, "id", None)
        state_id = getattr(state, "id", None)
        if action_id is not None and state_id is not None and action_id == state_id:
            raise ValueError(
                f"{action_key} aliases {state_key}; action must come from an "
                "independent HDF5 dataset"
            )


def _load_standardized_action_arrays(
    handle: h5py.File, frame_count: int, *, joint_only_120d: bool
) -> dict[str, np.ndarray]:
    """Read the independent HDF5 ``action/*`` group. Never substitute next state."""
    arrays = {
        "left_hand_joints_action": _read_standardized_vector(
            handle, "action/left_ee_joint_states", frame_count, 20
        ),
        "right_hand_joints_action": _read_standardized_vector(
            handle, "action/right_ee_joint_states", frame_count, 20
        ),
        "left_wrist_pose_action": _pose_wxyz_to_xyz_rotvec_rows(
            handle["action/left_ee_poses"][:], "action/left_ee_poses"
        ),
        "right_wrist_pose_action": _pose_wxyz_to_xyz_rotvec_rows(
            handle["action/right_ee_poses"][:], "action/right_ee_poses"
        ),
    }
    if not joint_only_120d:
        arrays["left_arm_joints_action"] = _read_standardized_vector(
            handle, "action/left_arm_joint_states", frame_count, 7
        )
        arrays["right_arm_joints_action"] = _read_standardized_vector(
            handle, "action/right_arm_joint_states", frame_count, 7
        )
    if any(len(values) != frame_count for values in arrays.values()):
        raise ValueError("standardized action length does not match timestamps")
    return arrays


def _collapse_standardized_validity(
    handle: h5py.File, key: str, frame_count: int
) -> np.ndarray:
    values = np.asarray(handle[key][:], dtype=bool)
    if values.ndim < 1 or values.shape[0] != frame_count:
        raise ValueError(
            f"standardized {key} must start with ({frame_count},), got {values.shape}"
        )
    if values.ndim > 1:
        values = np.all(values, axis=tuple(range(1, values.ndim)))
    return values


def _load_standardized_tactile_320(
    handle: h5py.File, side: str, frame_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the already calibrated/canonical 30 Hz HumanTouch pressure rows."""
    group = handle["tactile/taxel_pressure_3D"]
    layout = _decode_text(group.attrs.get("layout", "")).strip()
    fields = _decode_text(group.attrs.get("fields", "")).replace(" ", "")
    if layout != "palm_then_thumb_index_middle_ring_pinky":
        raise ValueError(f"unexpected standardized tactile layout: {layout!r}")
    if fields != "x,y,z,pressure":
        raise ValueError(f"unexpected standardized tactile fields: {fields!r}")

    dataset = handle[f"tactile/taxel_pressure_3D/{side}_ee"]
    expected_taxels = PALM_TAXELS + FINGERTIP_TAXELS
    if dataset.shape != (frame_count, expected_taxels, 4):
        raise ValueError(
            f"standardized {side} taxel_pressure_3D must be "
            f"({frame_count}, {expected_taxels}, 4), got {dataset.shape}"
        )
    pressure = np.asarray(dataset[:, :, 3], dtype=np.float32)
    valid = np.all(np.isfinite(pressure), axis=1) & np.all(pressure >= 0, axis=1)
    # The source contract is already palm[240] followed by five fingertips[16].
    # Reshaping preserves the reviewed matrix ordering exactly; no spatial flip
    # or first-frame baseline is applied a second time.
    palm = pressure[:, :PALM_TAXELS].reshape(
        frame_count, 1, PALM_HEIGHT, PALM_WIDTH
    )
    fingertip = pressure[:, PALM_TAXELS:].reshape(
        frame_count,
        FINGERTIP_COUNT,
        FINGERTIP_HEIGHT,
        FINGERTIP_WIDTH,
    )
    return fingertip, palm, valid


def _standardized_carrier_grid(
    fingertip: np.ndarray, palm: np.ndarray, side: str
) -> np.ndarray:
    """Reconstruct a display-only 23x20 carrier grid for preview generation."""
    frame_count = len(fingertip)
    grid = np.zeros((frame_count, 23, 20), dtype=np.float32)
    if side == "left":
        grid[:, 0:15, 4:20] = palm[:, 0, ::-1, :]
    elif side == "right":
        grid[:, 0:15, 0:16] = palm[:, 0, ::-1, :]
    else:
        raise ValueError(f"unknown side: {side}")
    for finger_index, column in enumerate(_finger_columns(side)):
        grid[:, 19:23, column] = fingertip[:, finger_index, ::-1, :]
    return grid


def _finger_columns(side: str) -> tuple[slice, ...]:
    if side == "left":
        return (slice(0, 4), slice(4, 8), slice(8, 12), slice(12, 16), slice(16, 20))
    if side == "right":
        return (slice(16, 20), slice(12, 16), slice(8, 12), slice(4, 8), slice(0, 4))
    raise ValueError(f"unknown side: {side}")


def split_moxian_tactile(
    pressure: np.ndarray, side: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return canonical fingertip, finger-middle, palm, and the raw 23x20 grid."""
    pressure = np.asarray(pressure, dtype=np.float32)
    if pressure.ndim != 2 or pressure.shape[1] != 460:
        raise ValueError(f"Moxian pressure must be (T, 460), got {pressure.shape}")
    grid = pressure.reshape(len(pressure), 23, 20)
    columns = _finger_columns(side)

    fingertip = np.stack([grid[:, 19:23, column] for column in columns], axis=1)
    finger_middle = np.stack([grid[:, 17:19, column] for column in columns], axis=1)
    if side == "left":
        palm = grid[:, 0:15, 4:20]
        padding = np.concatenate(
            [grid[:, 0:15, 0:4].reshape(len(grid), -1), grid[:, 15:17].reshape(len(grid), -1)],
            axis=1,
        )
    else:
        palm = grid[:, 0:15, 0:16]
        padding = np.concatenate(
            [grid[:, 0:15, 16:20].reshape(len(grid), -1), grid[:, 15:17].reshape(len(grid), -1)],
            axis=1,
        )

    # The canonical per-region order is defined by the Moxian channel_ids
    # stored with the audited HumanTouch reference data.  For both hands the
    # carrier rows run opposite to the matrix rows while columns stay in their
    # original order.  Applying the same vertical flip to every region makes
    # raw460 conversion agree exactly with the reference 15x16 / 4x4 layout.
    fingertip = fingertip[:, :, ::-1, :]
    finger_middle = finger_middle[:, :, ::-1, :]
    palm = palm[:, ::-1, :]

    if padding.shape[1] != 100:
        raise AssertionError(f"expected 100 padding positions, got {padding.shape}")
    return (
        fingertip.astype(np.float32),
        finger_middle.astype(np.float32),
        palm[:, None].astype(np.float32),
        grid,
    )


def _assert_split_round_trip(
    grid: np.ndarray,
    side: str,
    fingertip: np.ndarray,
    finger_middle: np.ndarray,
    palm: np.ndarray,
) -> tuple[float, float]:
    recovered = np.zeros_like(grid)
    columns = _finger_columns(side)
    if side == "left":
        recovered[:, 0:15, 4:20] = palm[:, 0, ::-1, :]
    else:
        recovered[:, 0:15, 0:16] = palm[:, 0, ::-1, :]
    for finger_index, column in enumerate(columns):
        tip = fingertip[:, finger_index, ::-1, :]
        middle = finger_middle[:, finger_index, ::-1, :]
        recovered[:, 19:23, column] = tip
        recovered[:, 17:19, column] = middle

    if side == "left":
        valid_mask = np.zeros((23, 20), dtype=bool)
        valid_mask[0:15, 4:20] = True
    else:
        valid_mask = np.zeros((23, 20), dtype=bool)
        valid_mask[0:15, 0:16] = True
    valid_mask[17:23, :] = True
    padding_mask = ~valid_mask
    round_trip_error = float(np.max(np.abs(recovered[:, valid_mask] - grid[:, valid_mask])))
    padding_max = float(np.max(np.abs(grid[:, padding_mask])))
    if round_trip_error != 0.0:
        raise AssertionError(f"{side} tactile split round-trip error: {round_trip_error}")
    return round_trip_error, padding_max


def _padding_diagnostics(grid: np.ndarray, side: str) -> dict[str, Any]:
    """Describe fixed non-taxel carrier bytes without treating them as pressure."""
    valid_mask = np.zeros((23, 20), dtype=bool)
    if side == "left":
        valid_mask[0:15, 4:20] = True
    elif side == "right":
        valid_mask[0:15, 0:16] = True
    else:
        raise ValueError(f"unknown side: {side}")
    valid_mask[17:23, :] = True

    padding_mask = ~valid_mask
    padding = np.asarray(grid)[:, padding_mask]
    flat_indices = np.flatnonzero(padding_mask.reshape(-1))
    active_columns = np.any(padding != 0, axis=0)
    return {
        "max_abs": float(np.max(np.abs(padding))),
        "nonzero_value_count": int(np.count_nonzero(padding)),
        "nonzero_frame_count": int(np.count_nonzero(np.any(padding != 0, axis=1))),
        "nonzero_flat_indices": flat_indices[active_columns].astype(int).tolist(),
    }


def _validate_structural_padding(grid: np.ndarray, side: str) -> dict[str, Any]:
    diagnostics = _padding_diagnostics(grid, side)
    active_indices = set(diagnostics["nonzero_flat_indices"])
    unexpected_indices = sorted(
        active_indices - KNOWN_PADDING_ARTIFACT_FLAT_INDICES[side]
    )
    if unexpected_indices:
        raise ValueError(
            f"{side} Moxian structural padding has unexpected non-zero flat "
            f"indices: {unexpected_indices}"
        )

    if active_indices:
        flat_grid = np.asarray(grid).reshape(len(grid), 460)
        audited_values = flat_grid[:, sorted(active_indices)]
        if not np.all((audited_values == 0.0) | (audited_values == 1.0)):
            raise ValueError(
                f"{side} Moxian known padding artifact must contain only 0/1"
            )
    return diagnostics


def _constant_rows(values: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(values)
    return np.repeat(values[None], count, axis=0)


def _contiguous_valid_segments(
    valid: np.ndarray, min_segment_frames: int
) -> tuple[list[tuple[int, int]], int]:
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 1:
        raise ValueError(f"validity mask must be one-dimensional, got {valid.shape}")
    if min_segment_frames < 1:
        raise ValueError("min_segment_frames must be at least 1")
    padded = np.r_[False, valid, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    segments: list[tuple[int, int]] = []
    dropped_short_frames = 0
    for start, end in zip(starts, ends, strict=True):
        if int(end - start) >= min_segment_frames:
            segments.append((int(start), int(end)))
        else:
            dropped_short_frames += int(end - start)
    return segments, dropped_short_frames


def _fixed_string_array(value: str, count: int, width: int) -> np.ndarray:
    if len(value) > width:
        raise ValueError(f"string of length {len(value)} exceeds fixed width U{width}: {value!r}")
    return np.full(count, value, dtype=f"<U{width}")


def _clock_fit_dict(fit: ClockFit) -> dict[str, Any]:
    return {
        "offset_seconds": fit.offset_seconds,
        "scale": fit.scale,
        "residual_p95_ms": fit.residual_p95_ms,
        "residual_p99_ms": fit.residual_p99_ms,
        "residual_max_ms": fit.residual_max_ms,
        "retained_samples": fit.retained_samples,
        "total_samples": fit.total_samples,
    }


def _timing_stats(sample_time: np.ndarray) -> dict[str, float | int]:
    sample_time = np.asarray(sample_time, dtype=np.float64)
    delta = np.diff(sample_time)
    if len(delta) < 1 or np.any(delta <= 0):
        raise ValueError("timing statistics require at least two increasing samples")
    median_period = float(np.median(delta))
    return {
        "samples": int(len(sample_time)),
        "measured_fps_from_median": 1.0 / median_period,
        "median_period_ms": median_period * 1e3,
        "p99_gap_ms": float(np.quantile(delta, 0.99) * 1e3),
        "max_gap_ms": float(np.max(delta) * 1e3),
    }


def _draw_tactile_grid(
    axis: plt.Axes,
    grid: np.ndarray,
    side: str,
    title: str,
    vmax: float,
) -> None:
    image = axis.imshow(grid, origin="lower", aspect="equal", cmap="magma", vmin=0, vmax=vmax)
    axis.set_title(title)
    axis.set_xlabel("carrier column")
    axis.set_ylabel("carrier row")
    axis.set_xticks([0, 4, 8, 12, 16, 19])
    axis.set_yticks([0, 5, 10, 14, 17, 19, 22])
    palm_x = 4 if side == "left" else 0
    axis.add_patch(Rectangle((palm_x - 0.5, -0.5), 16, 15, fill=False, edgecolor="cyan", linewidth=1.4))
    axis.add_patch(Rectangle((-0.5, 16.5), 20, 2, fill=False, edgecolor="lime", linewidth=1.4))
    axis.add_patch(Rectangle((-0.5, 18.5), 20, 4, fill=False, edgecolor="white", linewidth=1.4))
    axis.text(palm_x + 7.5, 7, "palm", ha="center", va="center", color="cyan", fontsize=8)
    axis.text(9.5, 17.5, "middle", ha="center", va="center", color="lime", fontsize=8)
    axis.text(9.5, 20.5, "fingertip", ha="center", va="center", color="white", fontsize=8)
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.02, label="ADC value")


def _save_preview(
    preview_dir: Path,
    camera_ego_rgb: np.ndarray,
    left_grid: np.ndarray,
    right_grid: np.ndarray,
    left_sum: np.ndarray,
    right_sum: np.ndarray,
    output_time: np.ndarray,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    preview_dir.mkdir(parents=True, exist_ok=True)
    key_frames = {
        "left-peak": int(np.argmax(left_sum)),
        "bilateral": int(np.argmax(np.minimum(left_sum, right_sum))),
        "right-peak": int(np.argmax(right_sum)),
    }
    labels = {
        "left-peak": "Left peak",
        "bilateral": "Bilateral contact",
        "right-peak": "Right peak",
    }
    vmax = float(max(np.quantile(left_grid, 0.999), np.quantile(right_grid, 0.999), 1.0))

    figure = plt.figure(figsize=(15, 14), constrained_layout=True)
    layout = figure.add_gridspec(4, 3, height_ratios=[1.0, 1.15, 1.15, 0.8])
    for column, (key, frame_index) in enumerate(key_frames.items()):
        image_axis = figure.add_subplot(layout[0, column])
        image_axis.imshow(camera_ego_rgb[frame_index])
        image_axis.set_title(
            f"{labels[key]}\nframe {frame_index}, t={output_time[frame_index]:.3f}s"
        )
        image_axis.axis("off")
        _draw_tactile_grid(
            figure.add_subplot(layout[1, column]),
            left_grid[frame_index],
            "left",
            "Left tactile",
            vmax,
        )
        _draw_tactile_grid(
            figure.add_subplot(layout[2, column]),
            right_grid[frame_index],
            "right",
            "Right tactile",
            vmax,
        )

        image_path = preview_dir / f"{key}.jpg"
        plt.imsave(image_path, camera_ego_rgb[frame_index], format="jpg")

    curve_axis = figure.add_subplot(layout[3, :])
    curve_axis.plot(output_time, left_sum, label="left", linewidth=1.0)
    curve_axis.plot(output_time, right_sum, label="right", linewidth=1.0)
    for key, frame_index in key_frames.items():
        curve_axis.axvline(output_time[frame_index], linewidth=0.9, linestyle="--")
        curve_axis.text(
            output_time[frame_index],
            max(left_sum.max(), right_sum.max()) * 0.98,
            labels[key],
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
        )
    curve_axis.set_xlabel("converted episode time (s)")
    curve_axis.set_ylabel("sum of 320 exported taxels")
    curve_axis.grid(alpha=0.2)
    curve_axis.legend(loc="upper right")
    figure.suptitle(
        "Spark real robot → FTP-1 pilot conversion\n"
        f"{metadata['output_frames']} frames at {metadata['target_fps_hz']:.1f} Hz; "
        "cyan=palm, green=finger-middle, white=fingertip",
        fontsize=14,
    )
    preview_png = preview_dir / "moxian_ftp1_preview.png"
    figure.savefig(preview_png, dpi=150)
    plt.close(figure)

    curve_indices = np.unique(
        np.linspace(0, len(output_time) - 1, min(600, len(output_time)), dtype=np.int64)
    )
    summary = {
        "metadata": metadata,
        "curve": {
            "time_seconds": np.round(output_time[curve_indices], 5).tolist(),
            "left_sum": np.round(left_sum[curve_indices], 4).tolist(),
            "right_sum": np.round(right_sum[curve_indices], 4).tolist(),
        },
        "frames": [
            {
                "key": key,
                "label": labels[key],
                "index": frame_index,
                "time_seconds": round(float(output_time[frame_index]), 6),
                "left_sum": round(float(left_sum[frame_index]), 4),
                "right_sum": round(float(right_sum[frame_index]), 4),
                "left_grid": np.round(left_grid[frame_index], 3).tolist(),
                "right_grid": np.round(right_grid[frame_index], 3).tolist(),
                "image_file": f"{key}.jpg",
            }
            for key, frame_index in key_frames.items()
        ],
    }
    summary_path = preview_dir / "preview_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return preview_png, summary_path


def _convert_standardized_30hz_episode(
    input_file: h5py.File,
    input_hdf5: Path,
    output_zarr: Path,
    *,
    instruction: str | None,
    target_fps: float,
    image_size: int,
    min_segment_frames: int,
    preview_dir: Path | None,
    save_preview: bool,
    write_sidecar: bool,
    joint_only_120d: bool = False,
) -> dict[str, Any]:
    """Convert a row-aligned 30 Hz Spark/HumanTouch file without resampling."""
    if "additional_info/frequency" not in input_file:
        raise KeyError("standardized HDF5 has no additional_info/frequency")
    source_fps = float(np.asarray(input_file["additional_info/frequency"][()]).item())
    if not math.isclose(source_fps, target_fps, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "standardized input is row-aligned and must not be resampled: "
            f"source_fps={source_fps}, requested target_fps={target_fps}"
        )

    timestamps = np.asarray(input_file["timestamps"][:], dtype=np.float64)
    if timestamps.ndim != 1 or len(timestamps) < 2:
        raise ValueError(f"standardized timestamps must be a 1D sequence, got {timestamps.shape}")
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("standardized timestamps must be finite and strictly increasing")
    expected_period = 1.0 / source_fps
    if not np.allclose(np.diff(timestamps), expected_period, atol=2e-7, rtol=0.0):
        raise ValueError(
            "standardized timestamps are not a regular "
            f"{source_fps:g} Hz row grid; refusing to repair them by resampling"
        )
    _assert_action_datasets_are_independent(
        input_file, frame_count=int(len(timestamps))
    )
    source_time = timestamps - timestamps[0]
    frame_count = int(len(source_time))
    source_indices = np.arange(frame_count, dtype=np.int64)

    camera_specs = {
        "camera_ego_rgb": ("vision/cam_head/colors", "validity/cam_head"),
        "left_wrist_camera_rgb": (
            "vision/cam_left_wrist/colors",
            "validity/cam_left_wrist",
        ),
        "right_wrist_camera_rgb": (
            "vision/cam_right_wrist/colors",
            "validity/cam_right_wrist",
        ),
    }
    decoded_images: dict[str, np.ndarray] = {}
    camera_valid: dict[str, np.ndarray] = {}
    for output_key, (image_key, validity_key) in camera_specs.items():
        if len(input_file[image_key]) != frame_count:
            raise ValueError(
                f"standardized {image_key} length {len(input_file[image_key])} "
                f"does not match timestamps {frame_count}"
            )
        decoded_images[output_key] = _decode_jpeg_sequence(
            input_file[image_key], source_indices, image_size
        )
        camera_valid[output_key] = _collapse_standardized_validity(
            input_file, validity_key, frame_count
        )

    camera_ego_pose, fixed_camera_extrinsic_wxyz = _load_fixed_camera_ego_pose(
        input_file, frame_count
    )
    state_arrays = {
        "camera_ego_pose": camera_ego_pose,
        "left_arm_joints": _read_standardized_vector(
            input_file, "state/left_arm_joint_states", frame_count, 7
        ),
        "right_arm_joints": _read_standardized_vector(
            input_file, "state/right_arm_joint_states", frame_count, 7
        ),
        "left_hand_joints": _read_standardized_vector(
            input_file, "state/left_ee_joint_states", frame_count, 20
        ),
        "right_hand_joints": _read_standardized_vector(
            input_file, "state/right_ee_joint_states", frame_count, 20
        ),
    }
    state_arrays.update(
        {
            "left_wrist_pose": _pose_wxyz_to_xyz_rotvec_rows(
                input_file["state/left_ee_poses"][:], "state/left_ee_poses"
            ),
            "right_wrist_pose": _pose_wxyz_to_xyz_rotvec_rows(
                input_file["state/right_ee_poses"][:], "state/right_ee_poses"
            ),
        }
    )
    if any(len(values) != frame_count for values in state_arrays.values()):
        raise ValueError("standardized pose/state length does not match timestamps")
    action_arrays = _load_standardized_action_arrays(
        input_file, frame_count, joint_only_120d=joint_only_120d
    )

    robot_state_valid = np.ones(frame_count, dtype=bool)
    state_validity_keys = [
        "validity/left_arm_joint_states",
        "validity/right_arm_joint_states",
        "validity/left_ee_joint_states",
        "validity/right_ee_joint_states",
    ]
    if joint_only_120d:
        state_validity_keys = [
            "validity/left_ee_joint_states",
            "validity/right_ee_joint_states",
            "validity/left_ee_pose",
            "validity/right_ee_pose",
        ]
    else:
        state_validity_keys.extend(
            ["validity/left_ee_pose", "validity/right_ee_pose"]
        )
    for validity_key in state_validity_keys:
        robot_state_valid &= _collapse_standardized_validity(
            input_file, validity_key, frame_count
        )
    action_valid = np.ones(frame_count, dtype=bool)
    action_validity_keys = [
        "validity/action_left_ee_joint_states",
        "validity/action_right_ee_joint_states",
    ]
    if not joint_only_120d:
        action_validity_keys.extend(
            [
                "validity/action_left_arm_joint_states",
                "validity/action_right_arm_joint_states",
            ]
        )
    for validity_key in action_validity_keys:
        action_valid &= _collapse_standardized_validity(
            input_file, validity_key, frame_count
        )

    left_tip, left_palm, left_valid = _load_standardized_tactile_320(
        input_file, "left", frame_count
    )
    right_tip, right_palm, right_valid = _load_standardized_tactile_320(
        input_file, "right", frame_count
    )
    left_grid = _standardized_carrier_grid(left_tip, left_palm, "left")
    right_grid = _standardized_carrier_grid(right_tip, right_palm, "right")

    resolved_instruction, instruction_source = _resolve_instruction(
        input_hdf5, input_file, instruction
    )
    exported_state_arrays = dict(state_arrays)
    exported_action_arrays = dict(action_arrays)
    if joint_only_120d:
        # FTP-1 keeps a fixed 120-D state/action container.  Omitting these
        # three pose arrays is the upstream-supported way to make the wrist
        # pose and head-pose slots exactly zero with a zero action-loss mask.
        # The arm (7-D) and dexterous-hand (20-D in FAAS slots) arrays remain.
        for arm_key in ("camera_ego_pose", "left_arm_joints", "right_arm_joints"):
            exported_state_arrays.pop(arm_key, None)
        for arm_action_key in ("left_arm_joints_action", "right_arm_joints_action"):
            exported_action_arrays.pop(arm_action_key, None)

    episode: dict[str, np.ndarray] = {
        **decoded_images,
        **exported_state_arrays,
        **exported_action_arrays,
        "left_hand_joints_idx": _constant_rows(WUJIHAND2_FAAS_INDEX, frame_count),
        "right_hand_joints_idx": _constant_rows(WUJIHAND2_FAAS_INDEX, frame_count),
        "left_tactile_data_fingertip": left_tip,
        "left_tactile_area_fingertip": _constant_rows(FINGERTIP_AREAS, frame_count),
        "left_tactile_sensor_fingertip": _fixed_string_array(SENSOR_NAME, frame_count, 32),
        "left_tactile_type_fingertip": _fixed_string_array(TACTILE_TYPE, frame_count, 8),
        "left_tactile_data_palm": left_palm,
        "left_tactile_area_palm": _constant_rows(PALM_AREAS, frame_count),
        "left_tactile_sensor_palm": _fixed_string_array(SENSOR_NAME, frame_count, 32),
        "left_tactile_type_palm": _fixed_string_array(TACTILE_TYPE, frame_count, 8),
        "right_tactile_data_fingertip": right_tip,
        "right_tactile_area_fingertip": _constant_rows(FINGERTIP_AREAS, frame_count),
        "right_tactile_sensor_fingertip": _fixed_string_array(SENSOR_NAME, frame_count, 32),
        "right_tactile_type_fingertip": _fixed_string_array(TACTILE_TYPE, frame_count, 8),
        "right_tactile_data_palm": right_palm,
        "right_tactile_area_palm": _constant_rows(PALM_AREAS, frame_count),
        "right_tactile_sensor_palm": _fixed_string_array(SENSOR_NAME, frame_count, 32),
        "right_tactile_type_palm": _fixed_string_array(TACTILE_TYPE, frame_count, 8),
        "sub_task_instruction": _fixed_string_array(
            resolved_instruction, frame_count, 128
        ),
    }

    if any(len(value) != frame_count for value in episode.values()):
        raise AssertionError("not all standardized output fields share the same time dimension")

    combined_valid = left_valid & right_valid & robot_state_valid & action_valid
    for output_key in camera_specs:
        combined_valid &= camera_valid[output_key]
    if frame_count < min_segment_frames:
        raise ValueError(
            f"standardized episode has only {frame_count} rows; minimum is "
            f"{min_segment_frames}"
        )
    if not np.all(combined_valid):
        invalid_counts = {
            "left_tactile": int((~left_valid).sum()),
            "right_tactile": int((~right_valid).sum()),
            "robot_state": int((~robot_state_valid).sum()),
            "action": int((~action_valid).sum()),
            **{
                output_key: int((~camera_valid[output_key]).sum())
                for output_key in camera_specs
            },
        }
        raise ValueError(
            "standardized 30 Hz source contains invalid rows; refusing to drop, "
            f"split, interpolate, or repair them: {invalid_counts}"
        )
    segments = [(0, frame_count)]
    kept_indices = np.arange(frame_count, dtype=np.int64)
    output_frame_count = frame_count
    numeric_arrays = [
        value
        for value in episode.values()
        if value.dtype.kind in "fiu" and value.dtype.kind != "b"
    ]
    if any(not np.all(np.isfinite(value[kept_indices])) for value in numeric_arrays):
        raise ValueError("kept standardized rows contain non-finite numeric values")

    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    replay_buffer = ReplayBuffer.create_from_path(str(output_zarr), mode="a")
    replay_buffer.add_episode(episode, compressors="disk")

    left_sum = left_tip.sum(axis=(1, 2, 3)) + left_palm.sum(axis=(1, 2, 3))
    right_sum = right_tip.sum(axis=(1, 2, 3)) + right_palm.sum(axis=(1, 2, 3))
    timing_stats = _timing_stats(source_time)
    zero_age_stats = {
        "limit": 0.0,
        "p50": 0.0,
        "p99": 0.0,
        "max": 0.0,
        "all_valid": True,
    }
    metadata = {
        "input_hdf5": str(input_hdf5),
        "input_mode": "standardized_30hz",
        "raw_hdf5": None,
        "output_zarr": str(output_zarr),
        "input_frames": frame_count,
        "output_frames": output_frame_count,
        "output_episode_count": 1,
        "resampling_performed": False,
        "row_alignment_strategy": (
            "one input row to one output row; no nearest-neighbor selection, "
            "interpolation, clock fitting, or stride sampling"
        ),
        "dropped_invalid_frames": 0,
        "dropped_short_segment_frames": 0,
        "minimum_segment_frames": min_segment_frames,
        "missing_state_validity_keys": [],
        "state_validity_strategy": (
            "row-aligned arm/hand joint validity masks; EE-pose validity is "
            "intentionally excluded"
            if joint_only_120d
            else "row-aligned standardized validity/* masks"
        ),
        "camera_validity_strategy": (
            "row-aligned validity/cam_* masks; JPEG decode and resize only"
        ),
        "tactile_validity_strategy": (
            "row-aligned finite nonnegative taxel_pressure_3D pressure; no resampling"
        ),
        "invalid_robot_state_grid_frames": int((~robot_state_valid).sum()),
        "segments": [
            {
                "grid_start": 0,
                "grid_end_exclusive": frame_count,
                "frames": frame_count,
                "source_start_seconds": float(source_time[0]),
                "source_end_seconds": float(source_time[-1]),
            }
        ],
        "target_fps_hz": source_fps,
        "source_common_start_seconds": float(source_time[0]),
        "source_common_end_seconds": float(source_time[-1]),
        "output_duration_seconds": float(
            max(frame_count - 1, 0) / source_fps
        ),
        "image_size": [image_size, image_size],
        "image_preprocessing": {
            "decode": "XPolicyLab.utils.process_data.decode_image_bit",
            "channel_transform": "none",
            "resize": f"cv2.INTER_AREA_to_{image_size}x{image_size}",
        },
        "sensor": SENSOR_NAME,
        "tactile_type": TACTILE_TYPE,
        "tactile_shapes": {
            "fingertip": [output_frame_count, *left_tip.shape[1:]],
            "palm": [output_frame_count, *left_palm.shape[1:]],
        },
        "unexported_tactile": {
            "xyz_components": (
                "taxel xyz columns 0:3 are geometry metadata and are not input to "
                "FTP-1's matrix pressure encoder"
            ),
            "forbidden_auxiliary_datasets": [
                "tactile/finger_pressure_60hz",
                "tactile/region_pressure_vector_60hz",
                "tactile/region_pressure_vector",
            ],
        },
        "left_tactile_age_ms": zero_age_stats,
        "right_tactile_age_ms": dict(zero_age_stats),
        "camera_error_ms": {
            key: {
                "limit": 0.0,
                "p99_abs": 0.0,
                "max_abs": 0.0,
                "all_valid": bool(np.all(camera_valid[key][kept_indices])),
            }
            for key in camera_specs
        },
        "measured_timing": {
            "cameras": {
                key: {**timing_stats, "kept_unique_source_frame_ratio": 1.0}
                for key in camera_specs
            },
            "left_tactile": timing_stats,
            "right_tactile": timing_stats,
        },
        "split_round_trip_max_abs": {"left": 0.0, "right": 0.0},
        "structural_padding": {
            "policy": "already removed by standardized HumanTouch 320-taxel export",
            "scope": "not present in this converter input",
        },
        "tactile_pressure_preprocessing": {
            "formula": "direct_copy_of_pressure_component",
            "source": "tactile/taxel_pressure_3D/{left,right}_ee[...,3]",
            "baseline_subtraction_in_ftp1_converter": False,
            "clipping_in_ftp1_converter": False,
            "spatial_reordering_in_ftp1_converter": False,
            "palm_slice": [0, PALM_TAXELS],
            "fingertip_slice": [PALM_TAXELS, PALM_TAXELS + FINGERTIP_TAXELS],
        },
        "camera_ego_pose": {
            "semantics": "fixed head-camera extrinsic repeated on the time axis",
            "source": "vision/cam_head/extrinsics[0]",
            "source_rows_verified_identical": True,
            "source_wxyz": fixed_camera_extrinsic_wxyz.tolist(),
            "output_representation": "xyz+rotation_vector",
            "exported_to_zarr": True,
            "interpolation_performed": False,
        },
        "standardized_source": {
            "frequency_hz": source_fps,
            "tactile_layout": "palm_then_thumb_index_middle_ring_pinky",
            "tactile_fields": "x,y,z,pressure",
            "state_row_mapping": "identity",
            "action_row_mapping": "identity",
            "rgb_row_mapping": "identity",
            "tactile_row_mapping": "identity",
            "timestamp_mapping": "subtract first timestamp only",
            "auxiliary_60hz_datasets_read": False,
        },
        "instruction": resolved_instruction,
        "instruction_source": instruction_source,
        "joint_only_120d": joint_only_120d,
        "action_source": "hdf5_action",
        "ftp1_state_action_contract": {
            "width": 120,
            "active_action_dimensions_per_step": 58 if joint_only_120d else 72,
            "right_wrist_pose_slots": [0, 9],
            "right_arm_joint_slots": [9, 16],
            "right_hand_faas_slots": [16, 48],
            "left_wrist_pose_slots": [48, 57],
            "left_arm_joint_slots": [57, 64],
            "left_hand_faas_slots": [64, 96],
            "head_pose_slots": [96, 105],
            "reserved_slots": [105, 120],
            "pose_arrays_exported": True,
            "pose_state_values_when_joint_only": "from_hdf5_state",
            "pose_action_values_when_joint_only": "from_hdf5_action",
            "pose_action_mask_when_joint_only": 1,
            "action_source": "hdf5_action",
        },
    }
    replay_buffer.root.attrs.update(metadata)

    if write_sidecar:
        metadata_path = output_zarr.parent / f"{output_zarr.stem}_conversion.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        metadata["metadata_json"] = str(metadata_path)
    if save_preview:
        if preview_dir is None:
            preview_dir = output_zarr.parent / f"{output_zarr.stem}_preview"
        preview_png, preview_summary = _save_preview(
            preview_dir,
            decoded_images["camera_ego_rgb"][kept_indices],
            left_grid[kept_indices],
            right_grid[kept_indices],
            left_sum[kept_indices],
            right_sum[kept_indices],
            np.arange(output_frame_count, dtype=np.float64) / source_fps,
            metadata,
        )
        metadata["preview_png"] = str(preview_png)
        metadata["preview_summary"] = str(preview_summary)
    return metadata


def _convert_episode_legacy_for_audit(
    input_hdf5: Path,
    output_zarr: Path,
    *,
    raw_hdf5: Path | None = None,
    instruction: str | None = None,
    target_fps: float = 30.0,
    image_size: int = 224,
    max_tactile_age_ms: float = 70.0,
    max_camera_error_ms: float | None = 70.0,
    min_segment_frames: int = 32,
    preview_dir: Path | None = None,
    overwrite: bool = False,
    save_preview: bool = True,
    write_sidecar: bool = True,
) -> dict[str, Any]:
    input_hdf5 = input_hdf5.resolve()
    output_zarr = output_zarr.resolve()
    if not input_hdf5.is_file():
        raise FileNotFoundError(input_hdf5)
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if min_segment_frames < 1:
        raise ValueError("min_segment_frames must be at least 1")
    if output_zarr.exists():
        if not overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {output_zarr}")
        shutil.rmtree(output_zarr)

    with h5py.File(input_hdf5, "r") as input_file:
        input_mode = _detect_input_mode(input_file)
        if input_mode == "standardized_30hz":
            if raw_hdf5 is not None:
                raise ValueError(
                    "--raw-hdf5 is not used for self-contained standardized 30 Hz input"
                )
            return _convert_standardized_30hz_episode(
                input_file,
                input_hdf5,
                output_zarr,
                instruction=instruction,
                target_fps=target_fps,
                image_size=image_size,
                min_segment_frames=min_segment_frames,
                preview_dir=preview_dir,
                save_preview=save_preview,
                write_sidecar=write_sidecar,
                joint_only_120d=False,
            )
        if input_mode == "processed":
            processed: h5py.File | None = input_file
            provenance = _load_provenance(processed)
            raw_path = _resolve_raw_path(input_hdf5, provenance, raw_hdf5)
            raw_context = h5py.File(raw_path, "r")
        else:
            processed = None
            raw_path = input_hdf5
            if raw_hdf5 is not None and raw_hdf5.resolve() != input_hdf5:
                raise ValueError("--raw-hdf5 cannot point elsewhere for raw-only input")
            raw_context = nullcontext(input_file)

        with raw_context as raw:
            head_timestamp = np.asarray(raw["cam_head/timestamp"][:], dtype=np.int64)
            head_origin_ns = int(head_timestamp[0])
            head_time = _relative_ns(head_timestamp, head_origin_ns)
            if processed is not None and len(head_time) != len(processed["timestamps"]):
                raise ValueError("raw head camera and processed episode length differ")
            if np.any(np.diff(head_time) <= 0):
                raise ValueError("head camera timestamps are not strictly increasing")

            if processed is not None:
                image_source = processed
                camera_paths = {
                    "camera_ego_rgb": ("cam_head", "vision/cam_head/colors"),
                    "left_wrist_camera_rgb": (
                        "cam_left_wrist",
                        "vision/cam_left_wrist/colors",
                    ),
                    "right_wrist_camera_rgb": (
                        "cam_right_wrist",
                        "vision/cam_right_wrist/colors",
                    ),
                }
                tactile_pressure_sources = {
                    "left": processed["tactile/left_ee/pressure"],
                    "right": processed["tactile/right_ee/pressure"],
                }
            else:
                image_source = raw
                camera_paths = {
                    "camera_ego_rgb": ("cam_head", "cam_head/color"),
                    "left_wrist_camera_rgb": (
                        "cam_left_wrist",
                        "cam_left_wrist/color",
                    ),
                    "right_wrist_camera_rgb": (
                        "cam_right_wrist",
                        "cam_right_wrist/color",
                    ),
                }
                tactile_pressure_sources = {
                    "left": raw["left_tactile/pressure"],
                    "right": raw["right_tactile/pressure"],
                }
            camera_times = {
                output_key: _relative_ns(raw[f"{raw_key}/timestamp"][:], head_origin_ns)
                for output_key, (raw_key, _) in camera_paths.items()
            }
            for output_key, (_, image_key) in camera_paths.items():
                if len(image_source[image_key]) != len(camera_times[output_key]):
                    raise ValueError(
                        f"{image_key} length does not match its camera timestamp stream"
                    )
                if np.any(np.diff(camera_times[output_key]) <= 0):
                    raise ValueError(f"{output_key} timestamps are not strictly increasing")

            left_stream = _deduplicate_tactile(
                raw, tactile_pressure_sources["left"], "left", head_time
            )
            right_stream = _deduplicate_tactile(
                raw, tactile_pressure_sources["right"], "right", head_time
            )
            start_candidates = [time[0] for time in camera_times.values()]
            start_candidates += [left_stream.mapped_time[0], right_stream.mapped_time[0]]
            end_candidates = [time[-1] for time in camera_times.values()]
            end_candidates += [left_stream.mapped_time[-1], right_stream.mapped_time[-1]]
            source_start = float(max(start_candidates))
            source_end = float(min(end_candidates))
            if source_end <= source_start:
                raise ValueError(
                    f"modalities have no common interval: start={source_start}, end={source_end}"
                )
            frame_count = int(math.floor((source_end - source_start) * target_fps + 1e-9)) + 1
            source_grid = source_start + np.arange(frame_count, dtype=np.float64) / target_fps
            output_time = np.arange(frame_count, dtype=np.float64) / target_fps

            if processed is not None:
                state_arrays = {
                    "camera_ego_pose": np.asarray(
                        processed["vision/cam_head/extrinsics"][:]
                    ),
                    "left_wrist_pose": np.asarray(
                        processed["state/left_ee_poses"][:]
                    ),
                    "right_wrist_pose": np.asarray(
                        processed["state/right_ee_poses"][:]
                    ),
                    "left_arm_joints": np.asarray(
                        processed["state/left_arm_joint_states"][:]
                    ),
                    "right_arm_joints": np.asarray(
                        processed["state/right_arm_joint_states"][:]
                    ),
                    "left_hand_joints": np.asarray(
                        processed["state/left_ee_joint_states"][:]
                    ),
                    "right_hand_joints": np.asarray(
                        processed["state/right_ee_joint_states"][:]
                    ),
                }
                state_source_valid = np.ones(len(head_time), dtype=bool)
                state_validity_keys = (
                    "validity/left_arm_joint_states",
                    "validity/right_arm_joint_states",
                    "validity/left_ee_joint_states",
                    "validity/right_ee_joint_states",
                    "validity/left_ee_pose",
                    "validity/right_ee_pose",
                )
                missing_state_validity: list[str] = []
                for validity_key in state_validity_keys:
                    if validity_key not in processed:
                        missing_state_validity.append(validity_key)
                        continue
                    values = np.asarray(processed[validity_key][:], dtype=bool)
                    if values.shape[0] != len(head_time):
                        raise ValueError(
                            f"{validity_key} length {values.shape[0]} does not match "
                            f"head camera length {len(head_time)}"
                        )
                    if values.ndim > 1:
                        values = np.all(values, axis=tuple(range(1, values.ndim)))
                    state_source_valid &= values
                state_validity_strategy = (
                    "processed validity/* masks when present; missing masks are "
                    "recorded and treated as valid"
                )
            else:
                state_arrays, state_source_valid = _load_raw_state_arrays(
                    raw, len(head_time)
                )
                state_arrays["camera_ego_pose"] = _raw_camera_pose_rows(len(head_time))
                missing_state_validity = []
                state_validity_strategy = (
                    "raw rows must be finite; EEF WXYZ quaternion norm must be within "
                    "0.1 of one; valid EEF is mapped through official Stage-2 default "
                    "Link7 world transform; violations fail conversion"
                )
            robot_state_valid = _interpolation_validity(
                head_time, state_source_valid, source_grid
            )

            camera_indices: dict[str, np.ndarray] = {}
            camera_valid: dict[str, np.ndarray] = {}
            camera_error: dict[str, np.ndarray] = {}
            camera_error_limit_ms: dict[str, float] = {}
            decoded_images: dict[str, np.ndarray] = {}
            for output_key, (raw_key, processed_key) in camera_paths.items():
                sample_time = camera_times[output_key]
                source_valid = np.ones(len(sample_time), dtype=bool)
                if processed is not None:
                    validity_candidates = (
                        f"validity/{raw_key}",
                        processed_key.replace("vision/", "validity/vision/"),
                    )
                    validity_key = next(
                        (key for key in validity_candidates if key in processed), None
                    )
                    if validity_key is not None:
                        source_valid = np.asarray(processed[validity_key][:], dtype=bool)
                        if source_valid.shape != sample_time.shape:
                            raise ValueError(
                                f"{validity_key} shape {source_valid.shape} does not match "
                                f"camera timestamps {sample_time.shape}"
                            )
                valid_source_indices = np.flatnonzero(source_valid)
                if len(valid_source_indices) < 2:
                    raise ValueError(f"{output_key} has fewer than two valid source frames")
                valid_sample_time = sample_time[valid_source_indices]
                local_indices, error = _nearest_indices(valid_sample_time, source_grid)
                indices = valid_source_indices[local_indices]
                if max_camera_error_ms is None:
                    # A 20 Hz source legitimately needs up to 25 ms when placed
                    # on a 30 Hz target grid.  Base the limit on both grids,
                    # while invalid source frames are skipped before matching.
                    source_period = float(np.median(np.diff(valid_sample_time)))
                    error_limit_ms = max(
                        500.0 / target_fps,
                        source_period * 1e3 * 0.55,
                    )
                else:
                    error_limit_ms = float(max_camera_error_ms)
                valid = np.abs(error) <= error_limit_ms * 1e-3 + 1e-12
                camera_indices[output_key] = indices
                camera_valid[output_key] = valid
                camera_error[output_key] = error
                camera_error_limit_ms[output_key] = error_limit_ms
                decoded_images[output_key] = _decode_jpeg_sequence(
                    image_source[processed_key], indices, image_size
                )

            left_tactile_period = float(np.median(np.diff(left_stream.mapped_time)))
            right_tactile_period = float(np.median(np.diff(right_stream.mapped_time)))
            left_max_age_seconds = max(
                max_tactile_age_ms * 1e-3, 2.1 * left_tactile_period
            )
            right_max_age_seconds = max(
                max_tactile_age_ms * 1e-3, 2.1 * right_tactile_period
            )
            left_pressure, left_valid, left_age, _ = _causal_zoh(
                left_stream, source_grid, left_max_age_seconds
            )
            right_pressure, right_valid, right_age, _ = _causal_zoh(
                right_stream, source_grid, right_max_age_seconds
            )

            left_tip, left_middle, left_palm, left_grid = split_moxian_tactile(
                left_pressure, "left"
            )
            right_tip, right_middle, right_palm, right_grid = split_moxian_tactile(
                right_pressure, "right"
            )
            left_round_trip, left_padding_after_baseline = _assert_split_round_trip(
                left_grid, "left", left_tip, left_middle, left_palm
            )
            right_round_trip, right_padding_after_baseline = _assert_split_round_trip(
                right_grid, "right", right_tip, right_middle, right_palm
            )
            # These 100 bytes are fixed non-taxel positions in Moxian's
            # published 23x20 point map.  They are excluded by topology, not
            # inferred from their values.  Two dataset-audited left slots can
            # contain exact 0/1 plateaus; record them for QC while continuing
            # to fail closed on any unseen index or value.
            left_padding_diagnostics = left_stream.raw_padding_diagnostics
            right_padding_diagnostics = right_stream.raw_padding_diagnostics
            if left_padding_diagnostics is None or right_padding_diagnostics is None:
                raise AssertionError("raw structural-padding audit was not retained")
            if (
                left_stream.first_frame_baseline is None
                or right_stream.first_frame_baseline is None
            ):
                raise AssertionError("tactile first-frame baseline was not retained")

            resolved_instruction, instruction_source = _resolve_instruction(
                input_hdf5, processed, instruction
            )

            episode: dict[str, np.ndarray] = {
                "timestamps": output_time,
                **decoded_images,
                "camera_ego_pose": _interp_pose_wxyz(
                    head_time, state_arrays["camera_ego_pose"], source_grid
                ),
                "left_wrist_pose": _interp_pose_wxyz(
                    head_time, state_arrays["left_wrist_pose"], source_grid
                ),
                "right_wrist_pose": _interp_pose_wxyz(
                    head_time, state_arrays["right_wrist_pose"], source_grid
                ),
                "left_arm_joints": _interp_vector(
                    head_time, state_arrays["left_arm_joints"], source_grid
                ),
                "right_arm_joints": _interp_vector(
                    head_time, state_arrays["right_arm_joints"], source_grid
                ),
                "left_hand_joints": _interp_vector(
                    head_time, state_arrays["left_hand_joints"], source_grid
                ),
                "right_hand_joints": _interp_vector(
                    head_time, state_arrays["right_hand_joints"], source_grid
                ),
                "left_hand_joints_idx": _constant_rows(WUJIHAND2_FAAS_INDEX, frame_count),
                "right_hand_joints_idx": _constant_rows(WUJIHAND2_FAAS_INDEX, frame_count),
                "left_tactile_data_fingertip": left_tip,
                "left_tactile_area_fingertip": _constant_rows(FINGERTIP_AREAS, frame_count),
                "left_tactile_sensor_fingertip": _fixed_string_array(
                    SENSOR_NAME, frame_count, 32
                ),
                "left_tactile_type_fingertip": _fixed_string_array(
                    TACTILE_TYPE, frame_count, 8
                ),
                "left_tactile_data_palm": left_palm,
                "left_tactile_area_palm": _constant_rows(PALM_AREAS, frame_count),
                "left_tactile_sensor_palm": _fixed_string_array(SENSOR_NAME, frame_count, 32),
                "left_tactile_type_palm": _fixed_string_array(TACTILE_TYPE, frame_count, 8),
                "right_tactile_data_fingertip": right_tip,
                "right_tactile_area_fingertip": _constant_rows(FINGERTIP_AREAS, frame_count),
                "right_tactile_sensor_fingertip": _fixed_string_array(
                    SENSOR_NAME, frame_count, 32
                ),
                "right_tactile_type_fingertip": _fixed_string_array(
                    TACTILE_TYPE, frame_count, 8
                ),
                "right_tactile_data_palm": right_palm,
                "right_tactile_area_palm": _constant_rows(PALM_AREAS, frame_count),
                "right_tactile_sensor_palm": _fixed_string_array(SENSOR_NAME, frame_count, 32),
                "right_tactile_type_palm": _fixed_string_array(TACTILE_TYPE, frame_count, 8),
                "sub_task_instruction": _fixed_string_array(
                    resolved_instruction, frame_count, 128
                ),
                # Loader-safe QC arrays.  They do not match a tactile-data prefix,
                # so FTP-1 ignores them while validation tools can inspect them.
                "qc_source_time_seconds": source_grid.astype(np.float64),
                "qc_left_tactile_valid": left_valid.astype(bool),
                "qc_right_tactile_valid": right_valid.astype(bool),
                "qc_left_tactile_age_ms": (left_age * 1e3).astype(np.float32),
                "qc_right_tactile_age_ms": (right_age * 1e3).astype(np.float32),
                "qc_robot_state_valid": robot_state_valid.astype(bool),
            }
            for output_key in camera_paths:
                qc_name = output_key.removesuffix("_rgb")
                episode[f"qc_{qc_name}_valid"] = camera_valid[output_key].astype(bool)
                episode[f"qc_{qc_name}_error_ms"] = (
                    camera_error[output_key] * 1e3
                ).astype(np.float32)

    if any(len(value) != frame_count for value in episode.values()):
        raise AssertionError("not all output fields share the same time dimension")

    combined_valid = left_valid & right_valid & robot_state_valid
    for output_key in camera_paths:
        combined_valid &= camera_valid[output_key]
    segments, dropped_short_segment_frames = _contiguous_valid_segments(
        combined_valid, min_segment_frames
    )
    if not segments:
        raise ValueError(
            "no synchronized segment is long enough after QC: "
            f"grid_frames={frame_count}, valid_frames={int(combined_valid.sum())}, "
            f"minimum={min_segment_frames}"
        )
    kept_indices = np.concatenate(
        [np.arange(start, end, dtype=np.int64) for start, end in segments]
    )
    output_frame_count = int(len(kept_indices))
    numeric_arrays = [
        value
        for value in episode.values()
        if value.dtype.kind in "fiu" and value.dtype.kind != "b"
    ]
    if any(not np.all(np.isfinite(value[kept_indices])) for value in numeric_arrays):
        raise ValueError("kept converted frames contain non-finite numeric values")

    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    replay_buffer = ReplayBuffer.create_from_path(str(output_zarr), mode="a")
    for start, end in segments:
        segment_episode = {key: value[start:end] for key, value in episode.items()}
        segment_episode["timestamps"] = (
            np.arange(end - start, dtype=np.float64) / target_fps
        )
        replay_buffer.add_episode(segment_episode, compressors="disk")

    left_sum = (
        left_tip.sum(axis=(1, 2, 3))
        + left_palm.sum(axis=(1, 2, 3))
    )
    right_sum = (
        right_tip.sum(axis=(1, 2, 3))
        + right_palm.sum(axis=(1, 2, 3))
    )
    metadata = {
        "input_hdf5": str(input_hdf5),
        "input_mode": input_mode,
        "raw_hdf5": str(raw_path),
        "output_zarr": str(output_zarr),
        "input_frames": int(len(head_time)),
        "resampled_grid_frames": frame_count,
        "output_frames": output_frame_count,
        "output_episode_count": len(segments),
        "dropped_invalid_frames": int((~combined_valid).sum()),
        "dropped_short_segment_frames": dropped_short_segment_frames,
        "minimum_segment_frames": min_segment_frames,
        "missing_state_validity_keys": missing_state_validity,
        "state_validity_strategy": state_validity_strategy,
        "camera_validity_strategy": (
            "nearest timestamp error threshold plus processed validity mask when present"
            if input_mode == "processed"
            else "nearest timestamp error threshold; malformed JPEG or stream mismatch fails"
        ),
        "tactile_validity_strategy": (
            "deduplicate equal timestamps, robust affine clock fit, causal zero-order "
            "hold, and maximum-age gate"
        ),
        "invalid_robot_state_grid_frames": int((~robot_state_valid).sum()),
        "segments": [
            {
                "grid_start": start,
                "grid_end_exclusive": end,
                "frames": end - start,
                "source_start_seconds": float(source_grid[start]),
                "source_end_seconds": float(source_grid[end - 1]),
            }
            for start, end in segments
        ],
        "target_fps_hz": float(target_fps),
        "source_common_start_seconds": source_start,
        "source_common_end_seconds": source_end,
        "output_duration_seconds": float(
            sum(max(end - start - 1, 0) for start, end in segments) / target_fps
        ),
        "image_size": [image_size, image_size],
        "sensor": SENSOR_NAME,
        "tactile_type": TACTILE_TYPE,
        "tactile_shapes": {
            "fingertip": [output_frame_count, *left_tip.shape[1:]],
            "palm": [output_frame_count, *left_palm.shape[1:]],
        },
        "unexported_tactile": {
            "finger_middle_shape_per_hand": [output_frame_count, *left_middle.shape[1:]],
            "reason": (
                "40 physical middle-finger taxels are excluded to match the reviewed "
                "HumanTouch 320-taxel contract (palm 240 + fingertips 80)"
            ),
        },
        "left_clock_fit": _clock_fit_dict(left_stream.fit),
        "right_clock_fit": _clock_fit_dict(right_stream.fit),
        "left_tactile_age_ms": {
            "limit": left_max_age_seconds * 1e3,
            "p50": float(np.quantile(left_age[kept_indices] * 1e3, 0.50)),
            "p99": float(np.quantile(left_age[kept_indices] * 1e3, 0.99)),
            "max": float(np.max(left_age[kept_indices] * 1e3)),
            "all_valid": bool(np.all(left_valid[kept_indices])),
        },
        "right_tactile_age_ms": {
            "limit": right_max_age_seconds * 1e3,
            "p50": float(np.quantile(right_age[kept_indices] * 1e3, 0.50)),
            "p99": float(np.quantile(right_age[kept_indices] * 1e3, 0.99)),
            "max": float(np.max(right_age[kept_indices] * 1e3)),
            "all_valid": bool(np.all(right_valid[kept_indices])),
        },
        "camera_error_ms": {
            key: {
                "limit": camera_error_limit_ms[key],
                "p99_abs": float(
                    np.quantile(np.abs(camera_error[key][kept_indices]) * 1e3, 0.99)
                ),
                "max_abs": float(
                    np.max(np.abs(camera_error[key][kept_indices])) * 1e3
                ),
                "all_valid": bool(np.all(camera_valid[key][kept_indices])),
            }
            for key in camera_paths
        },
        "measured_timing": {
            "cameras": {
                key: {
                    **_timing_stats(camera_times[key]),
                    "kept_unique_source_frame_ratio": float(
                        len(np.unique(camera_indices[key][kept_indices]))
                        / output_frame_count
                    ),
                }
                for key in camera_paths
            },
            "left_tactile": _timing_stats(left_stream.mapped_time),
            "right_tactile": _timing_stats(right_stream.mapped_time),
        },
        "split_round_trip_max_abs": {
            "left": left_round_trip,
            "right": right_round_trip,
        },
        "padding_max_abs": {
            "left": float(left_padding_diagnostics["max_abs"]),
            "right": float(right_padding_diagnostics["max_abs"]),
        },
        "padding_max_abs_after_first_frame_baseline": {
            "left": left_padding_after_baseline,
            "right": right_padding_after_baseline,
        },
        "structural_padding": {
            "policy": (
                "fixed 100-byte non-taxel mask from Moxian V1.1 point-layout "
                "table; audited left flat indices 102/282 may contain exact "
                "0/1, all values are audited and never exported"
            ),
            "scope": "raw source pressure before first-frame baseline and resampling",
            "left": left_padding_diagnostics,
            "right": right_padding_diagnostics,
        },
        "tactile_pressure_preprocessing": {
            "formula": "maximum(float32_pressure - first_source_frame, 0.0)",
            "scope": "per episode, per hand, per raw channel before resampling",
            "reference_contract": (
                "HumanTouch finger_pressure_60hz and taxel_pressure_3D pressure field"
            ),
            "initial_contact_caveat": (
                "a real contact already present in the first source frame is treated "
                "as baseline; the untouched raw HDF5 remains the audit source"
            ),
            "left_first_frame_baseline": {
                "values_float32": left_stream.first_frame_baseline.astype(float).tolist(),
                "nonzero_count": int(np.count_nonzero(left_stream.first_frame_baseline)),
                "sum": float(np.sum(left_stream.first_frame_baseline, dtype=np.float64)),
                "max": float(np.max(left_stream.first_frame_baseline)),
            },
            "right_first_frame_baseline": {
                "values_float32": right_stream.first_frame_baseline.astype(float).tolist(),
                "nonzero_count": int(np.count_nonzero(right_stream.first_frame_baseline)),
                "sum": float(np.sum(right_stream.first_frame_baseline, dtype=np.float64)),
                "max": float(np.max(right_stream.first_frame_baseline)),
            },
        },
        "instruction": resolved_instruction,
        "instruction_source": instruction_source,
    }
    if input_mode == "raw":
        metadata["upstream_raw_rule"] = {
            "stage1_source": SPARK_RAW_STAGE1_CONVERTER,
            "stage2_source": SPARK_RAW_STAGE2_CONVERTER,
            "benchmark": "spark0_real_bench",
            "state_time_alignment": "index_aligned_to_cam_head",
            "eef_input_layout": "[x,y,z,qw,qx,qy,qz] source WXYZ",
            "eef_output_frame": "spark_robot_base_world Link7",
            "eef_frame_transform": (
                "R=Rz(+pi/2)@Rsrc@XROBOT_FLANGE_OFFSET; "
                "p=Rz(+pi/2)@psrc+robot_base_translation+side Fix_Base world-x"
            ),
            "spark_real_to_robot_base_rotation": (
                SPARK_REAL_TO_ROBOT_BASE_R.tolist()
            ),
            "robot_base_translation_m": ROBOT_BASE_TRANSLATION.tolist(),
            "fix_base_offset_world_x_m": FIX_BASE_OFFSET_WORLD_X_M,
            "xrobot_flange_offset": XROBOT_FLANGE_OFFSET.tolist(),
            "camera_preset": RAW_CAMERA_PRESET,
            "camera_world_pose_wxyz": RAW_CAMERA_EGO_POSE_WXYZ.tolist(),
            "camera_coordinate_frame": "spark_robot_base_world",
            "camera_transform": "none; preset is already expressed in Spark world",
        }
    replay_buffer.root.attrs.update(metadata)

    if write_sidecar:
        metadata_path = output_zarr.parent / f"{output_zarr.stem}_conversion.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        metadata["metadata_json"] = str(metadata_path)
    if save_preview:
        if preview_dir is None:
            preview_dir = output_zarr.parent / f"{output_zarr.stem}_preview"
        preview_png, preview_summary = _save_preview(
            preview_dir,
            decoded_images["camera_ego_rgb"][kept_indices],
            left_grid[kept_indices],
            right_grid[kept_indices],
            left_sum[kept_indices],
            right_sum[kept_indices],
            np.arange(output_frame_count, dtype=np.float64) / target_fps,
            metadata,
        )
        metadata["preview_png"] = str(preview_png)
        metadata["preview_summary"] = str(preview_summary)
    return metadata


def convert_episode(
    input_hdf5: Path,
    output_zarr: Path,
    *,
    instruction: str | None = None,
    target_fps: float = 30.0,
    image_size: int = 224,
    min_segment_frames: int = 32,
    preview_dir: Path | None = None,
    overwrite: bool = False,
    save_preview: bool = True,
    write_sidecar: bool = True,
    joint_only_120d: bool = False,
) -> dict[str, Any]:
    """Production converter: accept only curated row-aligned 15/30 Hz schema."""
    input_hdf5 = input_hdf5.resolve()
    output_zarr = output_zarr.resolve()
    if not input_hdf5.is_file():
        raise FileNotFoundError(input_hdf5)
    if not _is_allowed_row_aligned_fps(target_fps):
        raise ValueError(
            "row-aligned conversion forbids resampling; "
            f"target_fps must be 15 or 30, got {target_fps}"
        )
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if min_segment_frames < 1:
        raise ValueError("min_segment_frames must be at least 1")

    # Validate the source contract before touching an existing output.  This
    # prevents an accidental --overwrite with a raw/60 Hz source from deleting
    # a valid result and then failing.
    with h5py.File(input_hdf5, "r") as input_file:
        input_mode = _detect_input_mode(
            input_file, joint_only_120d=joint_only_120d
        )
        if input_mode != "standardized_30hz":
            raise ValueError(
                "production conversion accepts only the curated standardized "
                "30 Hz HDF5 schema; raw/processed and embedded 60 Hz streams "
                f"are forbidden (detected {input_mode!r})"
            )

    if output_zarr.exists():
        if not overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {output_zarr}")
        shutil.rmtree(output_zarr)

    with h5py.File(input_hdf5, "r") as input_file:
        return _convert_standardized_30hz_episode(
            input_file,
            input_hdf5,
            output_zarr,
            instruction=instruction,
            target_fps=target_fps,
            image_size=image_size,
            min_segment_frames=min_segment_frames,
            preview_dir=preview_dir,
            save_preview=save_preview,
            write_sidecar=write_sidecar,
            joint_only_120d=joint_only_120d,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-hdf5", type=Path, required=True)
    parser.add_argument("--output-zarr", type=Path, required=True)
    parser.add_argument(
        "--instruction",
        default=None,
        help="optional instruction override; otherwise reuse the HDF5 instruction",
    )
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--min-segment-frames", type=int, default=32)
    parser.add_argument("--preview-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--no-sidecar", action="store_true")
    parser.add_argument(
        "--joint-only-120d",
        action="store_true",
        help=(
            "keep FTP-1's 120-D container but omit all wrist/head pose arrays; "
            "the upstream loader then supervises only arm and hand joints"
        ),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    cv2.setNumThreads(1)
    metadata = convert_episode(
        args.input_hdf5,
        args.output_zarr,
        instruction=args.instruction,
        target_fps=args.target_fps,
        image_size=args.image_size,
        min_segment_frames=args.min_segment_frames,
        preview_dir=args.preview_dir,
        overwrite=args.overwrite,
        save_preview=not args.no_preview,
        write_sidecar=not args.no_sidecar,
        joint_only_120d=args.joint_only_120d,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
