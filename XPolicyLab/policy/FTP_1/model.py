from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_batch_size, get_robot_action_dim_info


_POLICY_DIR = Path(__file__).resolve().parent
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"
_FTP1_ACTION_DIM = 120
_FTP1_ARM_JOINT_DIM = 7
_FTP1_HAND_JOINT_DIM = 32
_FTP1_GRIPPER_SLOT = 28
_SUPPORTED_ENV_CFG_TYPE = "tianji_marvin_wuji"

_SIDE_SLICES = {
    "right": {"pose": (0, 9), "arm": (9, 16), "hand": (16, 48)},
    "left": {"pose": (48, 57), "arm": (57, 64), "hand": (64, 96)},
}
_MOXIAN_TACTILE_KEYS = frozenset(
    {
        "right_tactile_fingertip",
        "right_tactile_palm",
        "left_tactile_fingertip",
        "left_tactile_palm",
    }
)
_FTP1_WORLD_EE_TRAINING_INSTRUCTIONS = (
    "Build a tower with the blocks.",
    "Wipe the blackboard clean.",
    "Insert the test tubes into the rack.",
    "Pack the shoes into the box.",
    "Place the phone in the designated location.",
    "Stack the bowls on top of one another.",
    "Stand the bottle upright.",
    "Transfer water from one container to the other using the dropper.",
)
_INSTRUCTION_ALIASES = {
    "build_tower": "Build a tower with the blocks.",
    "build tower": "Build a tower with the blocks.",
    "build tower.": "Build a tower with the blocks.",
    "build a tower": "Build a tower with the blocks.",
    "build a tower.": "Build a tower with the blocks.",
    "clean_blackboard": "Wipe the blackboard clean.",
    "clean blackboard": "Wipe the blackboard clean.",
    "clean blackboard.": "Wipe the blackboard clean.",
    "clean the blackboard": "Wipe the blackboard clean.",
    "clean the blackboard.": "Wipe the blackboard clean.",
    "wipe the blackboard clean": "Wipe the blackboard clean.",
    "insert_test_tubes": "Insert the test tubes into the rack.",
    "insert test tubes": "Insert the test tubes into the rack.",
    "insert test tubes.": "Insert the test tubes into the rack.",
    "insert the test tubes": "Insert the test tubes into the rack.",
    "insert the test tubes.": "Insert the test tubes into the rack.",
    "pask_shoes_into_box": "Pack the shoes into the box.",
    "pack_shoes_into_box": "Pack the shoes into the box.",
    "pask shoes into box": "Pack the shoes into the box.",
    "pask shoes into box.": "Pack the shoes into the box.",
    "pack shoes into box": "Pack the shoes into the box.",
    "pack shoes into box.": "Pack the shoes into the box.",
    "pack the shoes into the box": "Pack the shoes into the box.",
    "place_the_phone": "Place the phone in the designated location.",
    "place the phone": "Place the phone in the designated location.",
    "place the phone.": "Place the phone in the designated location.",
    "stack_bowls": "Stack the bowls on top of one another.",
    "stack bowls": "Stack the bowls on top of one another.",
    "stack bowls.": "Stack the bowls on top of one another.",
    "stack the bowls": "Stack the bowls on top of one another.",
    "stand_up_bottle": "Stand the bottle upright.",
    "stand up bottle": "Stand the bottle upright.",
    "stand up bottle.": "Stand the bottle upright.",
    "stand the bottle upright": "Stand the bottle upright.",
    "transfer_water_with_dropper": "Transfer water from one container to the other using the dropper.",
    "transfer water with dropper": "Transfer water from one container to the other using the dropper.",
    "transfer water with dropper.": "Transfer water from one container to the other using the dropper.",
    "transfer water with the dropper": "Transfer water from one container to the other using the dropper.",
    "transfer water with the dropper.": "Transfer water from one container to the other using the dropper.",
}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _first_config_value(data: Any, key: str) -> Any:
    if isinstance(data, dict):
        if key in data:
            return data[key]
        for value in data.values():
            found = _first_config_value(value, key)
            if found is not None:
                return found
    elif isinstance(data, list):
        for value in data:
            found = _first_config_value(value, key)
            if found is not None:
                return found
    return None


def _canonical_rep(value: Any, default: str) -> str:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"abs", "absolute"}:
        return "absolute"
    if normalized in {"rel", "relative", "delta", "mix"}:
        return normalized
    if normalized == "auto":
        return default
    raise ValueError(f"Unsupported FTP-1 representation: {value}")


def _quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = np.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0)) * 2.0
            scale = max(scale, 1e-8)
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif axis == 1:
            scale = np.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0.0)) * 2.0
            scale = max(scale, 1e-8)
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = np.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0.0)) * 2.0
            scale = max(scale, 1e-8)
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.array([w, x, y, z], dtype=np.float64)
    quaternion /= max(float(np.linalg.norm(quaternion)), 1e-8)
    return quaternion


def _pose7_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)
    if pose.size != 7:
        raise ValueError(f"Expected pose [x,y,z,qw,qx,qy,qz], got shape {pose.shape}")
    if not np.all(np.isfinite(pose)):
        raise ValueError("Pose contains NaN or infinity")
    if float(np.linalg.norm(pose[3:])) < 1e-8:
        raise ValueError("Pose quaternion has zero norm")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quaternion_wxyz_to_matrix(pose[3:])
    matrix[:3, 3] = pose[:3]
    return matrix


def _matrix_to_pose7(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.concatenate([matrix[:3, 3], _matrix_to_quaternion_wxyz(matrix[:3, :3])]).astype(np.float32)


def _matrix_to_pose9(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.concatenate([matrix[:3, 3], matrix[:2, :3].reshape(6)]).astype(np.float32)


def _pose9_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)
    if pose.size != 9:
        raise ValueError(f"Expected FTP-1 pose9d, got shape {pose.shape}")
    row1 = pose[3:6]
    row1 /= max(float(np.linalg.norm(row1)), 1e-8)
    row2 = pose[6:9] - float(np.dot(row1, pose[6:9])) * row1
    row2 /= max(float(np.linalg.norm(row2)), 1e-8)
    row3 = np.cross(row1, row2)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.stack([row1, row2, row3], axis=0)
    matrix[:3, 3] = pose[:3]
    return matrix


def _decode_pose_sequence(pose9: np.ndarray, base_matrix: np.ndarray, representation: str) -> np.ndarray:
    predicted = np.stack([_pose9_to_matrix(pose) for pose in pose9], axis=0)
    if representation == "absolute":
        decoded = predicted
    elif representation == "rel":
        decoded = predicted.copy()
        decoded[:, :3, 3] += base_matrix[:3, 3]
        decoded[:, :3, :3] = predicted[:, :3, :3] @ base_matrix[:3, :3]
    elif representation == "relative":
        decoded = base_matrix[None, ...] @ predicted
    elif representation == "delta":
        decoded = predicted.copy()
        decoded[:, :3, 3] = np.cumsum(predicted[:, :3, 3], axis=0) + base_matrix[:3, 3]
        current_rotation = base_matrix[:3, :3]
        for index in range(len(decoded)):
            current_rotation = predicted[index, :3, :3] @ current_rotation
            decoded[index, :3, :3] = current_rotation
    else:
        raise ValueError(f"Unsupported action_pose_rep: {representation}")
    return np.stack([_matrix_to_pose7(matrix) for matrix in decoded], axis=0)


def _ensure_hwc_image(image: Any) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3-D image, got {image.shape}")
    if image.shape[-1] == 3:
        image_hwc = image
    elif image.shape[0] == 3:
        image_hwc = np.transpose(image, (1, 2, 0))
    else:
        raise ValueError(f"Expected an RGB image with exactly 3 channels, got {image.shape}")

    if image_hwc.size == 0 or not np.issubdtype(image_hwc.dtype, np.number):
        raise ValueError(f"Expected a non-empty numeric RGB image, got dtype={image_hwc.dtype}")
    if not np.all(np.isfinite(image_hwc)):
        raise ValueError("RGB image contains NaN or infinity")
    image_min = float(image_hwc.min())
    image_max = float(image_hwc.max())
    if np.issubdtype(image_hwc.dtype, np.floating):
        image_hwc = image_hwc.astype(np.float32)
        if image_min >= 0.0 and image_max <= 1.0:
            image_hwc = image_hwc * 255.0
        elif image_min < 0.0 or image_max > 255.0:
            raise ValueError(
                f"Floating RGB image must be in [0,1] or [0,255], got [{image_min}, {image_max}]"
            )
        image_hwc = np.rint(image_hwc).astype(np.uint8)
    else:
        if image_min < 0.0 or image_max > 255.0:
            raise ValueError(f"Integer RGB image must be in [0,255], got [{image_min}, {image_max}]")
        image_hwc = image_hwc.astype(np.uint8, copy=False)
    # Msgpack-decoded arrays can be read-only, and CHW transposes can be
    # non-contiguous. Give the upstream wrapper a safe HWC uint8 buffer.
    return np.ascontiguousarray(image_hwc).copy()


class _TestWrapper:
    """Checkpoint-free wrapper used only by the documented wiring smoke test."""

    def __init__(self, action_horizon: int = 2):
        self.action_horizon = action_horizon

    def get_action_dim(self) -> int:
        return _FTP1_ACTION_DIM

    def get_state_dim(self) -> int:
        return _FTP1_ACTION_DIM

    def get_action_horizon(self) -> int:
        return self.action_horizon

    def infer(self, *, state: np.ndarray, **_: Any) -> np.ndarray:
        return np.repeat(np.asarray(state[-1:], dtype=np.float32), self.action_horizon, axis=0)


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = model_cfg
        self.action_type = str(model_cfg.get("action_type", "ee")).lower()
        if self.action_type != "ee":
            raise ValueError(
                "This FTP-1 checkpoint was trained with world-coordinate EE poses and hand joints; "
                f"action_type must be 'ee', got {self.action_type!r}"
            )

        self.env_cfg_type = model_cfg.get("env_cfg_type")
        if not self.env_cfg_type:
            raise ValueError("env_cfg_type is required for FTP_1")
        if self.env_cfg_type != _SUPPORTED_ENV_CFG_TYPE:
            raise ValueError(
                "The available Moxian checkpoint was trained for "
                f"env_cfg_type={_SUPPORTED_ENV_CFG_TYPE!r} only; got {self.env_cfg_type!r}"
            )
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        try:
            self.batch_size = get_batch_size(self.env_cfg_type)
        except (KeyError, FileNotFoundError):
            # Minimal real-robot env_cfg files commonly omit the simulator block.
            self.batch_size = int(model_cfg.get("batch_size", 1))
        self.num_arms = len(self.robot_action_dim_info["arm_dim"])
        if self.num_arms != 2:
            raise ValueError(
                f"The {_SUPPORTED_ENV_CFG_TYPE} checkpoint requires a dual-arm robot, got {self.num_arms} arm(s)"
            )
        if len(self.robot_action_dim_info["ee_dim"]) != self.num_arms:
            raise ValueError("arm_dim and ee_dim must contain the same number of entries")
        if any(int(dim) > _FTP1_ARM_JOINT_DIM for dim in self.robot_action_dim_info["arm_dim"]):
            raise ValueError("FTP-1 supports at most 7 arm joints per arm")
        if any(int(dim) > _FTP1_HAND_JOINT_DIM for dim in self.robot_action_dim_info["ee_dim"]):
            raise ValueError("FTP-1 supports at most 32 hand/end-effector joints per arm")

        self.single_arm_side = str(model_cfg.get("single_arm_side", "right")).lower()
        if self.single_arm_side not in _SIDE_SLICES:
            raise ValueError("single_arm_side must be left or right")

        self.test_mode = _as_bool(os.environ.get("FTP1_TEST_MODE"), default=False)
        train_config: dict[str, Any] = {}
        if self.test_mode:
            self.wrapper = _TestWrapper(int(model_cfg.get("test_action_horizon", 2)))
            checkpoint_root = None
        else:
            checkpoint_root = resolve_checkpoint_root(
                model_cfg,
                _CHECKPOINTS_DIR,
                policy_dir=_POLICY_DIR,
                explicit_keys=("checkpoint_path", "model_path", "ckpt_path"),
            )
            domain_name = model_cfg.get("domain_name")
            if not domain_name:
                raise ValueError("domain_name must match a checkpoint normalization domain")
            openpi_data_home = model_cfg.get("openpi_data_home")
            if openpi_data_home:
                openpi_data_home = Path(str(openpi_data_home)).expanduser().resolve()
                if not openpi_data_home.is_dir():
                    raise FileNotFoundError(f"OPENPI_DATA_HOME does not exist: {openpi_data_home}")
                os.environ["OPENPI_DATA_HOME"] = str(openpi_data_home)
            from openpi.policies import FTP1InferenceWrapper

            tactile_config = model_cfg.get("tactile_input_config_file") or None
            skip_normalization = _as_bool(model_cfg.get("skip_normalization"), default=False)
            self.wrapper = FTP1InferenceWrapper(
                checkpoint_dir=str(checkpoint_root),
                domain_name=str(domain_name),
                tactile_input_config_file=tactile_config,
                device=str(model_cfg.get("device", "cuda")),
                num_inference_steps=int(model_cfg.get("num_inference_steps", 10)),
                skip_normalization=skip_normalization,
            )
            if not skip_normalization and getattr(self.wrapper, "domain_norm_stats", None) is None:
                raise RuntimeError(
                    "Checkpoint inference requires domain normalization, but the selected checkpoint "
                    f"did not load stats for {domain_name!r}"
                )
            train_config_path = Path(self.wrapper.ckpt_dir) / "train_config.json"
            if train_config_path.exists():
                with train_config_path.open(encoding="utf-8") as file:
                    train_config = json.load(file)

        if int(self.wrapper.get_action_dim()) != _FTP1_ACTION_DIM:
            raise ValueError(
                f"This adapter expects FTP-1 unified action_dim={_FTP1_ACTION_DIM}, "
                f"checkpoint reports {self.wrapper.get_action_dim()}"
            )
        if int(self.wrapper.get_state_dim()) != _FTP1_ACTION_DIM:
            raise ValueError(
                f"This adapter expects FTP-1 unified state_dim={_FTP1_ACTION_DIM}, "
                f"checkpoint reports {self.wrapper.get_state_dim()}"
            )
        self.action_horizon = int(self.wrapper.get_action_horizon())
        if self.action_horizon < 1:
            raise ValueError(f"Checkpoint reports invalid action horizon {self.action_horizon}")

        self.proprioception_pose_rep = self._resolve_rep(
            "proprioception_pose_rep", train_config, default="absolute"
        )
        self.proprioception_joint_rep = self._resolve_rep(
            "proprioception_joint_rep", train_config, default="absolute"
        )
        self.action_pose_rep = self._resolve_rep("action_pose_rep", train_config, default="absolute")
        self.action_joint_rep = self._resolve_rep("action_joint_rep", train_config, default="absolute")

        joint_only_setting = model_cfg.get("joint_only_state_action", False)
        self.joint_only_state_action = False if str(joint_only_setting).lower() == "auto" else _as_bool(
            joint_only_setting
        )
        # joint_only_state_action is the EE-pose + hand-joint contract.  Arm-joint
        # slots stay zero, while world EE poses remain required and are read from
        # the HDF5-derived observation.
        if self.proprioception_pose_rep != "absolute" or self.action_pose_rep != "absolute":
            raise ValueError(
                "The real200_action32_15hz checkpoint requires absolute world-coordinate EE pose "
                f"representations, got proprioception={self.proprioception_pose_rep!r}, "
                f"action={self.action_pose_rep!r}"
            )
        self.training_instructions = tuple(
            model_cfg.get("training_instructions") or _FTP1_WORLD_EE_TRAINING_INSTRUCTIONS
        )

        self.camera_map = model_cfg.get("camera_map") or {
            "camera_ego_rgb_0": ["cam_head", "cam_third_view"],
        }
        self.required_cameras = set(model_cfg.get("required_cameras") or ["camera_ego_rgb_0"])
        image_size = model_cfg.get("image_size") or [224, 224]
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.input_color_order = str(model_cfg.get("input_color_order", "RGB")).upper()
        if self.input_color_order != "RGB":
            raise ValueError(
                "FTP-1 was trained on RGB images; input_color_order must be 'RGB', "
                f"got {self.input_color_order!r}"
            )
        self.action_start_index = int(model_cfg.get("action_start_index", 1))
        if self.action_start_index < 0 or self.action_start_index >= self.action_horizon:
            raise ValueError(
                f"action_start_index must be in [0,{self.action_horizon - 1}], "
                f"got {self.action_start_index}"
            )
        available_action_steps = self.action_horizon - self.action_start_index
        self.execute_horizon = model_cfg.get("execute_horizon")
        if self.execute_horizon is not None:
            self.execute_horizon = int(self.execute_horizon)
            if self.execute_horizon < 1 or self.execute_horizon > available_action_steps:
                raise ValueError(
                    f"execute_horizon must be in [1,{available_action_steps}] after "
                    f"action_start_index={self.action_start_index}, got {self.execute_horizon}"
                )
        self.use_action_mask = _as_bool(model_cfg.get("use_action_mask"), default=True)

        wrapper_model_config = getattr(self.wrapper, "model_config", None)
        checkpoint_uses_tactile = bool(getattr(wrapper_model_config, "use_tactile_input", False))
        configured_require_tactile = model_cfg.get("require_tactile", "auto")
        if str(configured_require_tactile).lower() != "auto":
            configured_value = _as_bool(configured_require_tactile)
            if configured_value != checkpoint_uses_tactile:
                raise ValueError(
                    "deploy.yml require_tactile disagrees with checkpoint use_tactile_input: "
                    f"configured={configured_value}, checkpoint={checkpoint_uses_tactile}"
                )
        self.require_tactile = checkpoint_uses_tactile
        self.tactile_contract = self._load_tactile_contract()
        if self.require_tactile and not self.test_mode and not self.tactile_contract:
            raise ValueError(
                "Checkpoint requires tactile input, but no domain tactile contract could be loaded from "
                "tactile_input_config_file.json."
            )

        self._obs_by_env_idx: dict[int, dict[str, Any]] = {}
        self._latest_env_idx_list: list[int] = []
        mode = "checkpoint-free test" if self.test_mode else f"checkpoint={checkpoint_root}"
        print(
            f"[FTP_1] initialized ({mode}, env={self.env_cfg_type}, action_type={self.action_type}, "
            f"joint_only_state_action={self.joint_only_state_action}, "
            f"pose_rep={self.action_pose_rep}, joint_rep={self.action_joint_rep}, "
            f"action_slice={self.action_start_index}:"
            f"{self.action_start_index + (self.execute_horizon or available_action_steps)})"
        )

    def _resolve_rep(self, key: str, train_config: dict[str, Any], default: str) -> str:
        configured = self.model_cfg.get(key, "auto")
        if configured is None or str(configured).strip().lower() == "auto":
            configured = _first_config_value(train_config, key)
        return _canonical_rep(configured, default)

    def _load_tactile_contract(self) -> dict[str, dict[str, Any]]:
        """Load the exact runtime tactile schema saved with the selected checkpoint."""
        if self.test_mode:
            return {}

        tactile_config_path = getattr(self.wrapper.model_config, "tactile_input_config_file", None)
        if not tactile_config_path:
            return {}
        tactile_config_path = Path(tactile_config_path)
        if not tactile_config_path.is_file():
            raise FileNotFoundError(f"Tactile input config not found: {tactile_config_path}")

        with tactile_config_path.open(encoding="utf-8") as file:
            tactile_configs = json.load(file)
        domain_name = str(self.model_cfg.get("domain_name", ""))
        domain_contract = tactile_configs.get(domain_name)
        if not isinstance(domain_contract, dict) or not domain_contract:
            raise KeyError(
                f"Tactile contract for domain {domain_name!r} is missing from {tactile_config_path}"
            )

        contract: dict[str, dict[str, Any]] = {}
        for key, entry in domain_contract.items():
            if not isinstance(entry, dict):
                raise TypeError(f"Invalid tactile contract entry for {key}: expected mapping")
            missing = {"shape", "function_areas", "sensor", "type"} - set(entry)
            if missing:
                raise KeyError(f"Tactile contract entry {key} is missing fields {sorted(missing)}")
            shape = tuple(int(value) for value in entry["shape"])
            function_areas = tuple(int(value) for value in entry["function_areas"])
            tactile_type = str(entry["type"]).lower()
            if len(shape) < 2 or shape[1] != len(function_areas):
                raise ValueError(
                    f"Tactile contract entry {key} has incompatible shape={shape} "
                    f"and function_areas={function_areas}"
                )
            if tactile_type != "matrix":
                raise ValueError(
                    f"This Moxian adapter supports matrix tactile inputs only; {key} uses {entry['type']!r}"
                )
            contract[str(key)] = {
                "shape": shape,
                "function_areas": function_areas,
                "sensor": str(entry["sensor"]),
                "type": tactile_type,
            }
        if set(contract) != _MOXIAN_TACTILE_KEYS:
            raise ValueError(
                "This Moxian adapter requires exactly four checkpoint tactile streams: "
                f"expected={sorted(_MOXIAN_TACTILE_KEYS)}, got={sorted(contract)}"
            )
        return contract

    def _side_specs(self) -> list[tuple[str, str]]:
        if self.num_arms == 1:
            return [(self.single_arm_side, "")]
        return [("left", "left_"), ("right", "right_")]

    def _hand_indices(self, side: str, dimension: int) -> np.ndarray:
        configured = self.model_cfg.get(f"{side}_hand_joint_indices")
        if configured is None:
            indices = [_FTP1_GRIPPER_SLOT] if dimension == 1 else list(range(dimension))
        else:
            indices = [int(index) for index in configured]
        if len(indices) != dimension:
            raise ValueError(
                f"{side}_hand_joint_indices has {len(indices)} entries, expected {dimension} for {self.env_cfg_type}"
            )
        if len(set(indices)) != len(indices) or any(index < 0 or index >= _FTP1_HAND_JOINT_DIM for index in indices):
            raise ValueError(f"Invalid {side}_hand_joint_indices: {indices}")
        return np.asarray(indices, dtype=np.int64)

    def _extract_image(self, observation: dict[str, Any], candidates: Any) -> np.ndarray | None:
        if isinstance(candidates, str):
            candidates = [candidates]
        vision = observation.get("vision", {})
        for candidate in candidates:
            if candidate not in vision:
                continue
            value = vision[candidate]
            if isinstance(value, dict):
                value = value.get("color", value.get("rgb"))
            if value is not None:
                return _ensure_hwc_image(value)
        return None

    def _build_images(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        for ftp_key, candidates in self.camera_map.items():
            image = self._extract_image(observation, candidates)
            if image is None:
                if ftp_key in self.required_cameras:
                    raise KeyError(f"Missing required camera {ftp_key}; checked {candidates}")
                continue
            if (image.shape[0], image.shape[1]) != self.image_size:
                import cv2

                image = cv2.resize(image, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)
            images[str(ftp_key)] = image
        return images

    def _prompt(self, observation: dict[str, Any]) -> str:
        prompt = observation.get("instruction", observation.get("instructions", ""))
        if isinstance(prompt, (list, tuple)):
            prompt = prompt[0] if prompt else ""
        prompt = " ".join(str(prompt).strip().split())
        if prompt in self.training_instructions:
            return prompt
        with_period = prompt if prompt.endswith(".") else f"{prompt}."
        if with_period in self.training_instructions:
            return with_period
        alias = _INSTRUCTION_ALIASES.get(prompt.lower()) or _INSTRUCTION_ALIASES.get(with_period.lower())
        if alias in self.training_instructions:
            return alias
        raise ValueError(
            f"FTP-1 instruction must match training; got {prompt!r}, "
            f"expected one of {list(self.training_instructions)}"
        )

    @staticmethod
    def _state_vector(state: dict[str, Any], key: str, dimension: int) -> np.ndarray:
        if key not in state:
            raise KeyError(f"Observation state is missing required key {key!r}")
        value = np.asarray(state[key], dtype=np.float32).reshape(-1)
        if value.size != dimension:
            raise ValueError(f"State {key} has {value.size} values, expected {dimension}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"State {key} contains NaN or infinity")
        return value

    def _build_state(self, observation: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
        packed = np.zeros(_FTP1_ACTION_DIM, dtype=np.float32)
        action_mask = np.zeros(_FTP1_ACTION_DIM, dtype=np.float32)
        state = observation.get("state")
        if not isinstance(state, dict):
            raise TypeError("Observation must provide a state mapping")
        context: dict[str, Any] = {"sides": {}}

        for arm_index, (side, prefix) in enumerate(self._side_specs()):
            slices = _SIDE_SLICES[side]
            arm_dim = int(self.robot_action_dim_info["arm_dim"][arm_index])
            hand_dim = int(self.robot_action_dim_info["ee_dim"][arm_index])
            hand_key = f"{prefix}ee_joint_state"
            # The world-EE dataset omits arm-joint arrays. Keep those unified
            # state/action slots at literal zero and masked out at runtime.
            arm = np.zeros(arm_dim, dtype=np.float32)
            hand = self._state_vector(state, hand_key, hand_dim)
            hand_indices = self._hand_indices(side, hand_dim)
            side_context: dict[str, Any] = {
                "arm": arm.copy(),
                "hand": hand.copy(),
                "hand_indices": hand_indices,
            }

            pose_value = state.get(f"{prefix}ee_pose", state.get(f"{prefix}tcp_pose"))
            if self.proprioception_pose_rep != "absolute":
                # The joint-training dataloader writes the SE(3) identity into each
                # relative wrist-state slot, even when no runtime ee pose is used.
                packed[slice(*slices["pose"])] = _matrix_to_pose9(np.eye(4, dtype=np.float64))
            elif pose_value is not None:
                base_pose_matrix = _pose7_to_matrix(np.asarray(pose_value))
                side_context["pose_matrix"] = base_pose_matrix
                packed[slice(*slices["pose"])] = _matrix_to_pose9(base_pose_matrix)
            else:
                raise KeyError(
                    f"Checkpoint uses absolute wrist proprioception; state must provide "
                    f"{prefix}ee_pose or {prefix}tcp_pose"
                )

            if self.proprioception_joint_rep == "absolute":
                packed[slices["arm"][0] : slices["arm"][0] + arm_dim] = arm
                hand_start = slices["hand"][0]
                packed[hand_start + hand_indices] = hand

            action_mask[slice(*slices["pose"])] = 1.0
            action_mask[slices["hand"][0] + hand_indices] = 1.0
            context["sides"][side] = side_context

        return packed[None, ...], context, action_mask

    @staticmethod
    def _tactile_value(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        for key in ("data", "value", "color", "image", "values"):
            if key in value:
                return value[key]
        raise KeyError(f"Could not find tactile payload in keys {sorted(value)}")

    def _build_tactiles(
        self, observation: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray] | None, dict[str, list[int]] | None, dict[str, str] | None]:
        section = observation.get("tactile", observation.get("touch", {}))
        mapping = self.model_cfg.get("tactile_map") or {}
        raw: dict[str, Any] = {}
        if mapping:
            for ftp_key, source_key in mapping.items():
                if isinstance(source_key, (list, tuple)):
                    selected = next((section[key] for key in source_key if key in section), None)
                else:
                    selected = section.get(source_key)
                if selected is not None:
                    raw[str(ftp_key)] = self._tactile_value(selected)
        elif isinstance(section, dict):
            raw = {str(key): self._tactile_value(value) for key, value in section.items()}

        expected_keys = set(self.tactile_contract)
        if self.require_tactile and expected_keys:
            actual_keys = set(raw)
            missing_keys = expected_keys - actual_keys
            unexpected_keys = actual_keys - expected_keys
            if missing_keys or unexpected_keys:
                raise KeyError(
                    "Runtime tactile keys do not match the checkpoint contract: "
                    f"missing={sorted(missing_keys)}, unexpected={sorted(unexpected_keys)}, "
                    f"expected={sorted(expected_keys)}"
                )

        if not raw:
            if self.require_tactile:
                raise KeyError(
                    "Checkpoint requires tactile input, but observation has no tactile/touch mapping. "
                    "Configure tactile_map, tactile_function_areas, and tactile_sensors in deploy.yml."
                )
            return None, None, None

        has_time_config = self.model_cfg.get("tactile_has_time_dim", False)
        single_area_keys = set(self.model_cfg.get("single_area_tactile_keys") or [])
        tactiles: dict[str, np.ndarray] = {}
        for key, value in raw.items():
            array = np.asarray(value)
            if key in single_area_keys or (array.ndim == 3 and array.shape[-1] in {1, 3}) or array.ndim <= 2:
                array = array[None, ...]
            has_time = has_time_config.get(key, False) if isinstance(has_time_config, dict) else has_time_config
            if not _as_bool(has_time):
                array = array[None, ...]
            array = array.astype(np.float32, copy=False)
            if not np.all(np.isfinite(array)):
                raise ValueError(f"Tactile input {key} contains NaN or infinity")
            if np.any(array < 0):
                raise ValueError(
                    f"Tactile input {key} contains negative pressure; expected max(raw-baseline, 0)"
                )
            tactiles[key] = np.ascontiguousarray(array).copy()

        deploy_areas = self.model_cfg.get("tactile_function_areas") or {}
        observation_areas = observation.get("tactile_function_areas") or {}
        deploy_sensors = self.model_cfg.get("tactile_sensors") or {}
        observation_sensors = observation.get("tactile_sensors") or {}
        function_areas: dict[str, list[int]] = {}
        sensors: dict[str, str] = {}
        for key, array in tactiles.items():
            expected = self.tactile_contract.get(key)
            if expected is None:
                areas = deploy_areas.get(key, observation_areas.get(key))
                function_areas[key] = [
                    int(value) for value in (areas if areas is not None else range(array.shape[1]))
                ]
                sensor = deploy_sensors.get(key, observation_sensors.get(key))
                if sensor is None:
                    raise KeyError(f"Missing tactile_sensors entry for {key}")
                sensors[key] = str(sensor)
                continue
            expected_shape = expected["shape"]
            expected_areas = [int(value) for value in expected["function_areas"]]
            expected_sensor = expected["sensor"]
            if array.shape != expected_shape:
                raise ValueError(
                    f"Tactile input {key} has shape {array.shape}, checkpoint expects {expected_shape}"
                )
            for source_name, source in (("deploy.yml", deploy_areas), ("observation", observation_areas)):
                if key in source:
                    declared_areas = [int(value) for value in source[key]]
                    if declared_areas != expected_areas:
                        raise ValueError(
                            f"Tactile input {key} declares function areas {declared_areas} in {source_name}, "
                            f"checkpoint expects {expected_areas}"
                        )
            for source_name, source in (
                ("deploy.yml", deploy_sensors),
                ("observation", observation_sensors),
            ):
                if key in source and str(source[key]) != expected_sensor:
                    raise ValueError(
                        f"Tactile input {key} declares sensor {source[key]!r} in {source_name}, "
                        f"checkpoint expects {expected_sensor!r}"
                    )
            function_areas[key] = expected_areas
            sensors[key] = expected_sensor
        return tactiles, function_areas, sensors

    def _encode_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        state, context, action_mask = self._build_state(observation)
        tactiles, tactile_function_areas, tactile_sensors = self._build_tactiles(observation)
        return {
            "images": self._build_images(observation),
            "state": state,
            "prompt": self._prompt(observation),
            "tactiles": tactiles,
            "tactile_function_areas": tactile_function_areas,
            "tactile_sensors": tactile_sensors,
            "action_mask": action_mask if self.use_action_mask else None,
            "context": context,
        }

    def update_obs(self, obs: dict[str, Any]) -> None:
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list: list[dict[str, Any]]) -> None:
        if not isinstance(obs_list, (list, tuple)) or not obs_list:
            raise ValueError("update_obs_batch expects a non-empty observation list")
        encoded_by_env_idx: dict[int, dict[str, Any]] = {}
        latest_env_idx_list: list[int] = []
        for index, observation in enumerate(obs_list):
            if not isinstance(observation, dict):
                raise TypeError(f"Observation {index} must be a mapping")
            env_idx = int(observation.get("env_idx", index))
            if env_idx in encoded_by_env_idx:
                raise ValueError(f"Duplicate env_idx={env_idx} in observation batch")
            encoded_by_env_idx[env_idx] = self._encode_observation(observation)
            latest_env_idx_list.append(env_idx)
        # Publish the new batch only after every observation validates, so one
        # malformed robot packet cannot leave a partially updated batch behind.
        self._obs_by_env_idx = encoded_by_env_idx
        self._latest_env_idx_list = latest_env_idx_list

    def _joint_actions_to_absolute(
        self, values: np.ndarray, base: np.ndarray, slots: np.ndarray | None = None
    ) -> np.ndarray:
        if self.action_joint_rep == "absolute":
            return values
        output = values.copy()
        if self.action_joint_rep == "mix" and slots is not None:
            relative_mask = slots != _FTP1_GRIPPER_SLOT
            output[:, relative_mask] += base[None, relative_mask]
        else:
            output += base[None, :]
        return output

    def _decode_actions(self, actions: np.ndarray, context: dict[str, Any]) -> list[dict[str, np.ndarray]]:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != _FTP1_ACTION_DIM:
            raise ValueError(f"FTP-1 returned unexpected action shape {actions.shape}")
        if actions.shape[0] < 1:
            raise ValueError("FTP-1 returned an empty action chunk")
        if not np.all(np.isfinite(actions)):
            raise ValueError("FTP-1 returned NaN or infinity; refusing to send actions to the robot")
        if actions.shape[0] != self.action_horizon:
            raise ValueError(
                f"FTP-1 returned {actions.shape[0]} action steps, checkpoint expects {self.action_horizon}"
            )
        action_end_index = self.action_horizon
        if self.execute_horizon is not None:
            action_end_index = self.action_start_index + self.execute_horizon
        actions = actions[self.action_start_index : action_end_index]

        decoded_by_side: dict[str, dict[str, np.ndarray]] = {}
        for side, _ in self._side_specs():
            slices = _SIDE_SLICES[side]
            side_context = context["sides"][side]
            hand_indices = side_context["hand_indices"]
            hand_values = actions[:, slices["hand"][0] + hand_indices]
            hand_values = self._joint_actions_to_absolute(hand_values, side_context["hand"], hand_indices)
            if "pose_matrix" not in side_context:
                raise KeyError(f"{side} ee_pose/tcp_pose is required for action_type=ee")
            pose9 = actions[:, slice(*slices["pose"])]
            decoded: dict[str, np.ndarray] = {
                "hand": hand_values,
                "pose": _decode_pose_sequence(pose9, side_context["pose_matrix"], self.action_pose_rep),
            }
            decoded_by_side[side] = decoded

        result: list[dict[str, np.ndarray]] = []
        for step in range(len(actions)):
            action: dict[str, np.ndarray] = {}
            for side, prefix in self._side_specs():
                decoded = decoded_by_side[side]
                action[f"{prefix}ee_pose"] = decoded["pose"][step].astype(np.float32)
                action[f"{prefix}ee_joint_state"] = decoded["hand"][step].astype(np.float32)
            if any(not np.all(np.isfinite(value)) for value in action.values()):
                raise ValueError("Decoded FTP-1 action contains NaN or infinity")
            result.append(action)
        return result

    def _infer_one(self, encoded: dict[str, Any]) -> list[dict[str, np.ndarray]]:
        action = self.wrapper.infer(
            images=encoded["images"],
            state=encoded["state"],
            prompt=encoded["prompt"],
            tactiles=encoded["tactiles"],
            tactile_function_areas=encoded["tactile_function_areas"],
            tactile_sensors=encoded["tactile_sensors"],
            action_mask=encoded["action_mask"],
        )
        return self._decode_actions(action, encoded["context"])

    def get_action(self) -> list[dict[str, np.ndarray]]:
        if not self._latest_env_idx_list:
            raise RuntimeError("Call update_obs before get_action")
        return self._infer_one(self._obs_by_env_idx[self._latest_env_idx_list[0]])

    def get_action_batch(self, env_idx_list: list[int] | None = None) -> list[list[dict[str, np.ndarray]]]:
        selected = self._latest_env_idx_list if env_idx_list is None else [int(index) for index in env_idx_list]
        if not selected:
            selected = list(range(self.batch_size))
        missing = [index for index in selected if index not in self._obs_by_env_idx]
        if missing:
            raise KeyError(f"No observation stored for env indices {missing}")
        return [self._infer_one(self._obs_by_env_idx[index]) for index in selected]

    def reset(self) -> None:
        self._obs_by_env_idx.clear()
        self._latest_env_idx_list.clear()
