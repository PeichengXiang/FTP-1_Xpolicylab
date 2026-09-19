from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


POLICY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY_DIR))
os.environ["FTP1_TEST_MODE"] = "1"

from model import Model, _FTP1_IDENTITY_POSE9, _SIDE_SLICES  # noqa: E402
from data_scripts.tactile_460_to_320 import convert_tactile_460_to_320  # noqa: E402


HAND_INDICES = [
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
]


def _make_model() -> Model:
    return Model(
        {
            "env_cfg_type": "tianji_marvin_wuji",
            "action_type": "joint",
            "left_hand_joint_indices": HAND_INDICES,
            "right_hand_joint_indices": HAND_INDICES,
            "proprioception_pose_rep": "relative",
            "proprioception_joint_rep": "absolute",
            "action_pose_rep": "relative",
            "action_joint_rep": "relative",
            "camera_map": {
                "camera_ego_rgb_0": ["cam_head"],
                "left_wrist_camera_rgb_0": ["cam_left_wrist"],
                "right_wrist_camera_rgb_0": ["cam_right_wrist"],
            },
            "required_cameras": [
                "camera_ego_rgb_0",
                "left_wrist_camera_rgb_0",
                "right_wrist_camera_rgb_0",
            ],
            "tactile_function_areas": {
                "right_tactile_fingertip": [0, 1, 2, 3, 4],
                "right_tactile_palm": [5],
                "left_tactile_fingertip": [24, 25, 26, 27, 28],
                "left_tactile_palm": [29],
            },
            "tactile_sensors": {
                "right_tactile_fingertip": "MoxianTactileGlove460",
                "right_tactile_palm": "MoxianTactileGlove460",
                "left_tactile_fingertip": "MoxianTactileGlove460",
                "left_tactile_palm": "MoxianTactileGlove460",
            },
        }
    )


def test_joint_obs_and_relative_action_round_trip() -> None:
    model = _make_model()
    left_arm = np.arange(7, dtype=np.float32) + 10.0
    right_arm = np.arange(7, dtype=np.float32) + 20.0
    left_hand = np.arange(20, dtype=np.float32) + 30.0
    right_hand = np.arange(20, dtype=np.float32) + 50.0
    observation = {
        "instruction": "Build tower",
        "vision": {
            "cam_head": np.zeros((224, 224, 3), dtype=np.uint8),
            "cam_left_wrist": np.zeros((224, 224, 3), dtype=np.uint8),
            "cam_right_wrist": np.zeros((224, 224, 3), dtype=np.uint8),
        },
        "state": {
            "left_arm_joint_state": left_arm,
            "left_ee_joint_state": left_hand,
            "right_arm_joint_state": right_arm,
            "right_ee_joint_state": right_hand,
        },
        "tactile": {
            "left_tactile_palm": np.zeros((15, 16), dtype=np.float32),
            "left_tactile_fingertip": np.zeros((5, 4, 4), dtype=np.float32),
            "right_tactile_palm": np.zeros((15, 16), dtype=np.float32),
            "right_tactile_fingertip": np.zeros((5, 4, 4), dtype=np.float32),
        },
    }

    packed, context, action_mask = model._build_state(observation)
    assert packed.shape == (1, 120)
    assert np.array_equal(packed[0, slice(*_SIDE_SLICES["left"]["pose"])], _FTP1_IDENTITY_POSE9)
    assert np.array_equal(packed[0, slice(*_SIDE_SLICES["right"]["pose"])], _FTP1_IDENTITY_POSE9)
    assert np.array_equal(packed[0, 57:64], left_arm)
    assert np.array_equal(packed[0, 9:16], right_arm)
    assert np.array_equal(packed[0, 64 + np.asarray(HAND_INDICES)], left_hand)
    assert np.array_equal(packed[0, 16 + np.asarray(HAND_INDICES)], right_hand)
    assert int(action_mask.sum()) == 72
    assert model._prompt(observation) == "Build tower."

    encoded = model._encode_observation(observation)
    assert set(encoded["images"]) == {
        "camera_ego_rgb_0",
        "left_wrist_camera_rgb_0",
        "right_wrist_camera_rgb_0",
    }
    assert encoded["tactiles"]["left_tactile_palm"].shape == (1, 1, 15, 16)
    assert encoded["tactiles"]["left_tactile_fingertip"].shape == (1, 5, 4, 4)
    assert encoded["tactile_function_areas"]["left_tactile_palm"] == [29]

    action_delta = np.zeros((2, 120), dtype=np.float32)
    left_arm_delta = np.arange(7, dtype=np.float32) / 100.0
    right_arm_delta = np.arange(7, dtype=np.float32) / 50.0
    left_hand_delta = np.arange(20, dtype=np.float32) / 40.0
    right_hand_delta = np.arange(20, dtype=np.float32) / 20.0
    action_delta[:, 57:64] = left_arm_delta
    action_delta[:, 9:16] = right_arm_delta
    action_delta[:, 64 + np.asarray(HAND_INDICES)] = left_hand_delta
    action_delta[:, 16 + np.asarray(HAND_INDICES)] = right_hand_delta

    decoded = model._decode_actions(action_delta, context)
    assert len(decoded) == 2
    for action in decoded:
        assert set(action) == {
            "left_arm_joint_state",
            "left_ee_joint_state",
            "right_arm_joint_state",
            "right_ee_joint_state",
        }
        assert np.allclose(action["left_arm_joint_state"], left_arm + left_arm_delta)
        assert np.allclose(action["right_arm_joint_state"], right_arm + right_arm_delta)
        assert np.allclose(action["left_ee_joint_state"], left_hand + left_hand_delta)
        assert np.allclose(action["right_ee_joint_state"], right_hand + right_hand_delta)


def test_tactile_spatial_map() -> None:
    grid = np.arange(460, dtype=np.float32).reshape(23, 20)
    left = convert_tactile_460_to_320(grid.reshape(-1), "left")
    right = convert_tactile_460_to_320(grid.reshape(-1), "right")
    assert left["left_tactile_palm"].shape == (15, 16)
    assert left["left_tactile_fingertip"].shape == (5, 4, 4)
    assert np.array_equal(left["left_tactile_palm"][0], grid[14, 4:20])
    assert np.array_equal(right["right_tactile_palm"][0], grid[14, 0:16])
    assert np.array_equal(left["left_tactile_fingertip"][0, 0], grid[22, 0:4])
    assert np.array_equal(right["right_tactile_fingertip"][0, 0], grid[22, 16:20])


if __name__ == "__main__":
    test_joint_obs_and_relative_action_round_trip()
    test_tactile_spatial_map()
    print("joint deployment contract: PASS")
