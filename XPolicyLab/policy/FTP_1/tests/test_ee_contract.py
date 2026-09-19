from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest


POLICY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY_DIR))
os.environ["FTP1_TEST_MODE"] = "1"

from model import Model, _SIDE_SLICES, _matrix_to_pose9, _pose7_to_matrix  # noqa: E402


HAND_INDICES = [
    7, 12, 22, 17, 1, 6, 11, 21, 16, 26,
    8, 13, 23, 18, 2, 9, 14, 24, 19, 3,
]


def _make_model(**overrides) -> Model:
    config = {
        "env_cfg_type": "tianji_marvin_wuji",
        "test_action_horizon": 33,
        "action_start_index": 1,
        "execute_horizon": 32,
        "left_hand_joint_indices": HAND_INDICES,
        "right_hand_joint_indices": HAND_INDICES,
        "proprioception_pose_rep": "absolute",
        "proprioception_joint_rep": "absolute",
        "action_pose_rep": "absolute",
        "action_joint_rep": "relative",
        "camera_map": {"camera_ego_rgb_0": ["cam_head"]},
        "required_cameras": ["camera_ego_rgb_0"],
    }
    config.update(overrides)
    return Model(config)


def _observation() -> dict:
    return {
        "instruction": "Build a tower with the blocks.",
        "vision": {"cam_head": np.zeros((224, 224, 3), dtype=np.uint8)},
        "state": {
            "left_ee_pose": np.array([0.3, 0.2, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "right_ee_pose": np.array([0.3, -0.2, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "left_ee_joint_state": np.arange(20, dtype=np.float32) / 20.0,
            "right_ee_joint_state": np.arange(20, dtype=np.float32) / 10.0,
        },
    }


def test_world_ee_state_mask_and_action_decode() -> None:
    model = _make_model()
    observation = _observation()
    packed, context, action_mask = model._build_state(observation)

    assert model.action_type == "ee"
    assert packed.shape == (1, 120)
    assert np.count_nonzero(action_mask) == 58
    assert np.count_nonzero(action_mask[9:16]) == 0
    assert np.count_nonzero(action_mask[57:64]) == 0
    for side, prefix in (("left", "left_"), ("right", "right_")):
        slices = _SIDE_SLICES[side]
        expected_pose9 = _matrix_to_pose9(_pose7_to_matrix(observation["state"][f"{prefix}ee_pose"]))
        assert np.allclose(packed[0, slice(*slices["pose"])], expected_pose9)
        assert np.count_nonzero(action_mask[slice(*slices["pose"])]) == 9
        assert np.count_nonzero(action_mask[slice(*slices["hand"])]) == 20

    encoded = model._encode_observation(observation)
    assert set(encoded["images"]) == {"camera_ego_rgb_0"}
    assert encoded["prompt"] == "Build a tower with the blocks."

    left_target = np.array([0.4, 0.25, 0.55, 0.9238795, 0.0, 0.3826834, 0.0], dtype=np.float32)
    right_target = np.array([0.4, -0.25, 0.55, 0.9238795, 0.0, -0.3826834, 0.0], dtype=np.float32)
    actions = np.zeros((33, 120), dtype=np.float32)
    hand_delta = np.arange(20, dtype=np.float32) / 100.0
    for side, target in (("left", left_target), ("right", right_target)):
        slices = _SIDE_SLICES[side]
        actions[:, slice(*slices["pose"])] = _matrix_to_pose9(_pose7_to_matrix(target))
        actions[:, slices["hand"][0] + np.asarray(HAND_INDICES)] = hand_delta

    decoded = model._decode_actions(actions, context)
    assert len(decoded) == 32
    for action in decoded:
        assert set(action) == {
            "left_ee_pose", "left_ee_joint_state", "right_ee_pose", "right_ee_joint_state"
        }
        assert np.allclose(action["left_ee_pose"], left_target, atol=1e-6)
        assert np.allclose(action["right_ee_pose"], right_target, atol=1e-6)
        assert np.allclose(
            action["left_ee_joint_state"], observation["state"]["left_ee_joint_state"] + hand_delta
        )
        assert np.allclose(
            action["right_ee_joint_state"], observation["state"]["right_ee_joint_state"] + hand_delta
        )


def test_execute_horizon_rejects_full_chunk_after_start_index() -> None:
    with pytest.raises(ValueError, match="execute_horizon must be in"):
        _make_model(execute_horizon=33)


def test_instruction_aliases_map_to_training_strings() -> None:
    model = _make_model()
    canonical = "Build a tower with the blocks."
    assert model._prompt({"instruction": canonical}) == canonical
    assert model._prompt({"instruction": "Build tower"}) == canonical
    assert model._prompt({"instruction": "build_tower"}) == canonical
    assert model._prompt({"instruction": "Wipe the blackboard clean."}) == "Wipe the blackboard clean."
    assert model._prompt({"instruction": "Pack the shoes into the box."}) == "Pack the shoes into the box."
    assert model._prompt({"instruction": "pask_shoes_into_box"}) == "Pack the shoes into the box."
    assert model._prompt({"instruction": "Place the phone in the designated location."}) == (
        "Place the phone in the designated location."
    )
    assert model._prompt({"instruction": "stack_bowls"}) == "Stack the bowls on top of one another."
    with pytest.raises(ValueError, match="must match training"):
        model._prompt({"instruction": "Do an unknown task."})


def test_negative_hand_joints_are_not_clamped() -> None:
    model = _make_model(action_joint_rep="absolute")
    observation = _observation()
    observation["state"]["left_ee_joint_state"] = np.full(20, -0.4, dtype=np.float32)
    observation["state"]["right_ee_joint_state"] = np.full(20, -0.7, dtype=np.float32)
    _, context, _ = model._build_state(observation)

    actions = np.zeros((33, 120), dtype=np.float32)
    left_hand = np.linspace(-0.698, -0.1, 20, dtype=np.float32)
    right_hand = np.linspace(-0.725, -0.05, 20, dtype=np.float32)
    actions[:, 64 + np.asarray(HAND_INDICES)] = left_hand
    actions[:, 16 + np.asarray(HAND_INDICES)] = right_hand

    decoded = model._decode_actions(actions, context)
    assert np.allclose(decoded[0]["left_ee_joint_state"], left_hand)
    assert np.allclose(decoded[0]["right_ee_joint_state"], right_hand)
    assert float(decoded[0]["left_ee_joint_state"].min()) < 0.0
    assert float(decoded[0]["right_ee_joint_state"].min()) < 0.0


def test_joint_control_is_rejected() -> None:
    with pytest.raises(ValueError, match="action_type must be 'ee'"):
        _make_model(action_type="joint")
