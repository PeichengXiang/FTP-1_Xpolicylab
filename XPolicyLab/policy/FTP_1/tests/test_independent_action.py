from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np


POLICY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY_DIR / "data_scripts"))
sys.path.insert(0, str(POLICY_DIR / "ftp1-policy" / "src"))

from convert_spark_moxian_dataset import (  # noqa: E402
    REQUIRED_INDEPENDENT_ACTION_KEYS,
    _zarr_has_independent_hdf5_action,
)
from parse_data_spark_moxian import (  # noqa: E402
    STANDARDIZED_30HZ_ACTION_KEYS,
    STANDARDIZED_30HZ_REQUIRED_KEYS,
    _assert_action_datasets_are_independent,
    _load_standardized_action_arrays,
)
from openpi.ftp1_action_array_keys import resolve_ftp1_action_array_key  # noqa: E402


class _Array:
    def __init__(self, values: np.ndarray) -> None:
        self._values = np.asarray(values)

    def __getitem__(self, item):
        return self._values[item]


def test_standardized_contract_requires_hdf5_action_group() -> None:
    assert STANDARDIZED_30HZ_ACTION_KEYS <= STANDARDIZED_30HZ_REQUIRED_KEYS
    for key in (
        "action/left_ee_poses",
        "action/right_ee_poses",
        "action/left_ee_joint_states",
        "action/right_ee_joint_states",
    ):
        assert key in STANDARDIZED_30HZ_REQUIRED_KEYS
    assert REQUIRED_INDEPENDENT_ACTION_KEYS == {
        "left_wrist_pose_action",
        "right_wrist_pose_action",
        "left_hand_joints_action",
        "right_hand_joints_action",
    }


def test_loader_prefers_independent_action_arrays() -> None:
    keys = {
        "left_hand_joints",
        "left_hand_joints_action",
        "left_wrist_pose",
        "left_wrist_pose_action",
    }
    assert resolve_ftp1_action_array_key(keys, "left_hand_joints") == "left_hand_joints_action"
    assert resolve_ftp1_action_array_key(keys, "left_wrist_pose") == "left_wrist_pose_action"
    assert resolve_ftp1_action_array_key({"left_hand_joints"}, "left_hand_joints") == (
        "left_hand_joints"
    )


def test_converter_reads_action_not_next_state() -> None:
    identity = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    shifted = np.array([[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    state_hand = np.full((1, 20), 1.09, dtype=np.float32)
    action_hand = np.full((1, 20), -0.12, dtype=np.float32)
    handle = {
        "state/left_ee_joint_states": _Array(state_hand),
        "state/right_ee_joint_states": _Array(state_hand),
        "state/left_ee_poses": _Array(identity),
        "state/right_ee_poses": _Array(identity),
        "action/left_ee_joint_states": _Array(action_hand),
        "action/right_ee_joint_states": _Array(action_hand),
        "action/left_ee_poses": _Array(shifted),
        "action/right_ee_poses": _Array(shifted),
    }

    arrays = _load_standardized_action_arrays(handle, 1, joint_only_120d=True)

    assert np.allclose(arrays["left_hand_joints_action"], -0.12)
    assert np.allclose(arrays["right_hand_joints_action"], -0.12)
    assert not np.allclose(arrays["left_hand_joints_action"], state_hand)
    assert np.allclose(arrays["left_wrist_pose_action"][0, :3], [0.1, 0.2, 0.3])
    assert "left_arm_joints_action" not in arrays


def test_converter_accepts_original_action_values_equal_to_next_state(tmp_path: Path) -> None:
    state_hand = np.linspace(0.1, 0.3, 40, dtype=np.float64).reshape(2, 20)
    state_pose = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    source = tmp_path / "episode.hdf5"
    with h5py.File(source, "w") as handle:
        for side in ("left", "right"):
            handle.create_dataset(f"state/{side}_ee_joint_states", data=state_hand)
            handle.create_dataset(f"state/{side}_ee_poses", data=state_pose)
            handle.create_dataset(
                f"action/{side}_ee_joint_states",
                data=np.vstack([state_hand[1:], state_hand[-1:]]),
            )
            handle.create_dataset(
                f"action/{side}_ee_poses",
                data=np.vstack([state_pose[1:], state_pose[-1:]]),
            )
        _assert_action_datasets_are_independent(handle, 2)


def test_converter_rejects_action_hard_link_to_state(tmp_path: Path) -> None:
    source = tmp_path / "aliased_episode.hdf5"
    with h5py.File(source, "w") as handle:
        for side in ("left", "right"):
            state_hand = handle.create_dataset(
                f"state/{side}_ee_joint_states", data=np.zeros((2, 20))
            )
            state_pose = handle.create_dataset(
                f"state/{side}_ee_poses", data=np.zeros((2, 7))
            )
            handle[f"action/{side}_ee_joint_states"] = state_hand
            handle[f"action/{side}_ee_poses"] = state_pose
        try:
            _assert_action_datasets_are_independent(handle, 2)
        except ValueError as exc:
            assert "aliases" in str(exc)
        else:
            raise AssertionError("hard-linked state/action datasets must be rejected")


def test_old_next_state_zarr_is_stale() -> None:
    assert _zarr_has_independent_hdf5_action(Path("/tmp/ftp1-missing-independent-action.zarr")) is False
