from __future__ import annotations

import sys
from pathlib import Path


POLICY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY_DIR / "ftp1-policy" / "src"))

from openpi.ftp1_image_keys import (  # noqa: E402
    HEAD_ONLY_IMAGE_KEYS,
    parse_used_image_keys,
    select_episode_image_keys,
)


def test_parse_all_or_empty_keeps_multiview() -> None:
    assert parse_used_image_keys("") is None
    assert parse_used_image_keys("all") is None
    assert parse_used_image_keys("*") is None
    assert parse_used_image_keys(["camera_ego_rgb"]) == HEAD_ONLY_IMAGE_KEYS


def test_single_view_selects_only_head_camera() -> None:
    keys = {
        "camera_ego_rgb",
        "left_wrist_camera_rgb",
        "right_wrist_camera_rgb",
        "left_hand_joints",
    }
    assert select_episode_image_keys(keys, HEAD_ONLY_IMAGE_KEYS) == ["camera_ego_rgb"]
    assert select_episode_image_keys(keys, None) == [
        "camera_ego_rgb",
        "right_wrist_camera_rgb",
        "left_wrist_camera_rgb",
    ]


def test_explicit_view_fails_if_missing() -> None:
    try:
        select_episode_image_keys({"left_wrist_camera_rgb"}, HEAD_ONLY_IMAGE_KEYS)
    except ValueError as exc:
        assert "camera_ego_rgb" in str(exc)
    else:
        raise AssertionError("missing head camera must fail")
