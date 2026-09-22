from __future__ import annotations

import sys
from pathlib import Path

import pytest


POLICY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY_DIR / "ftp1-policy" / "src"))

from openpi.ftp1_image_keys import (  # noqa: E402
    SPARK_MOXIAN_THREE_VIEW_IMAGE_KEYS,
    parse_used_image_keys,
    select_episode_image_keys,
)


def test_parse_all_or_empty_keeps_multiview() -> None:
    assert parse_used_image_keys("") is None
    assert parse_used_image_keys("all") is None
    assert parse_used_image_keys("*") is None
    assert parse_used_image_keys(",".join(SPARK_MOXIAN_THREE_VIEW_IMAGE_KEYS)) == SPARK_MOXIAN_THREE_VIEW_IMAGE_KEYS


def test_explicit_three_view_selection_preserves_token_order() -> None:
    keys = {
        "camera_ego_rgb",
        "left_wrist_camera_rgb",
        "right_wrist_camera_rgb",
        "left_hand_joints",
    }
    assert select_episode_image_keys(keys, SPARK_MOXIAN_THREE_VIEW_IMAGE_KEYS) == list(
        SPARK_MOXIAN_THREE_VIEW_IMAGE_KEYS
    )
    assert select_episode_image_keys(keys, None) == [
        "camera_ego_rgb",
        "right_wrist_camera_rgb",
        "left_wrist_camera_rgb",
    ]


@pytest.mark.parametrize("missing", SPARK_MOXIAN_THREE_VIEW_IMAGE_KEYS)
def test_explicit_three_view_selection_fails_if_any_view_is_missing(missing: str) -> None:
    available = set(SPARK_MOXIAN_THREE_VIEW_IMAGE_KEYS) - {missing}
    with pytest.raises(ValueError, match=missing):
        select_episode_image_keys(available, SPARK_MOXIAN_THREE_VIEW_IMAGE_KEYS)
