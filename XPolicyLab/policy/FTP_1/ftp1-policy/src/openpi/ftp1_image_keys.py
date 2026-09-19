"""Select which Zarr RGB streams FTP-1 should load for training."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

DEFAULT_FTP1_IMAGE_KEYS: tuple[str, ...] = (
    "camera_main_rgb",
    "camera_ego_rgb",
    "right_wrist_camera_rgb",
    "left_wrist_camera_rgb",
)
HEAD_ONLY_IMAGE_KEYS: tuple[str, ...] = ("camera_ego_rgb",)


def parse_used_image_keys(value: str | Sequence[str] | None) -> tuple[str, ...] | None:
    """Return an explicit camera allowlist, or None to use every present default camera.

    Empty / ``all`` / ``*`` keep the historical multi-view behavior: load every
    default key that exists in the episode. An explicit list is fail-closed if
    any requested key is missing later.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"all", "*"}:
            return None
        keys = tuple(part.strip() for part in text.split(",") if part.strip())
    else:
        keys = tuple(str(part).strip() for part in value if str(part).strip())
    unknown = [key for key in keys if key not in DEFAULT_FTP1_IMAGE_KEYS]
    if unknown:
        raise ValueError(
            f"unsupported image keys {unknown}; allowed={list(DEFAULT_FTP1_IMAGE_KEYS)}"
        )
    return keys or None


def select_episode_image_keys(
    data_key_set: Iterable[str],
    used_image_keys: tuple[str, ...] | None,
) -> list[str]:
    """Pick the RGB arrays that this episode should feed the model."""
    available = set(data_key_set)
    if used_image_keys is None:
        selected = [key for key in DEFAULT_FTP1_IMAGE_KEYS if key in available]
    else:
        missing = [key for key in used_image_keys if key not in available]
        if missing:
            raise ValueError(f"requested image keys missing from zarr: {missing}")
        selected = list(used_image_keys)
    if not selected:
        raise ValueError(
            "no usable RGB stream; at least one of "
            f"{list(DEFAULT_FTP1_IMAGE_KEYS)} must exist"
        )
    return selected
