"""Build an FTP-1 ``obs["tactile"]`` entry from one processed 460-channel glove."""

from __future__ import annotations

import numpy as np


def convert_tactile_460_to_320(
    pressure_460: np.ndarray,
    hand: str,
) -> dict[str, np.ndarray]:
    """Convert one upstream-processed glove frame into FTP-1 entries.

    The raw final dimension is a row-major ``23 x 20`` Moxian grid.  The
    packed order matches the real-data conversion pipeline:

    * palm: rows 14..0, using 16 columns (240 values);
    * fingertips: rows 22..19, grouped as thumb/index/middle/ring/pinky,
      using a 4-column block for each finger (5 x 16 values).

    Left and right gloves use mirrored column blocks.  Values from rows
    15..18 and the four non-palmar columns in rows 0..14 are discarded.

    Args:
        pressure_460: One tactile frame shaped exactly ``(460,)``. Any temporal
            calibration or episode-baseline subtraction belongs upstream; this
            function performs only the fixed spatial selection/reordering and
            non-negative clamp.
        hand: ``"left"`` or ``"right"``.

    Returns:
        The two entries to merge into ``obs["tactile"]``.  Their keys are
        ``{hand}_tactile_palm`` and ``{hand}_tactile_fingertip``; their
        ``float32`` values have shapes ``(15, 16)`` and ``(5, 4, 4)``.
        Negative values are clamped to zero, matching the data conversion.
    """
    side = str(hand).lower()
    if side not in {"left", "right"}:
        raise ValueError(f"hand must be 'left' or 'right', got {hand!r}")

    values = np.asarray(pressure_460, dtype=np.float32)
    if values.shape != (460,):
        raise ValueError(f"pressure_460 must have shape (460,), got {values.shape}")

    palm_columns = (range(4, 20) if side == "left" else range(0, 16))
    finger_starts = (
        (0, 4, 8, 12, 16) if side == "left" else (16, 12, 8, 4, 0)
    )
    indices_320 = [
        row * 20 + column
        for row in range(14, -1, -1)
        for column in palm_columns
    ]
    indices_320.extend(
        row * 20 + column
        for start in finger_starts
        for row in range(22, 18, -1)
        for column in range(start, start + 4)
    )
    if len(indices_320) != 320 or len(set(indices_320)) != 320:
        raise AssertionError("invalid 460-to-320 tactile channel map")

    packed = np.maximum(values[indices_320], 0.0).astype(np.float32, copy=False)
    return {
        f"{side}_tactile_palm": packed[:240].reshape(15, 16),
        f"{side}_tactile_fingertip": packed[240:].reshape(5, 4, 4),
    }


__all__ = ["convert_tactile_460_to_320"]
