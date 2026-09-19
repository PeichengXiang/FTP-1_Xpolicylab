"""Checkpoint compatibility helpers that do not import the full FTP-1 model stack."""

from __future__ import annotations

import logging

import torch


PALIGEMMA_EMBED_TOKENS_KEY = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
PALIGEMMA_LM_HEAD_KEY = "paligemma_with_expert.paligemma.lm_head.weight"
SHARED_IMAGE_CHUNK_PREFIX = "hpt_tactile_encoder.shared_image_chunk_encoder."


def apply_tied_paligemma_embedding_fallback(
    model: torch.nn.Module,
    missing_keys: list[str],
) -> list[str]:
    """Fill embed_tokens from the tied Paligemma lm_head when the checkpoint saved only one copy."""
    remaining = list(missing_keys)
    if PALIGEMMA_EMBED_TOKENS_KEY not in remaining:
        return remaining
    paligemma = getattr(getattr(model, "paligemma_with_expert", None), "paligemma", None)
    if paligemma is None:
        return remaining
    try:
        embed_weight = paligemma.model.language_model.embed_tokens.weight
        lm_head_weight = paligemma.lm_head.weight
    except AttributeError:
        return remaining
    if embed_weight.data_ptr() == lm_head_weight.data_ptr():
        remaining.remove(PALIGEMMA_EMBED_TOKENS_KEY)
        logging.info(
            "PaliGemma embed_tokens is tied to lm_head; the checkpoint only serialized %s",
            PALIGEMMA_LM_HEAD_KEY,
        )
        return remaining
    if tuple(embed_weight.shape) != tuple(lm_head_weight.shape):
        logging.warning(
            "Cannot remap %s from %s: shapes %s vs %s",
            PALIGEMMA_EMBED_TOKENS_KEY,
            PALIGEMMA_LM_HEAD_KEY,
            tuple(embed_weight.shape),
            tuple(lm_head_weight.shape),
        )
        return remaining
    embed_weight.data.copy_(lm_head_weight.data)
    remaining.remove(PALIGEMMA_EMBED_TOKENS_KEY)
    logging.info(
        "Copied %s into embed_tokens because FTP-1 checkpoints store the tied Paligemma embedding once",
        PALIGEMMA_LM_HEAD_KEY,
    )
    return remaining


def classify_unexpected_checkpoint_keys(
    model: torch.nn.Module,
    unexpected_keys: list[str],
) -> tuple[list[str], list[str]]:
    """Split unused image-tactile encoder keys from genuine unexpected weights."""
    hpt_tactile_encoder = getattr(model, "hpt_tactile_encoder", None)
    has_shared_image_chunk = (
        hpt_tactile_encoder is not None
        and getattr(hpt_tactile_encoder, "shared_image_chunk_encoder", None) is not None
    )
    expected_unused: list[str] = []
    unexpected: list[str] = []
    for key in unexpected_keys:
        if (not has_shared_image_chunk) and key.startswith(SHARED_IMAGE_CHUNK_PREFIX):
            expected_unused.append(key)
            continue
        unexpected.append(key)
    return expected_unused, unexpected


def summarize_checkpoint_load(
    missing_keys: list[str],
    unexpected_keys: list[str],
    expected_unused_keys: list[str],
) -> None:
    """Log checkpoint mismatches after applying known FTP-1 compatibility fallbacks."""
    if expected_unused_keys:
        logging.info(
            "Ignoring %d pretrained image-tactile encoder keys unused by this matrix-only domain",
            len(expected_unused_keys),
        )
    if missing_keys:
        logging.warning(f"Missing keys after FTP-1 compatibility fallbacks: {missing_keys}")
    if unexpected_keys:
        logging.warning(f"Unexpected keys after FTP-1 compatibility fallbacks: {unexpected_keys}")
    if not missing_keys and not unexpected_keys:
        logging.info("FTP-1 backbone checkpoint keys matched after tied-embedding / domain fallbacks")
