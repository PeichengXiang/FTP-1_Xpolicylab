from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from torch import nn


FTP1_SRC = Path(__file__).resolve().parents[1] / "ftp1-policy" / "src"
sys.path.insert(0, str(FTP1_SRC))

from openpi.policies.ftp1_checkpoint_compat import (  # noqa: E402
    apply_tied_paligemma_embedding_fallback,
    classify_unexpected_checkpoint_keys,
)


class _DummyPaliGemma(nn.Module):
    def __init__(self, tied: bool):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.embed_tokens = nn.Embedding(4, 8)
        self.lm_head = nn.Linear(8, 4, bias=False)
        if tied:
            self.model.language_model.embed_tokens.weight = self.lm_head.weight
        else:
            with torch.no_grad():
                self.lm_head.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(4, 8))
                self.model.language_model.embed_tokens.weight.zero_()


class _DummyFTP1(nn.Module):
    def __init__(self, *, tied: bool, has_shared_image: bool):
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.paligemma = _DummyPaliGemma(tied)
        self.hpt_tactile_encoder = nn.Module()
        self.hpt_tactile_encoder.shared_image_chunk_encoder = nn.Linear(2, 2) if has_shared_image else None


def test_tied_embed_tokens_are_treated_as_loaded() -> None:
    model = _DummyFTP1(tied=True, has_shared_image=False)
    remaining = apply_tied_paligemma_embedding_fallback(
        model,
        ["paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"],
    )
    assert remaining == []


def test_untied_embed_tokens_are_copied_from_lm_head() -> None:
    model = _DummyFTP1(tied=False, has_shared_image=False)
    remaining = apply_tied_paligemma_embedding_fallback(
        model,
        ["paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"],
    )
    assert remaining == []
    embed = model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight
    lm_head = model.paligemma_with_expert.paligemma.lm_head.weight
    assert torch.equal(embed, lm_head)


def test_image_tactile_keys_are_expected_on_matrix_only_models() -> None:
    model = _DummyFTP1(tied=True, has_shared_image=False)
    unused, unexpected = classify_unexpected_checkpoint_keys(
        model,
        [
            "hpt_tactile_encoder.shared_image_chunk_encoder.image_proj.weight",
            "some_other_unexpected.weight",
        ],
    )
    assert unused == ["hpt_tactile_encoder.shared_image_chunk_encoder.image_proj.weight"]
    assert unexpected == ["some_other_unexpected.weight"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_tied_embed_tokens_are_treated_as_loaded()
    test_untied_embed_tokens_are_copied_from_lm_head()
    test_image_tactile_keys_are_expected_on_matrix_only_models()
    print("pretrain load fallbacks: PASS")
