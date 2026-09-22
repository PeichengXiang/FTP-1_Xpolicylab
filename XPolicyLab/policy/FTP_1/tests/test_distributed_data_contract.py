from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch


POLICY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY_DIR / "ftp1-policy" / "src"))
sys.path.insert(0, str(POLICY_DIR / "ftp1-policy"))

import openpi.dataset_zarr as dataset_zarr  # noqa: E402
from openpi.training.data_loader import TorchDataLoader  # noqa: E402
from scripts.train_pytorch import set_seed  # noqa: E402


def test_replay_buffer_file_order_is_deterministic(monkeypatch) -> None:
    monkeypatch.setattr(
        dataset_zarr.os,
        "listdir",
        lambda _path: ["task_b.zarr", "ignore.txt", "task_c.zarr", "task_a.zarr"],
    )
    opened: list[str] = []

    def fake_open(path, _cache_dir):
        opened.append(str(path))
        return path

    monkeypatch.setattr(dataset_zarr, "get_replay_buffer", fake_open)

    buffers, names = dataset_zarr.get_replay_buffer_list("/dataset", cache_dir=None)

    assert names == ["task_a.zarr", "task_b.zarr", "task_c.zarr"]
    assert opened == [f"/dataset/{name}" for name in names]
    assert buffers == opened


class _RecordingSampler:
    def __init__(self, size: int) -> None:
        self.size = size
        self.epochs: list[int] = []

    def __iter__(self):
        return iter(range(self.size))

    def __len__(self) -> int:
        return self.size

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


def test_torch_loader_advances_plain_sampler_epoch() -> None:
    dataset = [np.array([idx], dtype=np.int64) for idx in range(8)]
    sampler = _RecordingSampler(len(dataset))
    loader = TorchDataLoader(
        dataset,
        local_batch_size=2,
        sampler=sampler,
        num_workers=0,
        seed=42,
        framework="pytorch",
    )

    loader.set_epoch(3)
    iterator = iter(loader)
    for _ in range(5):
        next(iterator)

    assert sampler.epochs[0] == 3
    assert sampler.epochs[-1] == 4


def test_global_rank_seed_is_distinct_and_reproducible(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def sample(rank: int) -> tuple[float, float, float]:
        set_seed(42, rank)
        return random.random(), float(np.random.random()), float(torch.rand(()))

    assert sample(0) == sample(0)
    assert sample(0) != sample(1)
