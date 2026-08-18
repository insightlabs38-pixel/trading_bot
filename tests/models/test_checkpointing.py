"""Tests for atomic continuation checkpoint save/restore behavior."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from trading_bot.training.checkpointing import (
    CheckpointCompatibilityError,
    CheckpointCorruptionError,
    CheckpointError,
    CheckpointIdentity,
    load_checkpoint,
    resolve_checkpoint_pointer,
    save_checkpoint,
)


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)


def identity(suffix: str = "") -> CheckpointIdentity:
    return CheckpointIdentity(
        model_config_sha256=("a" * 63) + (suffix or "a"),
        training_config_sha256="b" * 64,
        dataset_version="dataset-v1",
        split_version="split-v1",
    )


def state():
    model = Tiny()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    return model, optimizer, scheduler


def test_checkpoint_round_trip_restores_model_optimizer_scheduler_and_cursor(tmp_path: Path) -> None:
    model, optimizer, scheduler = state()
    loss = model.linear(torch.tensor([[1.0, 2.0]])).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    expected_weight = model.linear.weight.detach().clone()
    expected_lr = optimizer.param_groups[0]["lr"]
    save_checkpoint(
        tmp_path, "ckpt-1", model, optimizer, step=7, cursor=123, precision="bf16",
        identity=identity(), lr_scheduler=scheduler, is_best=True,
    )
    with torch.no_grad():
        model.linear.weight.zero_()
    optimizer.param_groups[0]["lr"] = 9.0
    restored = load_checkpoint(
        tmp_path, "ckpt-1", model, optimizer, expected_identity=identity(),
        lr_scheduler=scheduler,
    )
    assert torch.equal(model.linear.weight.detach(), expected_weight)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)
    assert (restored.step, restored.cursor, restored.precision) == (7, 123, "bf16")


def test_latest_and_best_pointers_are_updated(tmp_path: Path) -> None:
    model, optimizer, _ = state()
    common = dict(step=1, cursor=1, precision="fp32", identity=identity())
    save_checkpoint(tmp_path, "first", model, optimizer, is_best=True, **common)
    save_checkpoint(tmp_path, "second", model, optimizer, **common)
    assert resolve_checkpoint_pointer(tmp_path, "latest") == "second"
    assert resolve_checkpoint_pointer(tmp_path, "best") == "first"


def test_resume_identity_mismatch_is_rejected_before_state_restore(tmp_path: Path) -> None:
    model, optimizer, _ = state()
    save_checkpoint(
        tmp_path, "ckpt", model, optimizer, step=1, cursor=1, precision="fp32",
        identity=identity(),
    )
    before = model.linear.weight.detach().clone()
    with pytest.raises(CheckpointCompatibilityError, match="identity mismatch"):
        load_checkpoint(tmp_path, "ckpt", model, optimizer, expected_identity=identity("c"))
    assert torch.equal(before, model.linear.weight.detach())


def test_checkpoint_corruption_is_detected_before_deserialization(tmp_path: Path) -> None:
    model, optimizer, _ = state()
    save_checkpoint(
        tmp_path, "ckpt", model, optimizer, step=1, cursor=1, precision="fp32",
        identity=identity(),
    )
    path = tmp_path / "ckpt.pt"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(CheckpointCorruptionError, match="checksum mismatch"):
        load_checkpoint(tmp_path, "ckpt", model, optimizer, expected_identity=identity())


def test_rng_state_is_restored_for_true_continuation(tmp_path: Path) -> None:
    random.seed(11); np.random.seed(11); torch.manual_seed(11)
    model, optimizer, _ = state()
    save_checkpoint(
        tmp_path, "rng", model, optimizer, step=0, cursor=0, precision="fp32",
        identity=identity(),
    )
    expected = (random.random(), float(np.random.rand()), torch.rand(2))
    random.seed(999); np.random.seed(999); torch.manual_seed(999)
    load_checkpoint(tmp_path, "rng", model, optimizer, expected_identity=identity())
    actual = (random.random(), float(np.random.rand()), torch.rand(2))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_checkpoint_ids_are_immutable_and_cannot_be_overwritten(tmp_path: Path) -> None:
    model, optimizer, _ = state()
    kwargs = dict(step=0, cursor=0, precision="fp32", identity=identity())
    save_checkpoint(tmp_path, "same", model, optimizer, **kwargs)
    with pytest.raises(CheckpointError, match="already exists"):
        save_checkpoint(tmp_path, "same", model, optimizer, **kwargs)
