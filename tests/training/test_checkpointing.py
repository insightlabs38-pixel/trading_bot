"""Checkpoint integrity, bookkeeping, and exact CPU continuation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from trading_bot.models import MLPReturnModel
from trading_bot.training import (
    CheckpointCorruptionError,
    CheckpointIdentity,
    CheckpointManager,
    CheckpointResumeError,
    ModelOutput,
    Trainer,
    TrainingBatch,
)


def _batch(offset: int = 0) -> TrainingBatch:
    features = torch.tensor(
        [[1.0, 2.0], [2.0, 1.0], [3.0, 1.0], [1.0, 3.0]],
        dtype=torch.float32,
    )
    return TrainingBatch(
        features=features,
        targets={"return": 0.3 * features[:, 0] - 0.2 * features[:, 1]},
        asset_ids=("a", "b", "c", "d"),
        timestamps_ns=torch.arange(4, dtype=torch.int64) + offset,
    )


def _loss(output: ModelOutput, batch: TrainingBatch) -> torch.Tensor:
    return torch.nn.functional.mse_loss(
        output.require("expected_return").float(),
        batch.targets["return"].float(),
    )


def _identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        model_config_hash="model-config-sha",
        training_config_hash="training-config-sha",
        dataset_id="dataset-v1",
        split_id="split-v1",
    )


def _stack() -> tuple[
    MLPReturnModel,
    torch.optim.AdamW,
    torch.optim.lr_scheduler.StepLR,
    Trainer,
]:
    model = MLPReturnModel(2, hidden_features=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=_loss,
        max_steps=3,
        precision="fp32",
        seed=17,
    )
    return model, optimizer, scheduler, trainer


def test_checkpoint_resume_reproduces_exact_next_step_with_dropout(tmp_path: Path) -> None:
    model, optimizer, scheduler, trainer = _stack()
    initial_state = trainer.fit([_batch(0), _batch(10)])
    manager = CheckpointManager(tmp_path / "checkpoints")
    checkpoint = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=initial_state,
        identity=_identity(),
        precision="fp32",
        is_best=True,
    )
    reference_state = trainer.fit([_batch(20)], initial_state=initial_state)
    reference_parameters = [parameter.detach().clone() for parameter in model.parameters()]

    resumed_model, resumed_optimizer, resumed_scheduler, resumed_trainer = _stack()
    restored = manager.restore(
        "latest",
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        expected_identity=_identity(),
    )
    resumed_state = resumed_trainer.fit([_batch(20)], initial_state=restored.training_state)

    assert checkpoint == manager.latest() == manager.best()
    assert resumed_state == reference_state
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(reference_parameters, resumed_model.parameters(), strict=True)
    )


def test_checkpoint_rejects_wrong_identity_before_restore(tmp_path: Path) -> None:
    model, optimizer, scheduler, trainer = _stack()
    state = trainer.fit([_batch()])
    manager = CheckpointManager(tmp_path / "checkpoints")
    manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=state,
        identity=_identity(),
        precision="fp32",
    )
    wrong = CheckpointIdentity("other", "training-config-sha", "dataset-v1", "split-v1")
    with pytest.raises(CheckpointResumeError, match="identity"):
        manager.restore(
            "latest",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_identity=wrong,
        )


def test_checkpoint_corruption_is_detected_before_torch_restore(tmp_path: Path) -> None:
    model, optimizer, scheduler, trainer = _stack()
    state = trainer.fit([_batch()])
    manager = CheckpointManager(tmp_path / "checkpoints")
    checkpoint = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=state,
        identity=_identity(),
        precision="fp32",
    )
    state_path = checkpoint / "state.pt"
    payload = bytearray(state_path.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    state_path.write_bytes(payload)
    with pytest.raises(CheckpointCorruptionError, match="checksum mismatch"):
        manager.restore(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_identity=_identity(),
        )
