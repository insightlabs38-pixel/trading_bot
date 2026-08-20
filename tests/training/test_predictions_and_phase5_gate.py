"""Prediction artifact and three-architecture CPU gate for Phase 5."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from trading_bot.models import LinearReturnModel, MLPReturnModel, ResidualGatedReturnModel
from trading_bot.training import (
    CheckpointIdentity,
    CheckpointManager,
    ModelOutput,
    PredictionArtifact,
    PredictionArtifactError,
    Trainer,
    TrainingBatch,
    predict_records,
    write_prediction_artifact,
)
from trading_bot.training.contracts import TradingModel


def _batches() -> tuple[TrainingBatch, ...]:
    batches = []
    for batch_index in range(4):
        features = torch.tensor(
            [[1.0, 2.0], [2.0, 1.0], [3.0, 1.0], [1.0, 3.0]],
            dtype=torch.float32,
        ) + batch_index * 0.05
        batches.append(
            TrainingBatch(
                features=features,
                targets={"return": 0.3 * features[:, 0] - 0.2 * features[:, 1]},
                asset_ids=tuple(f"asset-{batch_index}-{index}" for index in range(4)),
                timestamps_ns=torch.arange(4, dtype=torch.int64) + batch_index * 10,
            )
        )
    return tuple(batches)


def _loss(output: ModelOutput, batch: TrainingBatch) -> torch.Tensor:
    return torch.nn.functional.mse_loss(
        output.require("expected_return").float(),
        batch.targets["return"].float(),
    )


def _artifact_mse(path: Path) -> float:
    """Evaluator-side smoke that depends only on the durable prediction contract."""
    records = PredictionArtifact(path).records()
    errors = [
        (record.expected_return - record.target) ** 2
        for record in records
        if record.expected_return is not None
    ]
    if len(errors) != len(records):
        raise AssertionError("expected return head missing from gate predictions")
    return sum(errors) / len(errors)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LinearReturnModel(2),
        lambda: MLPReturnModel(2, hidden_features=8),
        lambda: ResidualGatedReturnModel(2, hidden_features=8),
    ],
)
def test_three_architectures_train_checkpoint_resume_predict_and_evaluate(
    tmp_path: Path,
    factory: Callable[[], TradingModel],
) -> None:
    batches = _batches()
    model = factory()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=_loss,
        max_steps=3,
        precision="fp32",
        seed=23,
    )
    state = trainer.fit(batches[:2])
    model_name = model.__class__.__name__
    identity = CheckpointIdentity(
        model_config_hash=f"{model_name}-config",
        training_config_hash="training-config",
        dataset_id="dataset-v1",
        split_id="split-v1",
    )
    manager = CheckpointManager(tmp_path / model_name / "checkpoints")
    manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        training_state=state,
        identity=identity,
        precision="fp32",
    )

    resumed_model = factory()
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=0.02)
    resumed_trainer = Trainer(
        model=resumed_model,
        optimizer=resumed_optimizer,
        loss_fn=_loss,
        max_steps=3,
        precision="fp32",
        seed=23,
    )
    restored = manager.restore(
        "latest",
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=None,
        expected_identity=identity,
    )
    final_state = resumed_trainer.fit(batches[2:3], initial_state=restored.training_state)
    assert final_state.optimizer_step == 3

    records = predict_records(resumed_model, batches[3:], target_name="return")
    result = write_prediction_artifact(
        records,
        tmp_path / model_name / "predictions",
        dataset_id="dataset-v1",
        split_id="split-v1",
        model_config_hash=identity.model_config_hash,
        checkpoint_id=f"step-{final_state.optimizer_step:08d}",
        target_name="return",
    )
    assert result.record_count == 4
    assert _artifact_mse(result.path) >= 0.0


def test_prediction_artifact_detects_tampering(tmp_path: Path) -> None:
    model = LinearReturnModel(2)
    records = predict_records(model, _batches()[:1], target_name="return")
    result = write_prediction_artifact(
        records,
        tmp_path / "predictions",
        dataset_id="dataset-v1",
        split_id="split-v1",
        model_config_hash="model-config",
        checkpoint_id="step-1",
        target_name="return",
    )
    path = result.path / "predictions.parquet"
    payload = bytearray(path.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    path.write_bytes(payload)
    with pytest.raises(PredictionArtifactError, match="checksum mismatch"):
        PredictionArtifact(result.path)
