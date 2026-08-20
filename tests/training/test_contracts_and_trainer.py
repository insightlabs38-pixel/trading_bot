"""CPU tests for the architecture-neutral model and trainer contracts."""

from __future__ import annotations

import pytest
import torch

from trading_bot.models import LinearReturnModel, MLPReturnModel
from trading_bot.training import (
    ModelOutput,
    NonFiniteTrainingError,
    Trainer,
    TrainerRuntimeOptions,
    TrainingBatch,
    UnsupportedPrecisionError,
    benchmark_inference,
    count_parameters,
)


def _batch(offset: int = 0) -> TrainingBatch:
    features = torch.tensor(
        [[1.0, 2.0], [2.0, 1.0], [3.0, 1.0], [1.0, 3.0]],
        dtype=torch.float32,
    )
    target = 0.3 * features[:, 0] - 0.2 * features[:, 1]
    return TrainingBatch(
        features=features,
        targets={"return": target},
        asset_ids=("a", "b", "c", "d"),
        timestamps_ns=torch.arange(4, dtype=torch.int64) + offset,
    )


def _loss(output: ModelOutput, batch: TrainingBatch) -> torch.Tensor:
    return torch.nn.functional.mse_loss(
        output.require("expected_return").float(),
        batch.targets["return"].float(),
    )


def test_batch_output_parameter_count_and_inference_timing_contracts() -> None:
    batch = _batch()
    model = LinearReturnModel(2)
    output = model(batch)
    output.validate(batch.batch_size)
    assert output.require("expected_return").shape == (4,)
    assert count_parameters(model) == 3
    benchmark = benchmark_inference(model, batch, warmup=0, iterations=2)
    assert benchmark.samples == 8
    assert benchmark.samples_per_second > 0


def test_trainer_supports_bf16_gradient_accumulation_scheduler_and_heartbeat() -> None:
    model = MLPReturnModel(2, hidden_features=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    heartbeats = []
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=_loss,
        max_steps=2,
        gradient_accumulation_steps=2,
        gradient_clip_norm=0.5,
        options=TrainerRuntimeOptions(heartbeat_interval_steps=1),
    )
    assert trainer.precision == "bf16"
    state = trainer.fit([_batch(index * 10) for index in range(4)], heartbeat=heartbeats.append)
    assert state.optimizer_step == 2
    assert state.micro_step == 4
    assert state.samples_seen == 16
    assert len(heartbeats) == 2
    assert heartbeats[-1].learning_rate == pytest.approx(0.005)
    assert heartbeats[-1].gpu_memory is None


def test_early_stop_is_external_and_fast_campaign_mode_is_supported() -> None:
    model = LinearReturnModel(2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=_loss,
        max_steps=5,
        precision="fp32",
        options=TrainerRuntimeOptions(mode="fast_campaign", heartbeat_interval_steps=1),
    )
    state = trainer.fit(
        [_batch(index * 10) for index in range(5)],
        should_stop=lambda current: current.optimizer_step >= 2,
    )
    assert state.optimizer_step == 2
    assert state.stopped_early is True


def test_trainer_rejects_nonfinite_loss_and_cpu_fp8() -> None:
    model = LinearReturnModel(2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def invalid_loss(_output: ModelOutput, _batch: TrainingBatch) -> torch.Tensor:
        return torch.tensor(float("nan"), requires_grad=True)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=invalid_loss,
        max_steps=1,
        precision="fp32",
    )
    with pytest.raises(NonFiniteTrainingError, match="finite scalar"):
        trainer.fit([_batch()])

    with pytest.raises(UnsupportedPrecisionError, match="GPU-specific"):
        fp8_model = LinearReturnModel(2)
        Trainer(
            model=fp8_model,
            optimizer=torch.optim.SGD(fp8_model.parameters(), lr=0.01),
            loss_fn=_loss,
            max_steps=1,
            precision="fp8",
        )
