"""Tests for the architecture-agnostic common trainer."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from trading_bot.training.contracts import ModelBatch, ModelOutput
from trading_bot.training.trainer import (
    NonFiniteTrainingError,
    TrainerSettings,
    UnsupportedPrecisionError,
    configure_reproducibility,
    train_model,
)


class Regressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, batch: ModelBatch) -> ModelOutput:
        return ModelOutput(expected_return=self.linear(batch.features).squeeze(-1))


def batches() -> list[ModelBatch]:
    return [
        ModelBatch(
            features=torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
            targets=torch.tensor([[0.0], [2.0], [4.0], [6.0]]),
        )
    ]


def mse(output: ModelOutput, batch: ModelBatch) -> torch.Tensor:
    assert output.expected_return is not None
    assert batch.targets is not None
    return torch.mean((output.expected_return - batch.targets[:, 0]) ** 2)


def test_fp32_training_updates_model_and_emits_heartbeats() -> None:
    model = Regressor()
    before = model.linear.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    events = []
    result = train_model(
        model,
        optimizer,
        batches(),
        mse,
        settings=TrainerSettings(max_steps=3, precision="fp32"),
        heartbeat=events.append,
    )
    assert result.steps_completed == 3
    assert result.micro_steps_completed == 3
    assert len(events) == 3
    assert events[-1].gpu_memory_allocated_bytes == 0
    assert not torch.equal(before, model.linear.weight.detach())


def test_gradient_accumulation_counts_microsteps_and_scheduler_steps() -> None:
    model = Regressor()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    result = train_model(
        model,
        optimizer,
        batches(),
        mse,
        settings=TrainerSettings(
            max_steps=2,
            gradient_accumulation_steps=3,
            precision="fp32",
        ),
        lr_scheduler=scheduler,
    )
    assert result.steps_completed == 2
    assert result.micro_steps_completed == 6
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.025)


def test_external_stop_hook_controls_early_stopping() -> None:
    model = Regressor()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    result = train_model(
        model,
        optimizer,
        batches(),
        mse,
        settings=TrainerSettings(max_steps=10, precision="fp32"),
        should_stop=lambda event: event.step >= 2,
    )
    assert result.stopped_early
    assert result.steps_completed == 2


def test_nonfinite_loss_is_detected_before_optimizer_step() -> None:
    model = Regressor()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def bad_loss(output: ModelOutput, batch: ModelBatch) -> torch.Tensor:
        return torch.tensor(float("nan"), requires_grad=True)

    with pytest.raises(NonFiniteTrainingError, match="non-finite loss"):
        train_model(
            model,
            optimizer,
            batches(),
            bad_loss,
            settings=TrainerSettings(max_steps=1, precision="fp32"),
        )


def test_bf16_path_executes_on_cpu_reference_environment() -> None:
    model = Regressor()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    result = train_model(
        model,
        optimizer,
        batches(),
        mse,
        settings=TrainerSettings(max_steps=1, precision="bf16"),
    )
    assert result.steps_completed == 1


def test_fp8_requires_external_validated_gpu_environment() -> None:
    model = Regressor()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(UnsupportedPrecisionError, match="Transformer Engine"):
        train_model(
            model,
            optimizer,
            batches(),
            mse,
            settings=TrainerSettings(max_steps=1, precision="fp8"),
        )


def test_reproducibility_helper_reseeds_torch_rng() -> None:
    configure_reproducibility(seed=7, deterministic=True)
    first = torch.rand(3)
    configure_reproducibility(seed=7, deterministic=True)
    second = torch.rand(3)
    assert torch.equal(first, second)
