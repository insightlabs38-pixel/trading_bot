"""Tests for the common model batch/output and measurement contract."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from trading_bot.training.contracts import (
    ModelBatch,
    ModelContractError,
    ModelOutput,
    parameter_count,
    time_inference,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, batch: ModelBatch) -> ModelOutput:
        values = self.linear(batch.features)
        return ModelOutput(
            expected_return=values[:, 0],
            direction_logit=values[:, 1],
        )


def batch() -> ModelBatch:
    return ModelBatch(
        features=torch.ones(4, 3),
        targets=torch.zeros(4, 1),
        timestamps_ns=torch.arange(4, dtype=torch.int64),
        asset_ids=("a", "b", "c", "d"),
    )


def test_standard_batch_validates_metadata_dimensions_and_moves_device() -> None:
    value = batch()
    assert value.batch_size == 4
    moved = value.to("cpu")
    assert moved.features.device.type == "cpu"
    assert moved.asset_ids == value.asset_ids
    with pytest.raises(ModelContractError, match="targets batch"):
        ModelBatch(features=torch.ones(4, 3), targets=torch.ones(3, 1))
    with pytest.raises(ModelContractError, match="asset_ids"):
        ModelBatch(features=torch.ones(2, 3), asset_ids=("a",))


def test_model_output_requires_at_least_one_correctly_sized_head() -> None:
    with pytest.raises(ModelContractError, match="at least one"):
        ModelOutput().validate(4)
    with pytest.raises(ModelContractError, match="batch size"):
        ModelOutput(expected_return=torch.ones(3)).validate(4)
    output = ModelOutput(expected_return=torch.ones(4)).validate(4)
    assert set(output.present_heads()) == {"expected_return"}


def test_direction_probability_is_derived_from_logits() -> None:
    output = ModelOutput(direction_logit=torch.tensor([0.0, 2.0])).validate(2)
    assert output.direction_probability is not None
    assert output.direction_probability[0].item() == pytest.approx(0.5)


def test_parameter_count_reports_total_and_trainable_values() -> None:
    model = TinyModel()
    assert parameter_count(model) == 8
    model.linear.bias.requires_grad_(False)
    assert parameter_count(model, trainable_only=True) == 6


def test_inference_timing_validates_output_and_reports_mean() -> None:
    timing = time_inference(TinyModel(), batch(), iterations=3, warmup=1)
    assert timing.iterations == 3
    assert timing.total_seconds >= 0
    assert timing.mean_seconds >= 0


def test_quantile_and_uncertainty_heads_allow_extra_dimensions() -> None:
    output = ModelOutput(
        expected_return=torch.ones(4),
        volatility=torch.ones(4),
        uncertainty=torch.ones(4, 2),
        quantiles=torch.ones(4, 5),
    ).validate(4)
    assert output.quantiles is not None
    assert output.quantiles.shape == (4, 5)
