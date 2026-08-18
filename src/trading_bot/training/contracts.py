"""Common model batch/output contracts and measurement helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn


class ModelContractError(ValueError):
    """Raised when a batch or model output violates the shared training contract."""


@dataclass(frozen=True, slots=True)
class ModelBatch:
    """Standard batched research input shared across neural architectures."""

    features: Tensor
    targets: Tensor | None = None
    timestamps_ns: Tensor | None = None
    asset_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.features.ndim < 2:
            raise ModelContractError("features must have a batch dimension and feature dimensions")
        batch_size = int(self.features.shape[0])
        if self.targets is not None and int(self.targets.shape[0]) != batch_size:
            raise ModelContractError("targets batch dimension must match features")
        if self.timestamps_ns is not None:
            if self.timestamps_ns.ndim != 1 or int(self.timestamps_ns.shape[0]) != batch_size:
                raise ModelContractError("timestamps_ns must be one value per batch element")
        if self.asset_ids and len(self.asset_ids) != batch_size:
            raise ModelContractError("asset_ids must contain one ID per batch element")

    @property
    def batch_size(self) -> int:
        return int(self.features.shape[0])

    def to(self, device: torch.device | str) -> ModelBatch:
        return ModelBatch(
            features=self.features.to(device),
            targets=None if self.targets is None else self.targets.to(device),
            timestamps_ns=(
                None if self.timestamps_ns is None else self.timestamps_ns.to(device)
            ),
            asset_ids=self.asset_ids,
        )


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """Standard prediction surface; architectures fill only applicable heads."""

    expected_return: Tensor | None = None
    rank_score: Tensor | None = None
    direction_logit: Tensor | None = None
    volatility: Tensor | None = None
    uncertainty: Tensor | None = None
    quantiles: Tensor | None = None

    def validate(self, batch_size: int) -> ModelOutput:
        present = self.present_heads()
        if not present:
            raise ModelContractError("at least one prediction head must be present")
        for name, tensor in present.items():
            if tensor.ndim == 0 or int(tensor.shape[0]) != batch_size:
                raise ModelContractError(
                    f"model output {name} must have first dimension equal to batch size"
                )
        return self

    def present_heads(self) -> dict[str, Tensor]:
        return {
            name: value
            for name, value in (
                ("expected_return", self.expected_return),
                ("rank_score", self.rank_score),
                ("direction_logit", self.direction_logit),
                ("volatility", self.volatility),
                ("uncertainty", self.uncertainty),
                ("quantiles", self.quantiles),
            )
            if value is not None
        }

    @property
    def direction_probability(self) -> Tensor | None:
        return None if self.direction_logit is None else torch.sigmoid(self.direction_logit)


@runtime_checkable
class TradingModel(Protocol):
    """Minimal architecture contract consumed by the common trainer/evaluator."""

    def __call__(self, batch: ModelBatch) -> ModelOutput: ...


def parameter_count(model: nn.Module, *, trainable_only: bool = False) -> int:
    parameters = model.parameters()
    if trainable_only:
        return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


@dataclass(frozen=True, slots=True)
class InferenceTiming:
    iterations: int
    total_seconds: float

    @property
    def mean_seconds(self) -> float:
        return self.total_seconds / self.iterations


def time_inference(
    model: nn.Module,
    batch: ModelBatch,
    *,
    iterations: int = 10,
    warmup: int = 2,
) -> InferenceTiming:
    """Measure forward latency with CUDA synchronization when a CUDA batch is supplied."""
    if iterations <= 0 or warmup < 0:
        raise ValueError("iterations must be positive and warmup non-negative")
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            output = model(batch)
            if not isinstance(output, ModelOutput):
                raise ModelContractError("models must return ModelOutput")
            output.validate(batch.batch_size)
        _synchronize_if_cuda(batch.features)
        start = time.perf_counter()
        for _ in range(iterations):
            output = model(batch)
            if not isinstance(output, ModelOutput):
                raise ModelContractError("models must return ModelOutput")
            output.validate(batch.batch_size)
        _synchronize_if_cuda(batch.features)
        elapsed = time.perf_counter() - start
    return InferenceTiming(iterations=iterations, total_seconds=elapsed)


def _synchronize_if_cuda(tensor: Tensor) -> None:
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)
