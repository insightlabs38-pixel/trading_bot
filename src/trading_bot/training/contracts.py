"""Architecture-independent tensor contracts and systems measurements."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, cast

import torch
from torch import Tensor, nn

HeadName = Literal[
    "expected_return",
    "rank_score",
    "direction_probability",
    "volatility",
    "uncertainty",
    "quantiles",
]


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """One model-ready batch with identity fields preserved beside tensors."""

    features: Tensor
    targets: Mapping[str, Tensor]
    asset_ids: tuple[str, ...]
    timestamps_ns: Tensor

    def __post_init__(self) -> None:
        if self.features.ndim < 2:
            raise ValueError(
                "features must have a batch dimension and at least one feature dimension"
            )
        batch_size = int(self.features.shape[0])
        if batch_size <= 0:
            raise ValueError("training batches must not be empty")
        if len(self.asset_ids) != batch_size or self.timestamps_ns.shape != (batch_size,):
            raise ValueError("batch identity fields must match feature batch size")
        if self.timestamps_ns.dtype != torch.int64:
            raise ValueError("timestamps_ns must use int64")
        if any(not asset_id.strip() for asset_id in self.asset_ids):
            raise ValueError("asset_ids must not contain blank identifiers")
        if len(set(zip(self.asset_ids, self.timestamps_ns.tolist(), strict=True))) != batch_size:
            raise ValueError("asset/timestamp identities must be unique within a batch")
        if not self.targets:
            raise ValueError("training batches must contain at least one target")
        for name, target in self.targets.items():
            if not name.strip() or target.ndim == 0 or int(target.shape[0]) != batch_size:
                raise ValueError(
                    "every target must be named and share the feature batch dimension"
                )

    @property
    def batch_size(self) -> int:
        return int(self.features.shape[0])

    def to(self, device: torch.device) -> TrainingBatch:
        """Return the same logical batch with tensors moved to one device."""
        return TrainingBatch(
            features=self.features.to(device),
            targets={name: target.to(device) for name, target in self.targets.items()},
            asset_ids=self.asset_ids,
            timestamps_ns=self.timestamps_ns.to(device),
        )


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """Architecture-neutral prediction heads consumed by trainers and evaluators."""

    expected_return: Tensor | None = None
    rank_score: Tensor | None = None
    direction_probability: Tensor | None = None
    volatility: Tensor | None = None
    uncertainty: Tensor | None = None
    quantiles: Tensor | None = None

    def tensors(self) -> dict[str, Tensor]:
        values: dict[str, Tensor] = {}
        for name in (
            "expected_return",
            "rank_score",
            "direction_probability",
            "volatility",
            "uncertainty",
            "quantiles",
        ):
            value = cast(Tensor | None, getattr(self, name))
            if value is not None:
                values[name] = value
        return values

    def require(self, name: HeadName) -> Tensor:
        """Return a required head or fail before objective code silently changes semantics."""
        value = cast(Tensor | None, getattr(self, name))
        if value is None:
            raise KeyError(f"model output does not provide {name}")
        return value

    def validate(self, batch_size: int) -> None:
        """Validate shared batch dimensions and reject non-finite model outputs."""
        values = self.tensors()
        if not values:
            raise ValueError("model output must contain at least one prediction head")
        for name, tensor in values.items():
            if tensor.ndim == 0 or int(tensor.shape[0]) != batch_size:
                raise ValueError(f"model output {name} does not match batch size")
            if not bool(torch.isfinite(tensor).all().item()):
                raise FloatingPointError(f"model output {name} contains NaN/Inf")


class TradingModel(nn.Module):
    """Common PyTorch base class for every trainable architecture."""

    def forward(self, batch: TrainingBatch) -> ModelOutput:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class InferenceBenchmark:
    """Small systems-side inference timing result."""

    iterations: int
    samples: int
    elapsed_seconds: float
    samples_per_second: float
    mean_batch_milliseconds: float


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    """Return total or trainable parameter count without architecture-specific logic."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def benchmark_inference(
    model: TradingModel,
    batch: TrainingBatch,
    *,
    warmup: int = 2,
    iterations: int = 10,
) -> InferenceBenchmark:
    """Measure eager inference without assuming a CUDA device."""
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    device = next(model.parameters(), batch.features).device
    device_batch = batch.to(device)
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            cast(ModelOutput, model(device_batch)).validate(device_batch.batch_size)
        _synchronize_if_cuda(device)
        started = perf_counter()
        for _ in range(iterations):
            cast(ModelOutput, model(device_batch)).validate(device_batch.batch_size)
        _synchronize_if_cuda(device)
        elapsed = perf_counter() - started
    samples = iterations * device_batch.batch_size
    return InferenceBenchmark(
        iterations=iterations,
        samples=samples,
        elapsed_seconds=elapsed,
        samples_per_second=samples / elapsed,
        mean_batch_milliseconds=elapsed * 1_000.0 / iterations,
    )


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
