"""Shared CPU baseline data, objective, complexity, and timing contracts."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from trading_bot.config import ObjectiveConfig
from trading_bot.training.contracts import TrainingBatch


@dataclass(frozen=True, slots=True)
class BaselineTargetNames:
    """Target-column names shared by classical and neural Phase 7 baselines."""

    return_target: str
    direction_target: str

    def __post_init__(self) -> None:
        if not self.return_target.strip() or not self.direction_target.strip():
            raise ValueError("baseline target names must not be blank")


@dataclass(frozen=True, slots=True)
class BaselineSplit:
    """One immutable train/validation view over common :class:`TrainingBatch` values."""

    train_batches: tuple[TrainingBatch, ...]
    validation_batches: tuple[TrainingBatch, ...]
    dataset_id: str
    split_id: str

    def __post_init__(self) -> None:
        if not self.train_batches or not self.validation_batches:
            raise ValueError("baseline split requires non-empty train and validation batches")
        if not self.dataset_id.strip() or not self.split_id.strip():
            raise ValueError("baseline split identity fields must not be blank")


@dataclass(frozen=True, slots=True)
class BaselineComplexity:
    """Comparable learned-state size for classical and neural baselines."""

    learned_scalar_count: int
    serialized_bytes: int

    def __post_init__(self) -> None:
        if self.learned_scalar_count < 0 or self.serialized_bytes < 0:
            raise ValueError("baseline complexity values must be non-negative")


@dataclass(frozen=True, slots=True)
class BaselineInferenceBenchmark:
    """CPU inference timing for non-PyTorch estimators."""

    iterations: int
    samples: int
    elapsed_seconds: float
    samples_per_second: float
    mean_batch_milliseconds: float


def objective_target_name(objective: ObjectiveConfig, targets: BaselineTargetNames) -> str:
    """Map the shared validated objective schema to a target column."""
    if objective.kind in {"excess_return", "ranking"}:
        return targets.return_target
    if objective.kind == "direction":
        return targets.direction_target
    raise ValueError(
        f"objective kind {objective.kind!r} requires a multi-head neural baseline path"
    )


def flatten_batch_features(batch: TrainingBatch) -> NDArray[np.float32]:
    """Flatten all non-batch feature axes for tabular CPU estimators."""
    values = batch.features.detach().cpu().float().numpy()
    return np.asarray(values.reshape(batch.batch_size, -1), dtype=np.float32)


def scalar_target_values(batch: TrainingBatch, target_name: str) -> NDArray[np.float64]:
    """Return one scalar target per sample and reject ambiguous target shapes."""
    target = batch.targets.get(target_name)
    if target is None:
        raise KeyError(f"batch does not provide target {target_name!r}")
    values = target.detach().cpu().float().numpy()
    if values.shape == (batch.batch_size, 1):
        values = values.reshape(batch.batch_size)
    if values.shape != (batch.batch_size,):
        raise ValueError(f"target {target_name!r} must contain one scalar per sample")
    result = np.asarray(values, dtype=np.float64)
    if not bool(np.isfinite(result).all()):
        raise FloatingPointError(f"target {target_name!r} contains NaN/Inf")
    return result


def collect_tabular_training_data(
    batches: Iterable[TrainingBatch],
    *,
    target_name: str,
) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
    """Collect common batches into deterministic CPU arrays for classical estimators."""
    feature_blocks: list[NDArray[np.float32]] = []
    target_blocks: list[NDArray[np.float64]] = []
    feature_width: int | None = None
    for batch in batches:
        features = flatten_batch_features(batch)
        if feature_width is None:
            feature_width = int(features.shape[1])
        elif int(features.shape[1]) != feature_width:
            raise ValueError("all baseline batches must share the same flattened feature width")
        feature_blocks.append(features)
        target_blocks.append(scalar_target_values(batch, target_name))
    if not feature_blocks:
        raise ValueError("at least one baseline training batch is required")
    features = np.concatenate(feature_blocks, axis=0)
    targets = np.concatenate(target_blocks, axis=0)
    if not bool(np.isfinite(features).all()):
        raise FloatingPointError("baseline features contain NaN/Inf")
    return features, targets


def collect_tabular_features(
    batches: Iterable[TrainingBatch],
) -> NDArray[np.float32]:
    """Collect only features while preserving deterministic input order."""
    feature_blocks = [flatten_batch_features(batch) for batch in batches]
    if not feature_blocks:
        raise ValueError("at least one baseline batch is required")
    feature_width = int(feature_blocks[0].shape[1])
    if any(int(block.shape[1]) != feature_width for block in feature_blocks[1:]):
        raise ValueError("all baseline batches must share the same flattened feature width")
    features = np.concatenate(feature_blocks, axis=0)
    if not bool(np.isfinite(features).all()):
        raise FloatingPointError("baseline features contain NaN/Inf")
    return features


def benchmark_tabular_inference(
    predictor: Callable[[NDArray[np.float32]], NDArray[np.float64]],
    features: NDArray[np.float32],
    *,
    warmup: int = 2,
    iterations: int = 10,
) -> BaselineInferenceBenchmark:
    """Measure one classical predictor without imposing a throughput threshold."""
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    if features.ndim != 2 or int(features.shape[0]) <= 0:
        raise ValueError("benchmark features must be a non-empty two-dimensional array")
    for _ in range(warmup):
        _validated_prediction(predictor(features), int(features.shape[0]))
    started = perf_counter()
    for _ in range(iterations):
        _validated_prediction(predictor(features), int(features.shape[0]))
    elapsed = perf_counter() - started
    samples = iterations * int(features.shape[0])
    return BaselineInferenceBenchmark(
        iterations=iterations,
        samples=samples,
        elapsed_seconds=elapsed,
        samples_per_second=samples / max(elapsed, 1e-12),
        mean_batch_milliseconds=elapsed * 1_000.0 / iterations,
    )


def _validated_prediction(values: NDArray[np.float64], expected_rows: int) -> None:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.shape != (expected_rows,) or not bool(np.isfinite(result).all()):
        raise ValueError("baseline predictor returned an invalid score vector")
