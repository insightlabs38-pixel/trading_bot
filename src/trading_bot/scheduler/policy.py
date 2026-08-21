"""Frozen H200 scheduler operating policy loaded independently of model code."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml  # type: ignore[import-untyped]
from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from trading_bot.config.base import FrozenConfigModel


class DeadlineThresholds(FrozenConfigModel):
    broad_exploration_min_usable_hours: PositiveFloat
    normal_promotion_min_usable_hours: PositiveFloat
    restricted_expansion_min_usable_hours: PositiveFloat
    finalists_only_min_usable_hours: PositiveFloat
    avoid_expensive_min_usable_hours: PositiveFloat

    @model_validator(mode="after")
    def thresholds_descend(self) -> DeadlineThresholds:
        values = (
            self.broad_exploration_min_usable_hours,
            self.normal_promotion_min_usable_hours,
            self.restricted_expansion_min_usable_hours,
            self.finalists_only_min_usable_hours,
            self.avoid_expensive_min_usable_hours,
        )
        if any(left <= right for left, right in zip(values, values[1:], strict=True)):
            raise ValueError("deadline thresholds must be strictly descending")
        return self


class SchedulerResourcePolicy(FrozenConfigModel):
    exclusive_gpu_trials: bool = True
    max_gpu_trials: PositiveInt = 1
    max_cpu_evaluators: PositiveInt = 2
    allow_concurrent_tiny_trials_after_calibration: bool = False
    minimum_tiny_trial_throughput_gain: float = Field(default=1.10, gt=1.0)

    @model_validator(mode="after")
    def concurrency_requires_capacity(self) -> SchedulerResourcePolicy:
        if self.allow_concurrent_tiny_trials_after_calibration and self.max_gpu_trials < 2:
            raise ValueError("concurrent tiny trials require max_gpu_trials >= 2")
        if self.exclusive_gpu_trials and self.allow_concurrent_tiny_trials_after_calibration:
            raise ValueError("exclusive GPU policy cannot enable tiny-trial concurrency")
        return self


class SchedulerRuntimePolicy(FrozenConfigModel):
    schema_version: Literal[1] = 1
    policy_id: str = Field(min_length=1)
    initial_drain_reserve_minutes: PositiveInt = 90
    drain_safety_margin_minutes: PositiveInt = 20
    runtime_quantile: float = Field(default=0.90, gt=0.5, lt=1.0)
    runtime_safety_multiplier: float = Field(default=1.15, ge=1.0)
    pruning_grace_fraction: float = Field(default=0.50, gt=0.0, le=1.0)
    max_trial_retries: int = Field(default=2, ge=0, le=10)
    worker_kill_grace_seconds: PositiveFloat = 45.0
    snapshot_interval_minutes: PositiveInt = 5
    deadline_thresholds: DeadlineThresholds
    resources: SchedulerResourcePolicy = Field(default_factory=SchedulerResourcePolicy)


class SchedulerPolicyError(ValueError):
    """Raised when the frozen scheduler operating policy cannot be loaded."""


def load_scheduler_runtime_policy(path: str | Path) -> SchedulerRuntimePolicy:
    policy_path = Path(path)
    try:
        raw = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SchedulerPolicyError(f"unable to read scheduler policy {policy_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise SchedulerPolicyError(f"invalid scheduler YAML in {policy_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise SchedulerPolicyError("scheduler policy root must be a mapping")
    try:
        return SchedulerRuntimePolicy.model_validate(raw)
    except ValueError as exc:
        raise SchedulerPolicyError(f"invalid scheduler runtime policy: {exc}") from exc
