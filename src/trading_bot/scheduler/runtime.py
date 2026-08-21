"""Measured-runtime estimation, adaptive drain reserve, and launch guards."""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import Field

from trading_bot.config.base import FrozenConfigModel
from trading_bot.scheduler.db import CampaignDB
from trading_bot.scheduler.policy import SchedulerRuntimePolicy
from trading_bot.scheduler.types import DrainInputs, LaunchPriority, TrialSpec


class DeadlineMode(StrEnum):
    BROAD_EXPLORATION = "BROAD_EXPLORATION"
    NORMAL_PROMOTION = "NORMAL_PROMOTION"
    RESTRICTED_EXPANSION = "RESTRICTED_EXPANSION"
    FINALISTS_ONLY = "FINALISTS_ONLY"
    AVOID_EXPENSIVE = "AVOID_EXPENSIVE"
    DRAIN = "DRAIN"


class LaunchDecision(FrozenConfigModel):
    allowed: bool
    mode: DeadlineMode
    usable_seconds: float = Field(ge=0.0)
    estimated_runtime_seconds: float = Field(gt=0.0)
    reason: str = Field(min_length=1)


class RuntimeEstimator:
    """Conservative family/scale runtime estimator derived from observed partial budgets."""

    def __init__(self, db: CampaignDB, policy: SchedulerRuntimePolicy) -> None:
        self.db = db
        self.policy = policy

    def estimate_seconds(self, spec: TrialSpec) -> float:
        observations = self.db.runtime_observations(family=spec.family, scale=spec.scale)
        context = spec.config.get("context_length")
        if isinstance(context, int):
            exact = tuple(item for item in observations if item.context_length == context)
            if exact:
                observations = exact
        if not observations:
            return spec.fallback_runtime_seconds * self.policy.runtime_safety_multiplier
        normalized_full_runtime = sorted(
            item.runtime_seconds / item.budget_fraction for item in observations
        )
        quantile_index = max(
            0,
            math.ceil(self.policy.runtime_quantile * len(normalized_full_runtime)) - 1,
        )
        conservative_full = normalized_full_runtime[quantile_index]
        return (
            conservative_full
            * spec.budget_fraction
            * self.policy.runtime_safety_multiplier
        )


def dynamic_drain_reserve_seconds(
    policy: SchedulerRuntimePolicy,
    inputs: DrainInputs,
) -> float:
    """Reserve enough time for evaluator backlog, durable sync, and a fixed safety margin."""
    minimum = float(policy.initial_drain_reserve_minutes * 60)
    safety = float(policy.drain_safety_margin_minutes * 60)
    if inputs.unsynced_bytes > 0 and inputs.storage_bytes_per_second is None:
        return math.inf
    sync_seconds = 0.0
    if inputs.unsynced_bytes > 0:
        assert inputs.storage_bytes_per_second is not None
        sync_seconds = inputs.unsynced_bytes / inputs.storage_bytes_per_second
    dynamic = inputs.outstanding_evaluator_seconds + sync_seconds + safety
    return max(minimum, dynamic)


def usable_seconds(*, now: float, deadline_at: float, drain_reserve_seconds: float) -> float:
    if not math.isfinite(drain_reserve_seconds):
        return 0.0
    return max(0.0, deadline_at - now - drain_reserve_seconds)


def deadline_mode(
    policy: SchedulerRuntimePolicy,
    *,
    now: float,
    deadline_at: float,
    drain_reserve_seconds: float,
) -> DeadlineMode:
    usable_hours = usable_seconds(
        now=now,
        deadline_at=deadline_at,
        drain_reserve_seconds=drain_reserve_seconds,
    ) / 3600.0
    thresholds = policy.deadline_thresholds
    if usable_hours >= thresholds.broad_exploration_min_usable_hours:
        return DeadlineMode.BROAD_EXPLORATION
    if usable_hours >= thresholds.normal_promotion_min_usable_hours:
        return DeadlineMode.NORMAL_PROMOTION
    if usable_hours >= thresholds.restricted_expansion_min_usable_hours:
        return DeadlineMode.RESTRICTED_EXPANSION
    if usable_hours >= thresholds.finalists_only_min_usable_hours:
        return DeadlineMode.FINALISTS_ONLY
    if usable_hours >= thresholds.avoid_expensive_min_usable_hours:
        return DeadlineMode.AVOID_EXPENSIVE
    return DeadlineMode.DRAIN


def launch_decision(
    policy: SchedulerRuntimePolicy,
    spec: TrialSpec,
    *,
    estimated_runtime_seconds: float,
    now: float,
    deadline_at: float,
    drain_reserve_seconds: float,
) -> LaunchDecision:
    available = usable_seconds(
        now=now,
        deadline_at=deadline_at,
        drain_reserve_seconds=drain_reserve_seconds,
    )
    mode = deadline_mode(
        policy,
        now=now,
        deadline_at=deadline_at,
        drain_reserve_seconds=drain_reserve_seconds,
    )
    if mode == DeadlineMode.DRAIN:
        return LaunchDecision(
            allowed=False,
            mode=mode,
            usable_seconds=available,
            estimated_runtime_seconds=estimated_runtime_seconds,
            reason="campaign is inside adaptive drain reserve",
        )
    if estimated_runtime_seconds > available:
        return LaunchDecision(
            allowed=False,
            mode=mode,
            usable_seconds=available,
            estimated_runtime_seconds=estimated_runtime_seconds,
            reason="conservative runtime estimate does not fit before drain",
        )
    if mode == DeadlineMode.RESTRICTED_EXPANSION and spec.priority == LaunchPriority.OPTIONAL:
        return LaunchDecision(
            allowed=False,
            mode=mode,
            usable_seconds=available,
            estimated_runtime_seconds=estimated_runtime_seconds,
            reason="optional exploration disabled in restricted-expansion mode",
        )
    if mode in {DeadlineMode.FINALISTS_ONLY, DeadlineMode.AVOID_EXPENSIVE} and not (
        spec.stage == "finalists" or spec.priority == LaunchPriority.FINALIST
    ):
        return LaunchDecision(
            allowed=False,
            mode=mode,
            usable_seconds=available,
            estimated_runtime_seconds=estimated_runtime_seconds,
            reason="only finalist work may launch at this deadline tier",
        )
    return LaunchDecision(
        allowed=True,
        mode=mode,
        usable_seconds=available,
        estimated_runtime_seconds=estimated_runtime_seconds,
        reason="trial fits current deadline and priority policy",
    )
