"""Canonical causal portfolio return accounting and friction stress tests."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from trading_bot.evaluation.contracts import (
    BacktestStep,
    DailyReturn,
    EvaluationAssumptions,
    LatencyReturn,
    PositionPoint,
)
from trading_bot.evaluation.metrics import EconomicMetrics, economic_metrics


@dataclass(frozen=True, slots=True)
class FrictionMetrics:
    average_one_way_turnover: float
    cumulative_gross_traded_weight: float
    total_modeled_cost: float
    fee_cost: float
    spread_cost: float
    slippage_cost: float
    impact_cost: float
    break_even_cost_bps: float | None
    trade_count: int
    rebalance_count: int


@dataclass(frozen=True, slots=True)
class BacktestResult:
    steps: tuple[BacktestStep, ...]
    daily_returns: tuple[DailyReturn, ...]
    economic: EconomicMetrics
    friction: FrictionMetrics


@dataclass(frozen=True, slots=True)
class StressResult:
    label: str
    multiplier_or_delay: float
    economic: EconomicMetrics
    total_modeled_cost: float


def run_backtest(
    points: Sequence[PositionPoint],
    assumptions: EvaluationAssumptions,
    *,
    cost_multiplier: float = 1.0,
    spread_multiplier: float = 1.0,
) -> BacktestResult:
    """Apply weights chosen at t only to each point's subsequent realized return."""
    if not points:
        raise ValueError("backtest requires position points")
    for name, value in (
        ("cost_multiplier", cost_multiplier),
        ("spread_multiplier", spread_multiplier),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    identities = [(point.asset_id, point.timestamp_ns) for point in points]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate asset/timestamp position points are not allowed")

    grouped: dict[int, dict[str, PositionPoint]] = {}
    for point in points:
        grouped.setdefault(point.timestamp_ns, {})[point.asset_id] = point

    previous_weights: dict[str, float] = {}
    steps: list[BacktestStep] = []
    for timestamp in sorted(grouped):
        rows = grouped[timestamp]
        all_assets = sorted(set(previous_weights) | set(rows))
        gross_return = sum(
            rows[asset_id].weight * rows[asset_id].realized_return for asset_id in rows
        )
        fee_cost = 0.0
        spread_cost = 0.0
        slippage_cost = 0.0
        impact_cost = 0.0
        gross_traded_weight = 0.0
        trade_count = 0
        for asset_id in all_assets:
            current = rows[asset_id].weight if asset_id in rows else 0.0
            previous = previous_weights.get(asset_id, 0.0)
            delta = current - previous
            absolute_delta = abs(delta)
            if absolute_delta == 0:
                continue
            trade_count += 1
            gross_traded_weight += absolute_delta
            row = rows.get(asset_id)
            fee_bps = _component_bps(row, "fee_bps", assumptions.fee_bps)
            spread_bps = _component_bps(row, "spread_bps", assumptions.spread_bps)
            slippage_bps = _component_bps(row, "slippage_bps", assumptions.slippage_bps)
            impact_bps = _component_bps(row, "impact_bps", assumptions.impact_bps)
            fee_cost += absolute_delta * fee_bps / 10_000.0 * cost_multiplier
            spread_cost += (
                absolute_delta * spread_bps / 10_000.0 * cost_multiplier * spread_multiplier
            )
            slippage_cost += absolute_delta * slippage_bps / 10_000.0 * cost_multiplier
            impact_cost += absolute_delta * impact_bps / 10_000.0 * cost_multiplier

        total_cost = fee_cost + spread_cost + slippage_cost + impact_cost
        net_return = gross_return - total_cost
        if net_return <= -1.0:
            raise ValueError("modeled net step return would make NAV non-positive")
        steps.append(
            BacktestStep(
                timestamp_ns=timestamp,
                gross_return=gross_return,
                fee_cost=fee_cost,
                spread_cost=spread_cost,
                slippage_cost=slippage_cost,
                impact_cost=impact_cost,
                total_cost=total_cost,
                net_return=net_return,
                gross_traded_weight=gross_traded_weight,
                one_way_turnover=0.5 * gross_traded_weight,
                trade_count=trade_count,
            )
        )
        previous_weights = {
            asset_id: row.weight
            for asset_id, row in rows.items()
            if not math.isclose(row.weight, 0.0, abs_tol=1e-18)
        }

    daily = _aggregate_daily(steps)
    economic = economic_metrics(
        [row.net_return for row in daily],
        annualization_days=assumptions.annualization_days,
        risk_free_rate_annual=assumptions.risk_free_rate_annual,
    )
    friction = _friction_metrics(steps)
    return BacktestResult(
        steps=tuple(steps),
        daily_returns=daily,
        economic=economic,
        friction=friction,
    )


def cost_stress(
    points: Sequence[PositionPoint],
    assumptions: EvaluationAssumptions,
) -> tuple[StressResult, ...]:
    """Reprice the identical strategy path under frozen total-cost multipliers."""
    results = []
    for multiplier in assumptions.cost_stress_multipliers:
        result = run_backtest(points, assumptions, cost_multiplier=multiplier)
        results.append(
            StressResult(
                label="cost",
                multiplier_or_delay=multiplier,
                economic=result.economic,
                total_modeled_cost=result.friction.total_modeled_cost,
            )
        )
    return tuple(results)


def spread_stress(
    points: Sequence[PositionPoint],
    assumptions: EvaluationAssumptions,
    *,
    multipliers: Sequence[float] | None = None,
) -> tuple[StressResult, ...]:
    """Stress only the spread component while preserving all other costs."""
    grid = tuple(multipliers) if multipliers is not None else assumptions.cost_stress_multipliers
    if not grid:
        raise ValueError("spread stress grid must not be empty")
    results = []
    for multiplier in grid:
        result = run_backtest(points, assumptions, spread_multiplier=multiplier)
        results.append(
            StressResult(
                label="spread",
                multiplier_or_delay=multiplier,
                economic=result.economic,
                total_modeled_cost=result.friction.total_modeled_cost,
            )
        )
    return tuple(results)


def latency_stress(
    points: Sequence[PositionPoint],
    latency_returns: Sequence[LatencyReturn],
    assumptions: EvaluationAssumptions,
) -> tuple[StressResult, ...]:
    """Re-evaluate the same weights against caller-supplied delayed realized returns."""
    lookup: dict[tuple[str, int, float], float] = {}
    for row in latency_returns:
        key = (row.asset_id, row.timestamp_ns, row.delay_seconds)
        if key in lookup:
            raise ValueError("duplicate latency-return observation")
        lookup[key] = row.realized_return

    results: list[StressResult] = []
    for delay in assumptions.latency_stress_seconds:
        if delay == 0.0:
            delayed_points = tuple(points)
        else:
            delayed: list[PositionPoint] = []
            for point in points:
                key = (point.asset_id, point.timestamp_ns, delay)
                if key not in lookup:
                    raise ValueError(
                        f"latency stress is missing return for {point.asset_id} "
                        f"at {point.timestamp_ns} delay={delay}"
                    )
                delayed.append(replace(point, realized_return=lookup[key]))
            delayed_points = tuple(delayed)
        result = run_backtest(delayed_points, assumptions)
        results.append(
            StressResult(
                label="latency",
                multiplier_or_delay=delay,
                economic=result.economic,
                total_modeled_cost=result.friction.total_modeled_cost,
            )
        )
    return tuple(results)


def _component_bps(
    row: PositionPoint | None,
    attribute: str,
    fallback: float,
) -> float:
    if row is None:
        return fallback
    value = getattr(row, attribute)
    return fallback if value is None else float(value)


def _aggregate_daily(steps: Sequence[BacktestStep]) -> tuple[DailyReturn, ...]:
    grouped: dict[str, list[BacktestStep]] = {}
    for step in steps:
        date = datetime.fromtimestamp(step.timestamp_ns / 1_000_000_000, tz=UTC).date().isoformat()
        grouped.setdefault(date, []).append(step)

    nav = 1.0
    result: list[DailyReturn] = []
    for date in sorted(grouped):
        gross_factor = 1.0
        net_factor = 1.0
        total_cost = 0.0
        for step in grouped[date]:
            gross_factor *= 1.0 + step.gross_return
            net_factor *= 1.0 + step.net_return
            total_cost += step.total_cost
        gross_return = gross_factor - 1.0
        net_return = net_factor - 1.0
        nav *= 1.0 + net_return
        result.append(
            DailyReturn(
                date=date,
                gross_return=gross_return,
                total_cost=total_cost,
                net_return=net_return,
                nav=nav,
            )
        )
    return tuple(result)


def _friction_metrics(steps: Sequence[BacktestStep]) -> FrictionMetrics:
    turnovers = [step.one_way_turnover for step in steps]
    gross_traded = sum(step.gross_traded_weight for step in steps)
    gross_pnl = sum(step.gross_return for step in steps)
    break_even = gross_pnl / gross_traded * 10_000.0 if gross_traded > 0 else None
    return FrictionMetrics(
        average_one_way_turnover=statistics.fmean(turnovers) if turnovers else 0.0,
        cumulative_gross_traded_weight=gross_traded,
        total_modeled_cost=sum(step.total_cost for step in steps),
        fee_cost=sum(step.fee_cost for step in steps),
        spread_cost=sum(step.spread_cost for step in steps),
        slippage_cost=sum(step.slippage_cost for step in steps),
        impact_cost=sum(step.impact_cost for step in steps),
        break_even_cost_bps=break_even,
        trade_count=sum(step.trade_count for step in steps),
        rebalance_count=sum(step.trade_count > 0 for step in steps),
    )
