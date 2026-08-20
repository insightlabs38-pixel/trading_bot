from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trading_bot.evaluation import (
    EvaluationAssumptions,
    LatencyReturn,
    PositionPoint,
    cost_stress,
    latency_stress,
    run_backtest,
    spread_stress,
)


def _ts(day: int, minute: int = 0) -> int:
    return int(datetime(2026, 1, day, 14, 30 + minute, tzinfo=UTC).timestamp() * 1_000_000_000)


def test_known_cost_and_buy_and_hold_return_accounting() -> None:
    assumptions = EvaluationAssumptions(
        fee_bps=1.0,
        spread_bps=2.0,
        slippage_bps=3.0,
        impact_bps=4.0,
    )
    points = [
        PositionPoint("A", _ts(2), 0.5, 0.10),
        PositionPoint("B", _ts(2), -0.5, -0.10),
        PositionPoint("A", _ts(3), 0.5, 0.02),
        PositionPoint("B", _ts(3), -0.5, -0.02),
    ]
    result = run_backtest(points, assumptions)
    first = result.steps[0]
    assert first.gross_return == pytest.approx(0.10)
    assert first.gross_traded_weight == pytest.approx(1.0)
    assert first.one_way_turnover == pytest.approx(0.5)
    assert first.fee_cost == pytest.approx(0.0001)
    assert first.spread_cost == pytest.approx(0.0002)
    assert first.slippage_cost == pytest.approx(0.0003)
    assert first.impact_cost == pytest.approx(0.0004)
    assert first.total_cost == pytest.approx(0.001)
    assert first.net_return == pytest.approx(0.099)
    assert result.steps[1].gross_traded_weight == pytest.approx(0.0)
    assert result.steps[1].total_cost == pytest.approx(0.0)
    assert result.friction.trade_count == 2
    assert result.friction.rebalance_count == 1
    assert result.friction.average_one_way_turnover == pytest.approx(0.25)
    assert result.friction.break_even_cost_bps == pytest.approx(1200.0)


def test_cost_spread_and_latency_stress_reprice_same_strategy() -> None:
    assumptions = EvaluationAssumptions(
        fee_bps=1.0,
        spread_bps=2.0,
        slippage_bps=1.0,
        impact_bps=1.0,
        cost_stress_multipliers=(1.0, 2.0),
        latency_stress_seconds=(0.0, 1.0),
    )
    points = [
        PositionPoint("A", _ts(2), 1.0, 0.02),
        PositionPoint("A", _ts(3), 1.0, 0.01),
        PositionPoint("A", _ts(4), 1.0, -0.005),
    ]
    costs = cost_stress(points, assumptions)
    assert len(costs) == 2
    assert costs[1].total_modeled_cost == pytest.approx(costs[0].total_modeled_cost * 2)

    spreads = spread_stress(points, assumptions, multipliers=(1.0, 3.0))
    assert spreads[1].total_modeled_cost > spreads[0].total_modeled_cost
    assert spreads[1].total_modeled_cost < spreads[0].total_modeled_cost * 3

    delayed = [
        LatencyReturn("A", _ts(2), 1.0, 0.015),
        LatencyReturn("A", _ts(3), 1.0, 0.005),
        LatencyReturn("A", _ts(4), 1.0, -0.010),
    ]
    latency = latency_stress(points, delayed, assumptions)
    assert [row.multiplier_or_delay for row in latency] == [0.0, 1.0]
    assert latency[1].economic.final_nav < latency[0].economic.final_nav


def test_backtest_rejects_duplicate_identity() -> None:
    assumptions = EvaluationAssumptions()
    point = PositionPoint("A", _ts(2), 1.0, 0.01)
    with pytest.raises(ValueError, match="duplicate"):
        run_backtest([point, point], assumptions)
