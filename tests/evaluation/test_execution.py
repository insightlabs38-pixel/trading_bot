from __future__ import annotations

import pytest

from trading_bot.evaluation import (
    Order,
    Quote,
    implementation_shortfall_bps,
    liquidity_diagnostics,
    simulate_l1_order,
    summarize_implementation_shortfall,
)


def test_implementation_shortfall_and_liquidity_diagnostics() -> None:
    assert implementation_shortfall_bps(
        side="buy", decision_price=100.0, execution_price=100.1
    ) == pytest.approx(10.0)
    assert implementation_shortfall_bps(
        side="sell", decision_price=100.0, execution_price=99.9
    ) == pytest.approx(10.0)
    summary = summarize_implementation_shortfall([10.0, 20.0, 30.0, 40.0])
    assert summary.mean_bps == pytest.approx(25.0)
    assert summary.median_bps == pytest.approx(25.0)
    assert summary.p90_bps == pytest.approx(37.0)
    assert summary.p95_bps == pytest.approx(38.5)

    liquidity = liquidity_diagnostics(
        order_notional=50_000,
        relevant_market_volume_notional=1_000_000,
        position_notional=200_000,
        average_daily_dollar_volume=10_000_000,
    )
    assert liquidity.order_participation_fraction == pytest.approx(0.05)
    assert liquidity.position_adv_fraction == pytest.approx(0.02)


def test_l1_market_and_limit_simulator_is_strictly_no_lookahead() -> None:
    order = Order(
        asset_id="A",
        decision_timestamp_ns=1_000_000_000,
        side="buy",
        quantity=150.0,
    )
    quotes = [
        Quote(900_000_000, bid=98.9, ask=99.0, bid_size=1000, ask_size=1000),
        Quote(1_000_000_000, bid=99.9, ask=100.0, bid_size=100, ask_size=100),
        Quote(2_000_000_000, bid=100.0, ask=100.2, bid_size=100, ask_size=100),
    ]
    fill = simulate_l1_order(order, quotes)
    assert fill is not None
    assert fill.first_fill_timestamp_ns == 1_000_000_000
    assert fill.last_fill_timestamp_ns == 2_000_000_000
    assert fill.filled_quantity == pytest.approx(150.0)
    assert fill.average_price == pytest.approx((100 * 100.0 + 50 * 100.2) / 150)

    delayed = simulate_l1_order(order, quotes, latency_seconds=1.0)
    assert delayed is not None
    assert delayed.first_fill_timestamp_ns == 2_000_000_000
    assert delayed.filled_quantity == pytest.approx(100.0)

    limit = Order(
        asset_id="A",
        decision_timestamp_ns=1_000_000_000,
        side="buy",
        quantity=50,
        order_type="limit",
        limit_price=100.05,
    )
    limit_fill = simulate_l1_order(limit, quotes)
    assert limit_fill is not None
    assert limit_fill.first_fill_timestamp_ns == 1_000_000_000
    assert limit_fill.average_price == pytest.approx(100.0)
