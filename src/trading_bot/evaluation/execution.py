"""CPU reference execution metrics and top-of-book simulator."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from trading_bot.evaluation.contracts import Fill, Order, Quote, Side


@dataclass(frozen=True, slots=True)
class ImplementationShortfallMetrics:
    mean_bps: float
    median_bps: float
    p90_bps: float
    p95_bps: float
    observations: int


@dataclass(frozen=True, slots=True)
class LiquidityDiagnostics:
    order_participation_fraction: float
    position_adv_fraction: float


def implementation_shortfall_bps(
    *,
    side: Side,
    decision_price: float,
    execution_price: float,
) -> float:
    """Return positive basis points for execution worse than the decision price."""
    for name, value in (("decision_price", decision_price), ("execution_price", execution_price)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    sign = 1.0 if side == "buy" else -1.0
    return sign * (execution_price - decision_price) / decision_price * 10_000.0


def summarize_implementation_shortfall(
    values_bps: Sequence[float],
) -> ImplementationShortfallMetrics:
    if not values_bps:
        raise ValueError("implementation-shortfall summary requires observations")
    values = sorted(float(value) for value in values_bps)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("implementation-shortfall values must be finite")
    return ImplementationShortfallMetrics(
        mean_bps=statistics.fmean(values),
        median_bps=statistics.median(values),
        p90_bps=_quantile(values, 0.90),
        p95_bps=_quantile(values, 0.95),
        observations=len(values),
    )


def liquidity_diagnostics(
    *,
    order_notional: float,
    relevant_market_volume_notional: float,
    position_notional: float,
    average_daily_dollar_volume: float,
) -> LiquidityDiagnostics:
    values = {
        "order_notional": order_notional,
        "relevant_market_volume_notional": relevant_market_volume_notional,
        "position_notional": position_notional,
        "average_daily_dollar_volume": average_daily_dollar_volume,
    }
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("liquidity notionals must be finite and non-negative")
    if relevant_market_volume_notional <= 0 or average_daily_dollar_volume <= 0:
        raise ValueError("market-volume and ADV denominators must be positive")
    return LiquidityDiagnostics(
        order_participation_fraction=order_notional / relevant_market_volume_notional,
        position_adv_fraction=position_notional / average_daily_dollar_volume,
    )


def simulate_l1_order(
    order: Order,
    quotes: Sequence[Quote],
    *,
    latency_seconds: float = 0.0,
) -> Fill | None:
    """Simulate market/limit fills using only quotes at or after eligible execution time.

    This is a deterministic reference simulator for medium-frequency tests, not a
    queue-position or hidden-liquidity model.
    """
    if not math.isfinite(latency_seconds) or latency_seconds < 0:
        raise ValueError("latency_seconds must be finite and non-negative")
    eligible_timestamp = order.decision_timestamp_ns + round(latency_seconds * 1_000_000_000)
    ordered_quotes = sorted(quotes, key=lambda quote: quote.timestamp_ns)
    if any(
        ordered_quotes[index].timestamp_ns == ordered_quotes[index - 1].timestamp_ns
        for index in range(1, len(ordered_quotes))
    ):
        raise ValueError("duplicate quote timestamps are not allowed")

    remaining = order.quantity
    notional = 0.0
    filled = 0.0
    first_timestamp: int | None = None
    last_timestamp: int | None = None
    for quote in ordered_quotes:
        if quote.timestamp_ns < eligible_timestamp:
            continue
        if not _quote_is_marketable(order, quote):
            continue
        available = quote.ask_size if order.side == "buy" else quote.bid_size
        if available <= 0:
            continue
        quantity = min(remaining, available)
        price = quote.ask if order.side == "buy" else quote.bid
        notional += quantity * price
        filled += quantity
        remaining -= quantity
        if first_timestamp is None:
            first_timestamp = quote.timestamp_ns
        last_timestamp = quote.timestamp_ns
        if remaining <= 1e-12:
            break

    if filled <= 0 or first_timestamp is None or last_timestamp is None:
        return None
    return Fill(
        asset_id=order.asset_id,
        side=order.side,
        decision_timestamp_ns=order.decision_timestamp_ns,
        first_fill_timestamp_ns=first_timestamp,
        last_fill_timestamp_ns=last_timestamp,
        requested_quantity=order.quantity,
        filled_quantity=filled,
        average_price=notional / filled,
    )


def _quote_is_marketable(order: Order, quote: Quote) -> bool:
    if order.order_type == "market":
        return True
    assert order.limit_price is not None
    if order.side == "buy":
        return quote.ask <= order.limit_price
    return quote.bid >= order.limit_price


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    fraction = index - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
