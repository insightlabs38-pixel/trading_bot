"""Causal reference feature pipeline for medium-frequency equity panels."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Iterable
from zoneinfo import ZoneInfo


class FeaturePipelineError(RuntimeError):
    """Raised when feature inputs violate chronological or panel contracts."""


@dataclass(frozen=True, slots=True)
class FeatureObservation:
    security_id: str
    symbol: str
    sector: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        if not self.security_id or not self.symbol:
            raise ValueError("security_id and symbol must not be blank")
        prices = (self.open, self.high, self.low, self.close)
        if any(not math.isfinite(value) or value <= 0 for value in prices):
            raise ValueError("OHLC values must be finite and positive")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("high/low must contain open and close")
        if not math.isfinite(self.volume) or self.volume < 0:
            raise ValueError("volume must be finite and non-negative")
        if self.vwap is not None and (not math.isfinite(self.vwap) or self.vwap <= 0):
            raise ValueError("vwap must be finite and positive when present")


@dataclass(frozen=True, slots=True)
class FeaturePolicy:
    return_horizons: tuple[int, ...] = (1, 5, 15, 30, 60)
    relative_volume_window: int = 20
    volatility_window: int = 20
    range_window: int = 14
    momentum_window: int = 20
    timezone: str = "America/New_York"
    session_open: time = time(9, 30)

    def __post_init__(self) -> None:
        if not self.return_horizons or any(horizon <= 0 for horizon in self.return_horizons):
            raise ValueError("return horizons must be non-empty and positive")
        if len(set(self.return_horizons)) != len(self.return_horizons):
            raise ValueError("return horizons must be unique")
        if min(
            self.relative_volume_window,
            self.volatility_window,
            self.range_window,
            self.momentum_window,
        ) <= 0:
            raise ValueError("feature windows must be positive")


@dataclass(frozen=True, slots=True)
class FeatureRow:
    security_id: str
    symbol: str
    sector: str
    timestamp: datetime
    values: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class _History:
    closes: list[float] = field(default_factory=list)
    volumes: list[float] = field(default_factory=list)
    returns_1: list[float] = field(default_factory=list)
    true_ranges: list[float] = field(default_factory=list)


def compute_features(
    observations: Iterable[FeatureObservation],
    *,
    policy: FeaturePolicy | None = None,
) -> tuple[FeatureRow, ...]:
    """Compute reference causal features without relying on future panel observations."""
    policy = policy or FeaturePolicy()
    rows = tuple(observations)
    _validate_unique(rows)
    by_timestamp: dict[datetime, list[FeatureObservation]] = defaultdict(list)
    for row in rows:
        by_timestamp[row.timestamp].append(row)

    histories: dict[str, _History] = defaultdict(_History)
    output: list[FeatureRow] = []
    for timestamp in sorted(by_timestamp):
        panel = sorted(by_timestamp[timestamp], key=lambda row: row.security_id)
        prelim: list[tuple[FeatureObservation, dict[str, float]]] = []
        for row in panel:
            values = _security_features(row, histories[row.security_id], policy)
            _ensure_finite_values(row, values)
            prelim.append((row, values))
        _append_cross_sectional_features(prelim)
        for row, values in prelim:
            _ensure_finite_values(row, values)
            output.append(FeatureRow(row.security_id, row.symbol, row.sector, timestamp, values))
        for row, values in prelim:
            _advance_history(row, values, histories[row.security_id])
    return tuple(output)


def _validate_unique(rows: tuple[FeatureObservation, ...]) -> None:
    seen: set[tuple[str, datetime]] = set()
    for row in rows:
        identity = (row.security_id, row.timestamp)
        if identity in seen:
            raise FeaturePipelineError("duplicate security/timestamp feature observation")
        seen.add(identity)


def _security_features(
    row: FeatureObservation,
    history: _History,
    policy: FeaturePolicy,
) -> dict[str, float]:
    values: dict[str, float] = {
        "open": row.open,
        "high": row.high,
        "low": row.low,
        "close": row.close,
        "volume": row.volume,
        "dollar_volume": row.close * row.volume,
        "range_fraction": (row.high - row.low) / row.close,
        "close_to_open": row.close / row.open - 1.0,
        "close_to_vwap": 0.0 if row.vwap is None else row.close / row.vwap - 1.0,
    }
    for horizon in policy.return_horizons:
        values[f"return_{horizon}"] = _trailing_return(history.closes, row.close, horizon)

    prior_average_volume = _mean_tail(history.volumes, policy.relative_volume_window)
    values["relative_volume"] = (
        0.0
        if prior_average_volume is None or prior_average_volume == 0
        else row.volume / prior_average_volume
    )
    values["realized_volatility"] = _sample_std(history.returns_1[-policy.volatility_window :])

    previous_close = history.closes[-1] if history.closes else row.close
    true_range = max(
        row.high - row.low,
        abs(row.high - previous_close),
        abs(row.low - previous_close),
    )
    values["true_range_fraction"] = true_range / row.close
    prior_ranges = history.true_ranges[-policy.range_window :]
    values["average_true_range_fraction"] = (
        sum(prior_ranges) / len(prior_ranges) if prior_ranges else values["true_range_fraction"]
    )
    values["momentum"] = _trailing_return(history.closes, row.close, policy.momentum_window)
    values["trend_slope"] = _normalized_slope(
        history.closes[-(policy.momentum_window - 1) :] + [row.close]
    )
    values["log_dollar_volume"] = math.log1p(values["dollar_volume"])
    values.update(_time_features(row.timestamp, policy))
    return values


def _append_cross_sectional_features(
    prelim: list[tuple[FeatureObservation, dict[str, float]]],
) -> None:
    one_bar_returns = [values.get("return_1", 0.0) for _, values in prelim]
    market_return = sum(one_bar_returns) / len(one_bar_returns) if one_bar_returns else 0.0
    dispersion = statistics.pstdev(one_bar_returns) if len(one_bar_returns) > 1 else 0.0
    breadth = (
        sum(value > 0 for value in one_bar_returns) / len(one_bar_returns)
        if one_bar_returns
        else 0.0
    )
    ranks = _percentile_ranks(one_bar_returns)
    sectors: dict[str, list[float]] = defaultdict(list)
    for row, values in prelim:
        sectors[row.sector or "UNKNOWN"].append(values.get("return_1", 0.0))
    sector_means = {key: sum(values) / len(values) for key, values in sectors.items()}

    for index, (row, values) in enumerate(prelim):
        one_bar = values.get("return_1", 0.0)
        sector_return = sector_means[row.sector or "UNKNOWN"]
        values["market_return_1"] = market_return
        values["market_relative_return_1"] = one_bar - market_return
        values["sector_return_1"] = sector_return
        values["sector_relative_return_1"] = one_bar - sector_return
        values["cross_sectional_return_rank"] = ranks[index]
        values["market_breadth"] = breadth
        values["market_dispersion"] = dispersion
        values["market_cross_sectional_volatility"] = dispersion


def _ensure_finite_values(row: FeatureObservation, values: dict[str, float]) -> None:
    invalid = sorted(name for name, value in values.items() if not math.isfinite(value))
    if invalid:
        names = ", ".join(invalid)
        raise FeaturePipelineError(
            f"non-finite feature(s) for {row.security_id}@{row.timestamp.isoformat()}: {names}"
        )


def _advance_history(row: FeatureObservation, values: dict[str, float], history: _History) -> None:
    previous_close = history.closes[-1] if history.closes else row.close
    history.closes.append(row.close)
    history.volumes.append(row.volume)
    history.returns_1.append(values.get("return_1", 0.0))
    true_range = max(
        row.high - row.low,
        abs(row.high - previous_close),
        abs(row.low - previous_close),
    )
    history.true_ranges.append(true_range / row.close)


def _trailing_return(closes: list[float], current: float, horizon: int) -> float:
    if len(closes) < horizon:
        return 0.0
    return current / closes[-horizon] - 1.0


def _mean_tail(values: list[float], window: int) -> float | None:
    selected = values[-window:]
    return None if not selected else sum(selected) / len(selected)


def _sample_std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _normalized_slope(values: list[float]) -> float:
    if len(values) < 2 or values[0] == 0:
        return 0.0
    count = len(values)
    mean_x = (count - 1) / 2
    mean_y = sum(values) / count
    denominator = sum((index - mean_x) ** 2 for index in range(count))
    if denominator == 0:
        return 0.0
    numerator = sum((index - mean_x) * (value - mean_y) for index, value in enumerate(values))
    return (numerator / denominator) / values[0]


def _percentile_ranks(values: list[float]) -> list[float]:
    if not values:
        return []
    sorted_values = sorted(set(values))
    if len(sorted_values) == 1:
        return [0.5] * len(values)
    mapping = {value: index / (len(sorted_values) - 1) for index, value in enumerate(sorted_values)}
    return [mapping[value] for value in values]


def _time_features(timestamp: datetime, policy: FeaturePolicy) -> dict[str, float]:
    timezone = ZoneInfo(policy.timezone)
    local = timestamp.astimezone(timezone)
    session_minutes = (
        local.hour * 60 + local.minute - policy.session_open.hour * 60 - policy.session_open.minute
    )
    minute_angle = 2 * math.pi * session_minutes / 390
    weekday_angle = 2 * math.pi * local.weekday() / 5
    return {
        "minute_of_day_sin": math.sin(minute_angle),
        "minute_of_day_cos": math.cos(minute_angle),
        "day_of_week_sin": math.sin(weekday_angle),
        "day_of_week_cos": math.cos(weekday_angle),
    }
