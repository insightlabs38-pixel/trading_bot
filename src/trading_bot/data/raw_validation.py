"""Non-destructive validation for raw OHLCV bars before canonicalization."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from typing import Iterable


@dataclass(frozen=True, slots=True)
class RawBar:
    """Provider-neutral raw OHLCV bar used by the validation boundary."""

    asset_id: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None


class AnomalyCode(StrEnum):
    NAIVE_TIMESTAMP = "naive_timestamp"
    NON_UTC_TIMESTAMP = "non_utc_timestamp"
    DUPLICATE_BAR = "duplicate_bar"
    INVALID_OHLC = "invalid_ohlc"
    INVALID_PRICE = "invalid_price"
    INVALID_VOLUME = "invalid_volume"
    INVALID_VWAP = "invalid_vwap"
    MISSING_SESSION = "missing_session"
    MISSING_INTERVAL = "missing_interval"
    OUT_OF_ORDER = "out_of_order"


@dataclass(frozen=True, slots=True)
class RawDataAnomaly:
    code: AnomalyCode
    asset_id: str
    timestamp: datetime | None
    message: str


@dataclass(frozen=True, slots=True)
class RawValidationReport:
    total_rows: int
    anomaly_counts: dict[AnomalyCode, int]
    anomalies: tuple[RawDataAnomaly, ...]

    @property
    def is_valid(self) -> bool:
        return not self.anomalies

    def count(self, code: AnomalyCode) -> int:
        return self.anomaly_counts.get(code, 0)


def validate_raw_bars(
    bars: Iterable[RawBar],
    *,
    expected_interval: timedelta = timedelta(minutes=1),
    detect_missing_intervals: bool = True,
    expected_sessions: Iterable[date] | None = None,
    expected_assets: Iterable[str] | None = None,
) -> RawValidationReport:
    """Inspect raw bars and report anomalies without mutating or repairing input rows.

    ``expected_sessions`` is intentionally supplied by the caller instead of embedding an exchange
    calendar here. This keeps raw validation provider/calendar independent while still allowing an
    acquisition run to prove that an expected trading session is wholly absent. If
    ``expected_assets`` is also supplied, assets with no raw rows at all can be identified instead
    of disappearing from completeness checks. Session dates are interpreted at the UTC boundary.
    """
    if expected_interval <= timedelta(0):
        raise ValueError("expected_interval must be positive")

    session_dates = None if expected_sessions is None else tuple(expected_sessions)
    if session_dates is not None and len(set(session_dates)) != len(session_dates):
        raise ValueError("expected_sessions must not contain duplicate dates")

    asset_ids = None if expected_assets is None else tuple(item.strip() for item in expected_assets)
    if asset_ids is not None:
        if session_dates is None:
            raise ValueError("expected_assets requires expected_sessions")
        if any(not item for item in asset_ids):
            raise ValueError("expected_assets must not contain blank identifiers")
        if len(set(asset_ids)) != len(asset_ids):
            raise ValueError("expected_assets must not contain duplicate identifiers")

    rows = tuple(bars)
    anomalies: list[RawDataAnomaly] = []
    seen: Counter[tuple[str, datetime]] = Counter()
    by_asset_session: dict[tuple[str, date], list[datetime]] = defaultdict(list)
    previous_by_asset: dict[str, datetime] = {}
    observed_assets: set[str] = set()

    for bar in rows:
        timestamp = bar.timestamp
        observed_assets.add(bar.asset_id)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            anomalies.append(
                RawDataAnomaly(
                    AnomalyCode.NAIVE_TIMESTAMP,
                    bar.asset_id,
                    timestamp,
                    "timestamp must be timezone-aware",
                )
            )
        else:
            if timestamp.utcoffset() != timedelta(0):
                anomalies.append(
                    RawDataAnomaly(
                        AnomalyCode.NON_UTC_TIMESTAMP,
                        bar.asset_id,
                        timestamp,
                        "raw timestamp must be normalized to UTC at the validation boundary",
                    )
                )
            utc_timestamp = timestamp.astimezone(UTC)
            key = (bar.asset_id, utc_timestamp)
            seen[key] += 1
            if seen[key] > 1:
                anomalies.append(
                    RawDataAnomaly(
                        AnomalyCode.DUPLICATE_BAR,
                        bar.asset_id,
                        timestamp,
                        "duplicate asset/timestamp bar",
                    )
                )
            previous = previous_by_asset.get(bar.asset_id)
            if previous is not None and utc_timestamp < previous:
                anomalies.append(
                    RawDataAnomaly(
                        AnomalyCode.OUT_OF_ORDER,
                        bar.asset_id,
                        timestamp,
                        "bar is earlier than the prior row for this asset",
                    )
                )
            previous_by_asset[bar.asset_id] = utc_timestamp
            by_asset_session[(bar.asset_id, utc_timestamp.date())].append(utc_timestamp)

        price_values = (bar.open, bar.high, bar.low, bar.close)
        if any(not math.isfinite(value) or value <= 0 for value in price_values):
            anomalies.append(
                RawDataAnomaly(
                    AnomalyCode.INVALID_PRICE,
                    bar.asset_id,
                    timestamp,
                    "OHLC prices must be finite and strictly positive",
                )
            )
        elif bar.low > min(bar.open, bar.close) or bar.high < max(bar.open, bar.close):
            anomalies.append(
                RawDataAnomaly(
                    AnomalyCode.INVALID_OHLC,
                    bar.asset_id,
                    timestamp,
                    "low/high do not contain open and close",
                )
            )
        elif bar.high < bar.low:
            anomalies.append(
                RawDataAnomaly(
                    AnomalyCode.INVALID_OHLC,
                    bar.asset_id,
                    timestamp,
                    "high is below low",
                )
            )

        if not math.isfinite(bar.volume) or bar.volume < 0:
            anomalies.append(
                RawDataAnomaly(
                    AnomalyCode.INVALID_VOLUME,
                    bar.asset_id,
                    timestamp,
                    "volume must be finite and non-negative",
                )
            )
        if bar.vwap is not None and (not math.isfinite(bar.vwap) or bar.vwap <= 0):
            anomalies.append(
                RawDataAnomaly(
                    AnomalyCode.INVALID_VWAP,
                    bar.asset_id,
                    timestamp,
                    "VWAP must be finite and positive when present",
                )
            )

    if session_dates is not None:
        completeness_assets = observed_assets if asset_ids is None else set(asset_ids)
        _append_missing_session_anomalies(
            completeness_assets,
            set(session_dates),
            by_asset_session,
            anomalies,
        )
    if detect_missing_intervals:
        _append_missing_interval_anomalies(by_asset_session, expected_interval, anomalies)

    counts = Counter(anomaly.code for anomaly in anomalies)
    return RawValidationReport(
        total_rows=len(rows),
        anomaly_counts=dict(counts),
        anomalies=tuple(anomalies),
    )


def _append_missing_session_anomalies(
    assets: set[str],
    expected_sessions: set[date],
    grouped: dict[tuple[str, date], list[datetime]],
    anomalies: list[RawDataAnomaly],
) -> None:
    observed = set(grouped)
    for asset_id in sorted(assets):
        for session_date in sorted(expected_sessions):
            if (asset_id, session_date) in observed:
                continue
            anomalies.append(
                RawDataAnomaly(
                    AnomalyCode.MISSING_SESSION,
                    asset_id,
                    None,
                    f"expected session {session_date.isoformat()} has no raw bars",
                )
            )


def _append_missing_interval_anomalies(
    grouped: dict[tuple[str, date], list[datetime]],
    expected_interval: timedelta,
    anomalies: list[RawDataAnomaly],
) -> None:
    for (asset_id, _), timestamps in grouped.items():
        unique = sorted(set(timestamps))
        for previous, current in zip(unique, unique[1:], strict=False):
            gap = current - previous
            if gap > expected_interval:
                missing_count = max(1, int(gap / expected_interval) - 1)
                anomalies.append(
                    RawDataAnomaly(
                        AnomalyCode.MISSING_INTERVAL,
                        asset_id,
                        current,
                        f"gap of {gap} implies {missing_count} missing interval(s)",
                    )
                )
