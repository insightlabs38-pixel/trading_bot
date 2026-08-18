"""Session-aware causal resampling for canonical one-minute bars."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Iterable, Literal
from zoneinfo import ZoneInfo

from trading_bot.data.canonicalization import CanonicalBar


class ResamplingError(RuntimeError):
    """Raised when canonical bars violate the resampling contract."""


Frequency = Literal[5, 15, 30, 60, "1d"]


@dataclass(frozen=True, slots=True)
class SessionSpec:
    timezone: str = "America/New_York"
    open_time: time = time(9, 30)
    close_time: time = time(16, 0)
    base_interval_minutes: int = 1

    def __post_init__(self) -> None:
        if self.close_time <= self.open_time:
            raise ValueError("close_time must be later than open_time")
        if self.base_interval_minutes <= 0:
            raise ValueError("base_interval_minutes must be positive")

    @property
    def expected_session_bars(self) -> int:
        anchor = datetime.combine(datetime.min.date(), self.open_time)
        close = datetime.combine(datetime.min.date(), self.close_time)
        minutes = int((close - anchor).total_seconds() // 60)
        if minutes % self.base_interval_minutes:
            raise ValueError("session length must divide evenly by base interval")
        return minutes // self.base_interval_minutes


@dataclass(frozen=True, slots=True)
class ResampledBar:
    security_id: str
    symbol: str
    timestamp: datetime
    frequency: str
    adjusted_open: float
    adjusted_high: float
    adjusted_low: float
    adjusted_close: float
    adjusted_volume: float
    adjusted_vwap: float | None
    source_count: int
    window_start: datetime
    window_end: datetime
    complete: bool


def resample_canonical_bars(
    bars: Iterable[CanonicalBar],
    frequency: Frequency,
    *,
    session: SessionSpec | None = None,
    require_complete: bool = True,
) -> tuple[ResampledBar, ...]:
    """Aggregate canonical bars without crossing assets, sessions, or future bucket boundaries."""
    session = session or SessionSpec()
    timezone = ZoneInfo(session.timezone)
    rows = tuple(bars)
    seen: set[tuple[str, datetime]] = set()
    grouped: dict[tuple[str, object, int], list[CanonicalBar]] = defaultdict(list)

    for bar in rows:
        if bar.timestamp.tzinfo is None or bar.timestamp.utcoffset() is None:
            raise ResamplingError("canonical timestamps must be timezone-aware")
        identity = (bar.security_id, bar.timestamp)
        if identity in seen:
            raise ResamplingError("duplicate security/timestamp cannot be resampled")
        seen.add(identity)

        local = bar.timestamp.astimezone(timezone)
        session_date = local.date()
        open_dt = datetime.combine(session_date, session.open_time, timezone)
        close_dt = datetime.combine(session_date, session.close_time, timezone)
        if not open_dt <= local < close_dt:
            raise ResamplingError(
                f"bar {bar.security_id}@{bar.timestamp.isoformat()} is outside configured session"
            )
        elapsed_minutes = int((local - open_dt).total_seconds() // 60)
        if frequency == "1d":
            bucket_index = 0
        else:
            if frequency % session.base_interval_minutes:
                raise ValueError("frequency must divide evenly by the base interval")
            bucket_index = elapsed_minutes // frequency
        grouped[(bar.security_id, session_date, bucket_index)].append(bar)

    output: list[ResampledBar] = []
    for (_, _, _), bucket in sorted(
        grouped.items(),
        key=lambda item: min(row.timestamp for row in item[1]),
    ):
        ordered = sorted(bucket, key=lambda row: row.timestamp)
        expected = (
            session.expected_session_bars
            if frequency == "1d"
            else int(frequency) // session.base_interval_minutes
        )
        complete = _is_complete_bucket(ordered, expected, session.base_interval_minutes)
        if require_complete and not complete:
            continue
        output.append(_aggregate_bucket(ordered, frequency, complete))
    return tuple(output)


def _is_complete_bucket(
    rows: list[CanonicalBar],
    expected_count: int,
    base_interval_minutes: int,
) -> bool:
    if len(rows) != expected_count:
        return False
    expected_delta = timedelta(minutes=base_interval_minutes)
    return all(
        current.timestamp - previous.timestamp == expected_delta
        for previous, current in zip(rows, rows[1:], strict=False)
    )


def _aggregate_bucket(
    rows: list[CanonicalBar],
    frequency: Frequency,
    complete: bool,
) -> ResampledBar:
    first = rows[0]
    last = rows[-1]
    if any(row.security_id != first.security_id for row in rows):
        raise ResamplingError("bucket contains multiple securities")
    if any(row.symbol != first.symbol for row in rows):
        raise ResamplingError("symbol changes cannot occur inside a resampling bucket")

    volume = sum(row.adjusted_volume for row in rows)
    weighted = [
        (row.adjusted_vwap, row.adjusted_volume)
        for row in rows
        if row.adjusted_vwap is not None and row.adjusted_volume > 0
    ]
    vwap = None
    if weighted:
        weighted_volume = sum(weight for _, weight in weighted)
        numerator = sum(
            value * weight for value, weight in weighted if value is not None
        )
        vwap = numerator / weighted_volume

    return ResampledBar(
        security_id=first.security_id,
        symbol=first.symbol,
        timestamp=last.timestamp,
        frequency=str(frequency),
        adjusted_open=first.adjusted_open,
        adjusted_high=max(row.adjusted_high for row in rows),
        adjusted_low=min(row.adjusted_low for row in rows),
        adjusted_close=last.adjusted_close,
        adjusted_volume=volume,
        adjusted_vwap=vwap,
        source_count=len(rows),
        window_start=first.timestamp,
        window_end=last.timestamp,
        complete=complete,
    )
