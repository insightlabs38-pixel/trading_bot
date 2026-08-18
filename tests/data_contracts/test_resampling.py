"""Tests for session-aware causal resampling."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, time, timedelta

import pytest

from trading_bot.data.canonicalization import CanonicalBar
from trading_bot.data.resampling import ResamplingError, SessionSpec, resample_canonical_bars


def bar(
    minute: int,
    *,
    day: int = 2,
    security_id: str = "sec-1",
    offset: float = 0.0,
) -> CanonicalBar:
    timestamp = datetime(2024, 1, day, 14, 30, tzinfo=UTC) + timedelta(minutes=minute)
    price = 100.0 + offset + minute
    return CanonicalBar(
        security_id=security_id,
        symbol="AAA" if security_id == "sec-1" else "BBB",
        timestamp=timestamp,
        raw_open=price,
        raw_high=price + 1,
        raw_low=price - 1,
        raw_close=price + 0.5,
        raw_volume=10,
        raw_vwap=price + 0.25,
        adjusted_open=price,
        adjusted_high=price + 1,
        adjusted_low=price - 1,
        adjusted_close=price + 0.5,
        adjusted_volume=10,
        adjusted_vwap=price + 0.25,
        cumulative_split_factor=1,
        cash_dividend_per_share=0,
    )


def test_five_minute_aggregation_has_correct_ohlcv_and_bar_close_timestamp() -> None:
    rows = [bar(index) for index in range(5)]
    result = resample_canonical_bars(rows, 5)
    assert len(result) == 1
    aggregate = result[0]
    assert aggregate.timestamp == rows[-1].timestamp
    assert aggregate.adjusted_open == rows[0].adjusted_open
    assert aggregate.adjusted_high == max(row.adjusted_high for row in rows)
    assert aggregate.adjusted_low == min(row.adjusted_low for row in rows)
    assert aggregate.adjusted_close == rows[-1].adjusted_close
    assert aggregate.adjusted_volume == 50
    assert aggregate.adjusted_vwap == pytest.approx(sum(row.adjusted_vwap for row in rows) / 5)


@pytest.mark.parametrize("frequency", [5, 15, 30, 60])
def test_supported_intraday_frequencies_require_complete_source_bars(frequency: int) -> None:
    rows = [bar(index) for index in range(frequency)]
    result = resample_canonical_bars(rows, frequency)  # type: ignore[arg-type]
    assert len(result) == 1
    assert result[0].source_count == frequency
    assert result[0].complete


def test_incomplete_bucket_is_not_emitted_by_default() -> None:
    assert resample_canonical_bars([bar(0), bar(1), bar(2)], 5) == ()
    partial = resample_canonical_bars([bar(0), bar(1), bar(2)], 5, require_complete=False)
    assert len(partial) == 1
    assert not partial[0].complete


def test_gap_inside_bucket_marks_it_incomplete() -> None:
    rows = [bar(0), bar(1), bar(3), bar(4), bar(5)]
    assert resample_canonical_bars(rows, 5) == ()


def test_cross_session_rows_never_share_a_bucket() -> None:
    short_session = SessionSpec(open_time=time(9, 30), close_time=time(9, 35))
    first_day = [bar(index, day=2) for index in range(5)]
    second_day = [bar(index, day=3) for index in range(5)]
    result = resample_canonical_bars(first_day + second_day, "1d", session=short_session)
    assert len(result) == 2
    assert result[0].window_end.date() != result[1].window_end.date()


def test_daily_aggregate_uses_session_length_not_cross_day_state() -> None:
    short_session = SessionSpec(open_time=time(9, 30), close_time=time(9, 35))
    rows = [bar(index) for index in range(5)]
    result = resample_canonical_bars(rows, "1d", session=short_session)
    assert len(result) == 1
    assert result[0].source_count == 5
    assert result[0].frequency == "1d"


def test_future_bucket_data_does_not_change_completed_prior_bucket() -> None:
    first = [bar(index) for index in range(5)]
    prior = resample_canonical_bars(first, 5)[0]
    extended = resample_canonical_bars(first + [bar(index) for index in range(5, 10)], 5)
    assert extended[0] == prior


def test_new_york_session_alignment_handles_standard_time() -> None:
    result = resample_canonical_bars([bar(index) for index in range(5)], 5)
    assert len(result) == 1


def test_out_of_session_and_duplicate_rows_are_rejected() -> None:
    out_of_session = replace(
        bar(0),
        timestamp=datetime(2024, 1, 2, 22, 0, tzinfo=UTC),
    )
    with pytest.raises(ResamplingError, match="outside configured session"):
        resample_canonical_bars([out_of_session], 5)
    duplicate = bar(0)
    with pytest.raises(ResamplingError, match="duplicate"):
        resample_canonical_bars([duplicate, duplicate], 5, require_complete=False)
