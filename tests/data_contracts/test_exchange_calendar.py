"""Tests for production exchange-calendar session handling."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

import pytest

from trading_bot.data.calendar import CalendarSessionError, ExchangeCalendarSessionProvider
from trading_bot.data.canonicalization import CanonicalBar
from trading_bot.data.resampling import ResamplingError, resample_canonical_bars


def _bar(timestamp: datetime, index: int = 0) -> CanonicalBar:
    price = 100.0 + index / 100
    return CanonicalBar(
        security_id="sec-1",
        symbol="AAA",
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


def test_xnys_sessions_exclude_holidays_and_capture_early_closes() -> None:
    provider = ExchangeCalendarSessionProvider("XNYS")
    assert provider.session_dates(date(2024, 7, 1), date(2024, 7, 5)) == (
        date(2024, 7, 1),
        date(2024, 7, 2),
        date(2024, 7, 3),
        date(2024, 7, 5),
    )

    july_third = provider.session_spec(date(2024, 7, 3))
    july_fifth = provider.session_spec(date(2024, 7, 5))
    black_friday = provider.session_spec(date(2024, 11, 29))
    assert july_third.open_time == time(9, 30)
    assert july_third.close_time == time(13, 0)
    assert july_third.expected_session_bars == 210
    assert july_fifth.close_time == time(16, 0)
    assert july_fifth.expected_session_bars == 390
    assert black_friday.close_time == time(13, 0)


def test_non_session_date_fails_closed() -> None:
    provider = ExchangeCalendarSessionProvider("XNYS")
    with pytest.raises(CalendarSessionError, match="not a XNYS trading session"):
        provider.session_spec(date(2024, 7, 4))


def test_daily_resampling_uses_early_close_session_length() -> None:
    provider = ExchangeCalendarSessionProvider("XNYS")
    start = datetime(2024, 7, 3, 13, 30, tzinfo=UTC)
    rows = [_bar(start + timedelta(minutes=index), index) for index in range(210)]

    result = resample_canonical_bars(
        rows,
        "1d",
        session_resolver=provider.session_spec,
    )
    assert len(result) == 1
    assert result[0].complete
    assert result[0].source_count == 210


def test_resampling_rejects_bars_on_exchange_holiday() -> None:
    provider = ExchangeCalendarSessionProvider("XNYS")
    holiday_bar = _bar(datetime(2024, 7, 4, 13, 30, tzinfo=UTC))
    with pytest.raises(ResamplingError, match="no valid trading session"):
        resample_canonical_bars(
            [holiday_bar],
            5,
            session_resolver=provider.session_spec,
            require_complete=False,
        )
