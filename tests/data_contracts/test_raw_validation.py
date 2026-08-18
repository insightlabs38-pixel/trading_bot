"""Tests for non-destructive raw OHLCV validation."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta, timezone

from trading_bot.data.raw_validation import AnomalyCode, RawBar, validate_raw_bars


def bar(minute: int, **overrides: object) -> RawBar:
    payload: dict[str, object] = {
        "asset_id": "AAPL",
        "timestamp": datetime(2024, 1, 2, 14, minute, tzinfo=UTC),
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000.0,
        "vwap": 100.2,
    }
    payload.update(overrides)
    return RawBar(**payload)  # type: ignore[arg-type]


def test_clean_contiguous_bars_pass() -> None:
    report = validate_raw_bars([bar(30), bar(31), bar(32)])
    assert report.is_valid
    assert report.total_rows == 3


def test_naive_and_non_utc_timestamps_are_reported() -> None:
    naive = bar(30, timestamp=datetime(2024, 1, 2, 14, 30))
    eastern = bar(31, timestamp=datetime(2024, 1, 2, 9, 31, tzinfo=timezone(timedelta(hours=-5))))
    report = validate_raw_bars([naive, eastern], detect_missing_intervals=False)
    assert report.count(AnomalyCode.NAIVE_TIMESTAMP) == 1
    assert report.count(AnomalyCode.NON_UTC_TIMESTAMP) == 1


def test_duplicate_rows_are_reported_without_deduplication() -> None:
    duplicate = bar(30)
    rows = [duplicate, duplicate]
    report = validate_raw_bars(rows, detect_missing_intervals=False)
    assert report.total_rows == 2
    assert report.count(AnomalyCode.DUPLICATE_BAR) == 1
    assert rows == [duplicate, duplicate]


def test_impossible_ohlc_relationships_are_reported() -> None:
    report = validate_raw_bars(
        [bar(30, high=99.5), bar(31, low=101.0)],
        detect_missing_intervals=False,
    )
    assert report.count(AnomalyCode.INVALID_OHLC) == 2


def test_corrupt_prices_volume_and_vwap_are_reported() -> None:
    report = validate_raw_bars(
        [
            bar(30, close=math.nan),
            bar(31, volume=-1.0),
            bar(32, vwap=math.inf),
        ],
        detect_missing_intervals=False,
    )
    assert report.count(AnomalyCode.INVALID_PRICE) == 1
    assert report.count(AnomalyCode.INVALID_VOLUME) == 1
    assert report.count(AnomalyCode.INVALID_VWAP) == 1


def test_missing_intraday_intervals_are_reported_per_asset_and_date() -> None:
    report = validate_raw_bars([bar(30), bar(33)])
    assert report.count(AnomalyCode.MISSING_INTERVAL) == 1
    anomaly = next(item for item in report.anomalies if item.code == AnomalyCode.MISSING_INTERVAL)
    assert "2 missing interval" in anomaly.message


def test_cross_date_gap_is_not_treated_as_missing_intraday_intervals() -> None:
    next_day = bar(30, timestamp=datetime(2024, 1, 3, 14, 30, tzinfo=UTC))
    report = validate_raw_bars([bar(30), next_day])
    assert report.count(AnomalyCode.MISSING_INTERVAL) == 0


def test_out_of_order_asset_rows_are_reported() -> None:
    report = validate_raw_bars([bar(31), bar(30)], detect_missing_intervals=False)
    assert report.count(AnomalyCode.OUT_OF_ORDER) == 1


def test_multiple_assets_do_not_create_cross_asset_gap_flags() -> None:
    msft = bar(35, asset_id="MSFT")
    report = validate_raw_bars([bar(30), msft])
    assert report.count(AnomalyCode.MISSING_INTERVAL) == 0
