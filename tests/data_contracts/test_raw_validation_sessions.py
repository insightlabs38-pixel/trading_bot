"""Additional raw-validation coverage for expected-session completeness."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from trading_bot.data.raw_validation import AnomalyCode, RawBar, validate_raw_bars


def _bar(asset_id: str, day: int, minute: int = 30) -> RawBar:
    return RawBar(
        asset_id=asset_id,
        timestamp=datetime(2024, 1, day, 14, minute, tzinfo=UTC),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1000.0,
        vwap=100.2,
    )


def test_expected_session_set_detects_wholly_missing_session_per_asset() -> None:
    report = validate_raw_bars(
        [
            _bar("AAPL", 2),
            _bar("AAPL", 4),
            _bar("MSFT", 2),
            _bar("MSFT", 3),
            _bar("MSFT", 4),
        ],
        expected_sessions=(date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)),
    )

    missing = [item for item in report.anomalies if item.code == AnomalyCode.MISSING_SESSION]
    assert len(missing) == 1
    assert missing[0].asset_id == "AAPL"
    assert missing[0].timestamp is None
    assert "2024-01-03" in missing[0].message


def test_expected_sessions_do_not_invent_assets_absent_from_input() -> None:
    report = validate_raw_bars(
        [_bar("AAPL", 2)],
        expected_sessions=(date(2024, 1, 2), date(2024, 1, 3)),
    )
    missing = [item for item in report.anomalies if item.code == AnomalyCode.MISSING_SESSION]
    assert [item.asset_id for item in missing] == ["AAPL"]


def test_duplicate_expected_session_dates_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate dates"):
        validate_raw_bars(
            [_bar("AAPL", 2)],
            expected_sessions=(date(2024, 1, 2), date(2024, 1, 2)),
        )
