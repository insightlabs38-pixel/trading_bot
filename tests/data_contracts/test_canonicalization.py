"""Tests for causal split/dividend canonicalization."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from trading_bot.data.canonicalization import (
    CanonicalizationError,
    canonicalize_bars,
    total_return_between,
)
from trading_bot.data.raw_validation import RawBar
from trading_bot.data.security_master import (
    CorporateAction,
    CorporateActionType,
    SecurityMaster,
    SecurityRecord,
    SecurityType,
    SymbolPeriod,
)


def security_master(*, include_future_split: bool = False) -> SecurityMaster:
    actions = [
        CorporateAction(
            security_id="sec-1",
            action_type=CorporateActionType.SPLIT,
            effective_date=date(2020, 8, 31),
            split_ratio=4.0,
        ),
        CorporateAction(
            security_id="sec-1",
            action_type=CorporateActionType.CASH_DIVIDEND,
            effective_date=date(2020, 9, 1),
            cash_amount=0.25,
        ),
        CorporateAction(
            security_id="sec-1",
            action_type=CorporateActionType.SYMBOL_CHANGE,
            effective_date=date(2020, 9, 2),
            old_symbol="OLD",
            new_symbol="NEW",
        ),
    ]
    if include_future_split:
        actions.append(
            CorporateAction(
                security_id="sec-1",
                action_type=CorporateActionType.SPLIT,
                effective_date=date(2030, 1, 1),
                split_ratio=2.0,
            )
        )
    return SecurityMaster(
        version="sm-v1",
        securities=(
            SecurityRecord(
                security_id="sec-1",
                security_type=SecurityType.COMMON_STOCK,
                exchange="NASDAQ",
                listing_date=date(2010, 1, 1),
            ),
        ),
        symbols=(
            SymbolPeriod(
                security_id="sec-1",
                symbol="OLD",
                start_date=date(2010, 1, 1),
                end_date=date(2020, 9, 1),
            ),
            SymbolPeriod(
                security_id="sec-1",
                symbol="NEW",
                start_date=date(2020, 9, 2),
            ),
        ),
        corporate_actions=tuple(actions),
    )


def test_raw_values_are_preserved_alongside_adjusted_values() -> None:
    before = RawBar(
        asset_id="sec-1",
        timestamp=datetime(2020, 8, 30, 14, 30, tzinfo=UTC),
        open=100,
        high=101,
        low=99,
        close=100,
        volume=400,
        vwap=100,
    )
    result = canonicalize_bars([before], security_master())[0]
    assert result.raw_close == 100
    assert result.adjusted_close == 100
    assert before.close == 100


def test_split_factor_is_applied_only_on_and_after_effective_date() -> None:
    bars = [
        RawBar("sec-1", datetime(2020, 8, 30, 14, 30, tzinfo=UTC), 100, 101, 99, 100, 400, 100),
        RawBar("sec-1", datetime(2020, 8, 31, 14, 30, tzinfo=UTC), 25, 26, 24, 25, 1600, 25),
    ]
    before, after = canonicalize_bars(bars, security_master())
    assert before.cumulative_split_factor == 1.0
    assert after.cumulative_split_factor == 4.0
    assert after.adjusted_close == pytest.approx(100.0)
    assert after.adjusted_volume == pytest.approx(400.0)


def test_future_split_does_not_change_historical_canonical_values() -> None:
    bar = RawBar(
        "sec-1",
        datetime(2020, 9, 1, 14, 30, tzinfo=UTC),
        25,
        26,
        24,
        25,
        1600,
        25,
    )
    without_future = canonicalize_bars([bar], security_master())[0]
    with_future = canonicalize_bars([bar], security_master(include_future_split=True))[0]
    assert without_future == with_future


def test_cash_dividend_is_explicit_and_total_return_helper_uses_it() -> None:
    previous = RawBar(
        "sec-1",
        datetime(2020, 8, 31, 14, 30, tzinfo=UTC),
        25,
        25,
        25,
        25,
        1600,
        25,
    )
    current = RawBar(
        "sec-1",
        datetime(2020, 9, 1, 14, 30, tzinfo=UTC),
        24.75,
        24.75,
        24.75,
        24.75,
        1600,
        24.75,
    )
    first, second = canonicalize_bars([previous, current], security_master())
    assert second.cash_dividend_per_share == 0.25
    assert total_return_between(first, second) == pytest.approx(0.0)


def test_symbol_change_uses_point_in_time_security_master_symbol() -> None:
    bar = RawBar(
        "sec-1",
        datetime(2020, 9, 2, 14, 30, tzinfo=UTC),
        25,
        26,
        24,
        25,
        1600,
        25,
    )
    result = canonicalize_bars([bar], security_master())[0]
    assert result.security_id == "sec-1"
    assert result.symbol == "NEW"


def test_unknown_or_naive_bar_is_rejected() -> None:
    unknown = RawBar(
        "missing",
        datetime(2020, 9, 1, 14, 30, tzinfo=UTC),
        1,
        1,
        1,
        1,
        1,
    )
    with pytest.raises(CanonicalizationError, match="cannot resolve"):
        canonicalize_bars([unknown], security_master())
    naive = RawBar("sec-1", datetime(2020, 9, 1, 14, 30), 1, 1, 1, 1, 1)
    with pytest.raises(CanonicalizationError, match="timezone-aware"):
        canonicalize_bars([naive], security_master())
