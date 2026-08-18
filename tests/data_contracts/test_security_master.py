"""Tests for point-in-time security identity and corporate-action metadata."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from trading_bot.data.security_master import (
    CorporateAction,
    CorporateActionType,
    SecurityMaster,
    SecurityRecord,
    SecurityType,
    SymbolPeriod,
)


def master() -> SecurityMaster:
    return SecurityMaster(
        version="sm-v1",
        securities=(
            SecurityRecord(
                security_id="sec-1",
                security_type=SecurityType.COMMON_STOCK,
                exchange="NASDAQ",
                listing_date=date(2010, 1, 1),
                issuer_name="Example One",
                sector="Technology",
            ),
            SecurityRecord(
                security_id="sec-2",
                security_type=SecurityType.COMMON_STOCK,
                exchange="NYSE",
                listing_date=date(2012, 1, 1),
                delisting_date=date(2020, 6, 30),
                issuer_name="Delisted Corp",
                sector="Industrials",
            ),
        ),
        symbols=(
            SymbolPeriod(
                security_id="sec-1",
                symbol="OLD",
                start_date=date(2010, 1, 1),
                end_date=date(2018, 12, 31),
            ),
            SymbolPeriod(
                security_id="sec-1",
                symbol="NEW",
                start_date=date(2019, 1, 1),
            ),
            SymbolPeriod(
                security_id="sec-2",
                symbol="GONE",
                start_date=date(2012, 1, 1),
                end_date=date(2020, 6, 30),
            ),
        ),
        corporate_actions=(
            CorporateAction(
                security_id="sec-1",
                action_type=CorporateActionType.SYMBOL_CHANGE,
                effective_date=date(2019, 1, 1),
                old_symbol="OLD",
                new_symbol="NEW",
                source_id="action-1",
            ),
            CorporateAction(
                security_id="sec-1",
                action_type=CorporateActionType.SPLIT,
                effective_date=date(2020, 8, 31),
                split_ratio=4.0,
                source_id="action-2",
            ),
        ),
    )


def test_symbol_history_resolves_point_in_time() -> None:
    value = master()
    assert value.symbol_for("sec-1", date(2018, 6, 1)) == "OLD"
    assert value.symbol_for("sec-1", date(2024, 1, 1)) == "NEW"
    assert value.security_for_symbol("OLD", date(2018, 6, 1)).security_id == "sec-1"


def test_delisted_security_remains_resolvable_historically() -> None:
    value = master()
    assert value.security_for_symbol("GONE", date(2019, 1, 1)).security_id == "sec-2"
    with pytest.raises(KeyError):
        value.security_for_symbol("GONE", date(2021, 1, 1))


def test_active_common_equities_are_point_in_time() -> None:
    value = master()
    assert {item.security_id for item in value.active_common_equities(date(2019, 1, 1))} == {
        "sec-1",
        "sec-2",
    }
    assert {item.security_id for item in value.active_common_equities(date(2021, 1, 1))} == {
        "sec-1"
    }


def test_actions_can_be_queried_only_through_as_of_date() -> None:
    value = master()
    actions = value.actions_for("sec-1", through=date(2019, 12, 31))
    assert [action.action_type for action in actions] == [CorporateActionType.SYMBOL_CHANGE]


def test_overlapping_symbol_periods_for_same_security_are_rejected() -> None:
    payload = master().model_dump(mode="python")
    payload["symbols"] = tuple(payload["symbols"]) + (
        SymbolPeriod(
            security_id="sec-1",
            symbol="OVERLAP",
            start_date=date(2018, 6, 1),
            end_date=date(2019, 6, 1),
        ),
    )
    with pytest.raises(ValidationError, match="overlapping symbol periods"):
        SecurityMaster.model_validate(payload)


def test_same_ticker_cannot_overlap_across_unrelated_securities() -> None:
    payload = master().model_dump(mode="python")
    payload["securities"] = tuple(payload["securities"]) + (
        SecurityRecord(
            security_id="sec-3",
            security_type=SecurityType.COMMON_STOCK,
            exchange="NYSE",
            listing_date=date(2019, 1, 1),
        ),
    )
    payload["symbols"] = tuple(payload["symbols"]) + (
        SymbolPeriod(
            security_id="sec-3",
            symbol="NEW",
            start_date=date(2019, 1, 1),
        ),
    )
    with pytest.raises(ValidationError, match="same symbol overlaps"):
        SecurityMaster.model_validate(payload)


def test_symbol_period_must_respect_listing_boundaries() -> None:
    payload = master().model_dump(mode="python")
    payload["symbols"] = tuple(payload["symbols"]) + (
        SymbolPeriod(
            security_id="sec-2",
            symbol="PRE",
            start_date=date(2011, 1, 1),
            end_date=date(2011, 12, 31),
        ),
    )
    with pytest.raises(ValidationError, match="before listing_date"):
        SecurityMaster.model_validate(payload)


def test_split_and_dividend_require_action_specific_values() -> None:
    with pytest.raises(ValidationError, match="split_ratio"):
        CorporateAction(
            security_id="sec-1",
            action_type=CorporateActionType.SPLIT,
            effective_date=date(2020, 1, 1),
        )
    with pytest.raises(ValidationError, match="cash_amount"):
        CorporateAction(
            security_id="sec-1",
            action_type=CorporateActionType.CASH_DIVIDEND,
            effective_date=date(2020, 1, 1),
        )


def test_exchange_security_type_and_sector_metadata_are_preserved() -> None:
    record = master().get_security("sec-1")
    assert record.exchange == "NASDAQ"
    assert record.security_type == SecurityType.COMMON_STOCK
    assert record.sector == "Technology"
