"""Additional security-master consistency tests for identity and symbol changes."""

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


def _record() -> SecurityRecord:
    return SecurityRecord(
        security_id="sec-1",
        security_type=SecurityType.COMMON_STOCK,
        exchange="NASDAQ",
        listing_date=date(2010, 1, 1),
    )


def _valid_master() -> SecurityMaster:
    return SecurityMaster(
        version="sm-v2",
        securities=(_record(),),
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
        ),
        corporate_actions=(
            CorporateAction(
                security_id="sec-1",
                action_type=CorporateActionType.SYMBOL_CHANGE,
                effective_date=date(2019, 1, 1),
                old_symbol="OLD",
                new_symbol="NEW",
            ),
        ),
    )


def test_every_security_requires_point_in_time_symbol_history() -> None:
    with pytest.raises(ValidationError, match="at least one symbol period"):
        SecurityMaster(version="sm-v2", securities=(_record(),), symbols=())


def test_symbol_change_action_must_match_new_symbol_effective_date() -> None:
    payload = _valid_master().model_dump(mode="python")
    payload["corporate_actions"] = (
        CorporateAction(
            security_id="sec-1",
            action_type=CorporateActionType.SYMBOL_CHANGE,
            effective_date=date(2019, 1, 2),
            old_symbol="OLD",
            new_symbol="NEW",
        ),
    )
    with pytest.raises(ValidationError, match="new_symbol/effective_date"):
        SecurityMaster.model_validate(payload)


def test_symbol_change_old_symbol_must_match_immediately_prior_history() -> None:
    payload = _valid_master().model_dump(mode="python")
    payload["corporate_actions"] = (
        CorporateAction(
            security_id="sec-1",
            action_type=CorporateActionType.SYMBOL_CHANGE,
            effective_date=date(2019, 1, 1),
            old_symbol="WRONG",
            new_symbol="NEW",
        ),
    )
    with pytest.raises(ValidationError, match="immediately prior"):
        SecurityMaster.model_validate(payload)


def test_symbol_change_requires_distinct_symbols() -> None:
    with pytest.raises(ValidationError, match="distinct"):
        CorporateAction(
            security_id="sec-1",
            action_type=CorporateActionType.SYMBOL_CHANGE,
            effective_date=date(2019, 1, 1),
            old_symbol="SAME",
            new_symbol="SAME",
        )


def test_symbol_history_is_returned_in_chronological_order() -> None:
    history = _valid_master().symbol_history("sec-1")
    assert [period.symbol for period in history] == ["OLD", "NEW"]
