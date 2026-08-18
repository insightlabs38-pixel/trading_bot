"""Point-in-time security-master, symbol history, and corporate-action contracts."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SecurityType(StrEnum):
    COMMON_STOCK = "common_stock"
    ADR = "adr"
    ETF = "etf"
    PREFERRED = "preferred"
    OTHER = "other"


class CorporateActionType(StrEnum):
    SPLIT = "split"
    CASH_DIVIDEND = "cash_dividend"
    SYMBOL_CHANGE = "symbol_change"
    MERGER = "merger"
    SPINOFF = "spinoff"
    OTHER = "other"


class SecurityRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    security_id: str = Field(min_length=1)
    security_type: SecurityType
    exchange: str = Field(min_length=1)
    listing_date: date
    delisting_date: date | None = None
    issuer_name: str | None = None
    sector: str | None = None

    @model_validator(mode="after")
    def validate_dates(self) -> SecurityRecord:
        if self.delisting_date is not None and self.delisting_date < self.listing_date:
            raise ValueError("delisting_date cannot precede listing_date")
        return self

    def is_listed_on(self, as_of: date) -> bool:
        return self.listing_date <= as_of and (
            self.delisting_date is None or as_of <= self.delisting_date
        )


class SymbolPeriod(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    security_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    start_date: date
    end_date: date | None = None

    @model_validator(mode="after")
    def validate_dates(self) -> SymbolPeriod:
        if self.end_date is not None and self.end_date < self.start_date:
            raise ValueError("symbol period end_date cannot precede start_date")
        return self

    def contains(self, as_of: date) -> bool:
        return self.start_date <= as_of and (self.end_date is None or as_of <= self.end_date)


class CorporateAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    security_id: str = Field(min_length=1)
    action_type: CorporateActionType
    effective_date: date
    split_ratio: float | None = Field(default=None, gt=0)
    cash_amount: float | None = Field(default=None, ge=0)
    old_symbol: str | None = None
    new_symbol: str | None = None
    source_id: str | None = None

    @model_validator(mode="after")
    def validate_action_fields(self) -> CorporateAction:
        if self.split_ratio is not None and not math.isfinite(self.split_ratio):
            raise ValueError("split_ratio must be finite")
        if self.cash_amount is not None and not math.isfinite(self.cash_amount):
            raise ValueError("cash_amount must be finite")
        if self.action_type == CorporateActionType.SPLIT and self.split_ratio is None:
            raise ValueError("split action requires split_ratio")
        if self.action_type == CorporateActionType.CASH_DIVIDEND and self.cash_amount is None:
            raise ValueError("cash dividend action requires cash_amount")
        if self.action_type == CorporateActionType.SYMBOL_CHANGE:
            if not self.old_symbol or not self.new_symbol:
                raise ValueError("symbol change requires old_symbol and new_symbol")
            if self.old_symbol == self.new_symbol:
                raise ValueError("symbol change requires distinct old_symbol and new_symbol")
        return self


class SecurityMaster(BaseModel):
    """Immutable point-in-time reference dataset keyed by permanent security ID."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(min_length=1)
    securities: tuple[SecurityRecord, ...]
    symbols: tuple[SymbolPeriod, ...]
    corporate_actions: tuple[CorporateAction, ...] = ()

    @model_validator(mode="after")
    def validate_consistency(self) -> SecurityMaster:
        records = {record.security_id: record for record in self.securities}
        if len(records) != len(self.securities):
            raise ValueError("security IDs must be unique")

        periods_by_security: dict[str, list[SymbolPeriod]] = defaultdict(list)
        periods_by_symbol: dict[str, list[SymbolPeriod]] = defaultdict(list)
        for period in self.symbols:
            record = records.get(period.security_id)
            if record is None:
                raise ValueError(f"symbol period references unknown security {period.security_id}")
            if period.start_date < record.listing_date:
                raise ValueError("symbol period cannot start before listing_date")
            if record.delisting_date is not None:
                period_end = period.end_date or record.delisting_date
                if period_end > record.delisting_date:
                    raise ValueError("symbol period cannot extend beyond delisting_date")
            periods_by_security[period.security_id].append(period)
            periods_by_symbol[period.symbol].append(period)

        for security_id, record in records.items():
            periods = periods_by_security.get(security_id)
            if not periods:
                raise ValueError(f"security {security_id} must have at least one symbol period")
            _validate_symbol_lifetime(record, periods)
        for periods in periods_by_security.values():
            _assert_non_overlapping(periods, "overlapping symbol periods for one security")
        for periods in periods_by_symbol.values():
            _assert_non_overlapping(periods, "same symbol overlaps across securities")

        action_documents: set[str] = set()
        source_ids: set[tuple[str, str]] = set()
        for action in self.corporate_actions:
            record = records.get(action.security_id)
            if record is None:
                message = f"corporate action references unknown security {action.security_id}"
                raise ValueError(message)
            if not record.is_listed_on(action.effective_date):
                raise ValueError("corporate action must fall within listing/delisting dates")
            document = action.model_dump_json()
            if document in action_documents:
                raise ValueError("duplicate corporate actions are not allowed")
            action_documents.add(document)
            if action.source_id is not None:
                source_identity = (action.security_id, action.source_id)
                if source_identity in source_ids:
                    raise ValueError("corporate-action source IDs must be unique per security")
                source_ids.add(source_identity)
            if action.action_type == CorporateActionType.SYMBOL_CHANGE:
                _validate_symbol_change_action(action, periods_by_security[action.security_id])
        return self

    def get_security(self, security_id: str) -> SecurityRecord:
        for record in self.securities:
            if record.security_id == security_id:
                return record
        raise KeyError(security_id)

    def symbol_for(self, security_id: str, as_of: date) -> str:
        record = self.get_security(security_id)
        if not record.is_listed_on(as_of):
            raise KeyError(f"security {security_id} is not listed on {as_of}")
        matches = [
            period.symbol
            for period in self.symbols
            if period.security_id == security_id and period.contains(as_of)
        ]
        if len(matches) != 1:
            raise KeyError(f"no unique symbol for {security_id} on {as_of}")
        return matches[0]

    def security_for_symbol(self, symbol: str, as_of: date) -> SecurityRecord:
        matches = [
            period.security_id
            for period in self.symbols
            if period.symbol == symbol and period.contains(as_of)
        ]
        if len(matches) != 1:
            raise KeyError(f"no unique security for {symbol} on {as_of}")
        record = self.get_security(matches[0])
        if not record.is_listed_on(as_of):
            raise KeyError(f"security {record.security_id} is not listed on {as_of}")
        return record

    def symbol_history(self, security_id: str) -> tuple[SymbolPeriod, ...]:
        self.get_security(security_id)
        periods = [period for period in self.symbols if period.security_id == security_id]
        return tuple(sorted(periods, key=lambda period: period.start_date))

    def active_common_equities(self, as_of: date) -> tuple[SecurityRecord, ...]:
        return tuple(
            record
            for record in self.securities
            if record.security_type == SecurityType.COMMON_STOCK and record.is_listed_on(as_of)
        )

    def actions_for(
        self,
        security_id: str,
        *,
        through: date | None = None,
    ) -> tuple[CorporateAction, ...]:
        self.get_security(security_id)
        actions = [
            action
            for action in self.corporate_actions
            if action.security_id == security_id
            and (through is None or action.effective_date <= through)
        ]
        return tuple(sorted(actions, key=lambda action: action.effective_date))


def _validate_symbol_lifetime(record: SecurityRecord, periods: list[SymbolPeriod]) -> None:
    ordered = sorted(periods, key=lambda period: period.start_date)
    if ordered[0].start_date != record.listing_date:
        raise ValueError("symbol history must begin on the security listing_date")
    last = ordered[-1]
    if record.delisting_date is None:
        if last.end_date is not None:
            raise ValueError("active security symbol history must remain open-ended")
    elif last.end_date != record.delisting_date:
        raise ValueError("delisted security symbol history must end on delisting_date")


def _validate_symbol_change_action(
    action: CorporateAction,
    periods: list[SymbolPeriod],
) -> None:
    assert action.old_symbol is not None and action.new_symbol is not None
    new_periods = [
        period
        for period in periods
        if period.symbol == action.new_symbol and period.start_date == action.effective_date
    ]
    if len(new_periods) != 1:
        raise ValueError("symbol change action new_symbol/effective_date must match symbol history")

    prior_periods = [
        period
        for period in periods
        if period.end_date is not None and period.end_date < action.effective_date
    ]
    if not prior_periods:
        raise ValueError("symbol change action requires prior symbol history")
    prior = max(prior_periods, key=lambda period: period.end_date or period.start_date)
    if prior.symbol != action.old_symbol:
        raise ValueError(
            "symbol change action old_symbol must match immediately prior symbol history"
        )


def _assert_non_overlapping(periods: list[SymbolPeriod], message: str) -> None:
    ordered = sorted(periods, key=lambda period: period.start_date)
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if previous.end_date is None or current.start_date <= previous.end_date:
            raise ValueError(message)
