"""Causal corporate-action canonicalization that preserves raw observations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Iterable

from trading_bot.data.raw_validation import RawBar
from trading_bot.data.security_master import CorporateActionType, SecurityMaster


class CanonicalizationError(RuntimeError):
    """Raised when a raw bar cannot be mapped to a valid security/action state."""


@dataclass(frozen=True, slots=True)
class CanonicalBar:
    """Raw and causally adjusted views of the same timestamped observation."""

    security_id: str
    symbol: str
    timestamp: datetime
    raw_open: float
    raw_high: float
    raw_low: float
    raw_close: float
    raw_volume: float
    raw_vwap: float | None
    adjusted_open: float
    adjusted_high: float
    adjusted_low: float
    adjusted_close: float
    adjusted_volume: float
    adjusted_vwap: float | None
    cumulative_split_factor: float
    cash_dividend_per_share: float


def canonicalize_bars(
    bars: Iterable[RawBar],
    security_master: SecurityMaster,
) -> tuple[CanonicalBar, ...]:
    """Map raw bars to permanent IDs and causal split-adjusted values.

    Split factors use only actions whose effective date is on or before the bar's date. This is a
    forward/causal normalization: future split information never changes historical observations.
    Raw values are retained alongside the derived values.
    """
    output: list[CanonicalBar] = []
    for bar in bars:
        if bar.timestamp.tzinfo is None or bar.timestamp.utcoffset() is None:
            raise CanonicalizationError("raw bar timestamp must be timezone-aware")
        timestamp = bar.timestamp.astimezone(UTC)
        try:
            security = security_master.get_security(bar.asset_id)
            if not security.is_listed_on(timestamp.date()):
                raise CanonicalizationError(
                    f"security {bar.asset_id} is not listed on {timestamp.date()}"
                )
            symbol = security_master.symbol_for(bar.asset_id, timestamp.date())
        except KeyError as exc:
            raise CanonicalizationError(
                f"cannot resolve security/symbol for {bar.asset_id} on {timestamp.date()}"
            ) from exc

        split_factor = 1.0
        dividend = 0.0
        for action in security_master.actions_for(bar.asset_id, through=timestamp.date()):
            if action.action_type == CorporateActionType.SPLIT:
                assert action.split_ratio is not None
                split_factor *= action.split_ratio
            elif (
                action.action_type == CorporateActionType.CASH_DIVIDEND
                and action.effective_date == timestamp.date()
            ):
                assert action.cash_amount is not None
                dividend += action.cash_amount

        output.append(
            CanonicalBar(
                security_id=bar.asset_id,
                symbol=symbol,
                timestamp=timestamp,
                raw_open=bar.open,
                raw_high=bar.high,
                raw_low=bar.low,
                raw_close=bar.close,
                raw_volume=bar.volume,
                raw_vwap=bar.vwap,
                adjusted_open=bar.open * split_factor,
                adjusted_high=bar.high * split_factor,
                adjusted_low=bar.low * split_factor,
                adjusted_close=bar.close * split_factor,
                adjusted_volume=bar.volume / split_factor,
                adjusted_vwap=None if bar.vwap is None else bar.vwap * split_factor,
                cumulative_split_factor=split_factor,
                cash_dividend_per_share=dividend,
            )
        )
    return tuple(output)


def total_return_between(previous: CanonicalBar, current: CanonicalBar) -> float:
    """Return split-continuous total return using an explicitly recorded cash dividend.

    The dividend is scaled to the same causal share basis as ``adjusted_close``. No future dividend
    or split is incorporated into either observation.
    """
    if previous.security_id != current.security_id:
        raise CanonicalizationError("return observations must belong to the same security")
    if current.timestamp <= previous.timestamp:
        raise CanonicalizationError("return observations must be strictly chronological")
    dividend_on_adjusted_basis = (
        current.cash_dividend_per_share * current.cumulative_split_factor
    )
    return (
        current.adjusted_close + dividend_on_adjusted_basis
    ) / previous.adjusted_close - 1.0
