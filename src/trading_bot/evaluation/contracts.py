"""CPU-only evaluation data contracts independent of training implementations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

Side = Literal["buy", "sell"]
OrderType = Literal["market", "limit"]


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class EvaluationAssumptions:
    """Frozen economic assumptions consumed by the canonical evaluator."""

    annualization_days: int = 252
    risk_free_rate_annual: float = 0.0
    fee_bps: float = 0.0
    spread_bps: float = 0.0
    slippage_bps: float = 0.0
    impact_bps: float = 0.0
    cost_stress_multipliers: tuple[float, ...] = (1.0, 1.25, 1.5, 2.0)
    latency_stress_seconds: tuple[float, ...] = (0.0, 0.25, 1.0, 5.0, 15.0, 30.0)
    minimum_positive_fold_fraction: float = 0.70

    def __post_init__(self) -> None:
        if self.annualization_days < 1:
            raise ValueError("annualization_days must be positive")
        for name in ("fee_bps", "spread_bps", "slippage_bps", "impact_bps"):
            value = getattr(self, name)
            _require_finite(name, value)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        _require_finite("risk_free_rate_annual", self.risk_free_rate_annual)
        if not 0 <= self.minimum_positive_fold_fraction <= 1:
            raise ValueError("minimum_positive_fold_fraction must be in [0, 1]")
        if not self.cost_stress_multipliers or 1.0 not in self.cost_stress_multipliers:
            raise ValueError("cost stress multipliers must include 1.0")
        if any(value <= 0 or not math.isfinite(value) for value in self.cost_stress_multipliers):
            raise ValueError("cost stress multipliers must be finite and positive")
        if len(set(self.cost_stress_multipliers)) != len(self.cost_stress_multipliers):
            raise ValueError("cost stress multipliers must be unique")
        if not self.latency_stress_seconds or 0.0 not in self.latency_stress_seconds:
            raise ValueError("latency stress grid must include 0.0")
        if any(value < 0 or not math.isfinite(value) for value in self.latency_stress_seconds):
            raise ValueError("latency stress values must be finite and non-negative")
        if len(set(self.latency_stress_seconds)) != len(self.latency_stress_seconds):
            raise ValueError("latency stress values must be unique")


@dataclass(frozen=True, slots=True)
class PredictionPoint:
    """One prediction/target pair with optional robustness dimensions."""

    asset_id: str
    timestamp_ns: int
    target: float
    score: float
    fold_id: str = "unlabeled"
    regime: str = "unlabeled"
    horizon: str = "unlabeled"
    sector: str = "unlabeled"
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.asset_id.strip():
            raise ValueError("asset_id must not be blank")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        _require_finite("target", self.target)
        _require_finite("score", self.score)
        for name in ("fold_id", "regime", "horizon", "sector"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be blank")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True, slots=True)
class PositionPoint:
    """One causal portfolio weight paired with the subsequent realized return."""

    asset_id: str
    timestamp_ns: int
    weight: float
    realized_return: float
    fee_bps: float | None = None
    spread_bps: float | None = None
    slippage_bps: float | None = None
    impact_bps: float | None = None
    market_volume_notional: float | None = None
    average_daily_dollar_volume: float | None = None

    def __post_init__(self) -> None:
        if not self.asset_id.strip():
            raise ValueError("asset_id must not be blank")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        for name in ("weight", "realized_return"):
            _require_finite(name, getattr(self, name))
        if self.realized_return <= -1.0:
            raise ValueError("realized_return must be greater than -100%")
        for name in ("fee_bps", "spread_bps", "slippage_bps", "impact_bps"):
            value = getattr(self, name)
            if value is not None:
                _require_finite(name, value)
                if value < 0:
                    raise ValueError(f"{name} must be non-negative")
        for name in ("market_volume_notional", "average_daily_dollar_volume"):
            value = getattr(self, name)
            if value is not None:
                _require_finite(name, value)
                if value <= 0:
                    raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class LatencyReturn:
    """Alternative realized return for one decision under an artificial execution delay."""

    asset_id: str
    timestamp_ns: int
    delay_seconds: float
    realized_return: float

    def __post_init__(self) -> None:
        if not self.asset_id.strip():
            raise ValueError("asset_id must not be blank")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        _require_finite("delay_seconds", self.delay_seconds)
        _require_finite("realized_return", self.realized_return)
        if self.delay_seconds < 0:
            raise ValueError("delay_seconds must be non-negative")
        if self.realized_return <= -1.0:
            raise ValueError("realized_return must be greater than -100%")


@dataclass(frozen=True, slots=True)
class BacktestStep:
    timestamp_ns: int
    gross_return: float
    fee_cost: float
    spread_cost: float
    slippage_cost: float
    impact_cost: float
    total_cost: float
    net_return: float
    gross_traded_weight: float
    one_way_turnover: float
    trade_count: int

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        for name in (
            "gross_return",
            "fee_cost",
            "spread_cost",
            "slippage_cost",
            "impact_cost",
            "total_cost",
            "net_return",
            "gross_traded_weight",
            "one_way_turnover",
        ):
            _require_finite(name, getattr(self, name))
        if (
            min(
                self.fee_cost,
                self.spread_cost,
                self.slippage_cost,
                self.impact_cost,
                self.total_cost,
                self.gross_traded_weight,
                self.one_way_turnover,
            )
            < 0
        ):
            raise ValueError("costs and turnover must be non-negative")
        if self.trade_count < 0:
            raise ValueError("trade_count must be non-negative")


@dataclass(frozen=True, slots=True)
class DailyReturn:
    date: str
    gross_return: float
    total_cost: float
    net_return: float
    nav: float

    def __post_init__(self) -> None:
        if not self.date:
            raise ValueError("date must not be blank")
        for name in ("gross_return", "total_cost", "net_return", "nav"):
            _require_finite(name, getattr(self, name))
        if self.nav <= 0:
            raise ValueError("nav must remain positive")


@dataclass(frozen=True, slots=True)
class Quote:
    """Top-of-book snapshot used by the reference execution simulator."""

    timestamp_ns: int
    bid: float
    ask: float
    bid_size: float
    ask_size: float

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        for name in ("bid", "ask", "bid_size", "ask_size"):
            _require_finite(name, getattr(self, name))
        if self.bid <= 0 or self.ask <= 0 or self.bid_size < 0 or self.ask_size < 0:
            raise ValueError("quote prices must be positive and sizes non-negative")
        if self.bid > self.ask:
            raise ValueError("crossed quote is invalid")


@dataclass(frozen=True, slots=True)
class Order:
    asset_id: str
    decision_timestamp_ns: int
    side: Side
    quantity: float
    order_type: OrderType = "market"
    limit_price: float | None = None

    def __post_init__(self) -> None:
        if not self.asset_id.strip():
            raise ValueError("asset_id must not be blank")
        if self.decision_timestamp_ns < 0:
            raise ValueError("decision_timestamp_ns must be non-negative")
        _require_finite("quantity", self.quantity)
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.order_type == "limit":
            if self.limit_price is None:
                raise ValueError("limit orders require limit_price")
            _require_finite("limit_price", self.limit_price)
            if self.limit_price <= 0:
                raise ValueError("limit_price must be positive")
        elif self.limit_price is not None:
            raise ValueError("market orders must not define limit_price")


@dataclass(frozen=True, slots=True)
class Fill:
    asset_id: str
    side: Side
    decision_timestamp_ns: int
    first_fill_timestamp_ns: int
    last_fill_timestamp_ns: int
    requested_quantity: float
    filled_quantity: float
    average_price: float

    @property
    def fill_fraction(self) -> float:
        return self.filled_quantity / self.requested_quantity


@dataclass(frozen=True, slots=True)
class ValidityEvidence:
    """Hard-gate evidence supplied by upstream audit and evaluator checks."""

    data_leakage_free: bool = True
    cost_accounting_complete: bool = True
    final_holdout_clean: bool = True
    evaluation_data_complete: bool = True
    exposure_valid: bool = True
    coverage_fraction: float = 1.0
    minimum_coverage_fraction: float = 1.0

    def __post_init__(self) -> None:
        for name in ("coverage_fraction", "minimum_coverage_fraction"):
            value = getattr(self, name)
            _require_finite(name, value)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class FactorObservation:
    date: str
    strategy_excess_return: float
    factors: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.date:
            raise ValueError("date must not be blank")
        _require_finite("strategy_excess_return", self.strategy_excess_return)
        if not self.factors:
            raise ValueError("at least one factor is required")
        for name, value in self.factors.items():
            if not name.strip():
                raise ValueError("factor name must not be blank")
            _require_finite(f"factor {name}", value)
