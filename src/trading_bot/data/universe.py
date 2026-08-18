"""Point-in-time liquid-universe construction from historical security and liquidity data."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from typing import Iterable

from trading_bot.data.security_master import SecurityMaster


class UniverseConstructionError(RuntimeError):
    """Raised when universe inputs or policy cannot produce a valid snapshot."""


@dataclass(frozen=True, slots=True)
class LiquidityObservation:
    """Historical daily liquidity observation available after its observation date."""

    security_id: str
    observation_date: date
    close_price: float
    volume: float

    def __post_init__(self) -> None:
        if not self.security_id.strip():
            raise ValueError("security_id must not be blank")
        if self.close_price <= 0:
            raise ValueError("close_price must be positive")
        if self.volume < 0:
            raise ValueError("volume must be non-negative")

    @property
    def dollar_volume(self) -> float:
        return self.close_price * self.volume


@dataclass(frozen=True, slots=True)
class UniversePolicy:
    """Explicit, versioned selection inputs rather than hidden research assumptions."""

    version: str
    target_size: int
    trailing_observations: int
    minimum_history_observations: int
    minimum_price: float = 0.0
    minimum_average_dollar_volume: float = 0.0

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("policy version must not be blank")
        if self.target_size <= 0:
            raise ValueError("target_size must be positive")
        if self.trailing_observations <= 0:
            raise ValueError("trailing_observations must be positive")
        if not 0 < self.minimum_history_observations <= self.trailing_observations:
            raise ValueError(
                "minimum_history_observations must be between 1 and trailing_observations"
            )
        if self.minimum_price < 0 or self.minimum_average_dollar_volume < 0:
            raise ValueError("minimum price/liquidity thresholds must be non-negative")

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def policy_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class UniverseMember:
    security_id: str
    symbol: str
    average_dollar_volume: float
    latest_price: float
    history_observations: int
    rank: int


@dataclass(frozen=True, slots=True)
class UniverseSnapshot:
    as_of: date
    security_master_version: str
    policy_version: str
    policy_sha256: str
    members: tuple[UniverseMember, ...]

    @property
    def security_ids(self) -> tuple[str, ...]:
        return tuple(member.security_id for member in self.members)


def build_universe_snapshot(
    security_master: SecurityMaster,
    observations: Iterable[LiquidityObservation],
    *,
    as_of: date,
    policy: UniversePolicy,
) -> UniverseSnapshot:
    """Select eligible names using only information strictly before ``as_of``."""
    historical: dict[str, list[LiquidityObservation]] = defaultdict(list)
    for observation in observations:
        if observation.observation_date < as_of:
            historical[observation.security_id].append(observation)

    candidates: list[tuple[str, str, float, float, int]] = []
    for security in security_master.active_common_equities(as_of):
        history = sorted(
            historical.get(security.security_id, ()),
            key=lambda item: item.observation_date,
        )
        if not history:
            continue
        trailing = history[-policy.trailing_observations :]
        if len(trailing) < policy.minimum_history_observations:
            continue
        latest_price = trailing[-1].close_price
        if latest_price < policy.minimum_price:
            continue
        average_dollar_volume = sum(item.dollar_volume for item in trailing) / len(trailing)
        if average_dollar_volume < policy.minimum_average_dollar_volume:
            continue
        try:
            symbol = security_master.symbol_for(security.security_id, as_of)
        except KeyError as exc:
            raise UniverseConstructionError(
                f"eligible security {security.security_id} lacks a point-in-time symbol"
            ) from exc
        candidates.append(
            (
                security.security_id,
                symbol,
                average_dollar_volume,
                latest_price,
                len(trailing),
            )
        )

    candidates.sort(key=lambda item: (-item[2], item[0]))
    selected = candidates[: policy.target_size]
    members = tuple(
        UniverseMember(
            security_id=security_id,
            symbol=symbol,
            average_dollar_volume=average_dollar_volume,
            latest_price=latest_price,
            history_observations=history_count,
            rank=rank,
        )
        for rank, (
            security_id,
            symbol,
            average_dollar_volume,
            latest_price,
            history_count,
        ) in enumerate(selected, start=1)
    )
    return UniverseSnapshot(
        as_of=as_of,
        security_master_version=security_master.version,
        policy_version=policy.version,
        policy_sha256=policy.policy_sha256(),
        members=members,
    )


def build_universe_snapshots(
    security_master: SecurityMaster,
    observations: Iterable[LiquidityObservation],
    *,
    rebalance_dates: Iterable[date],
    policy: UniversePolicy,
) -> tuple[UniverseSnapshot, ...]:
    """Freeze membership on explicit version-controlled rebalance dates."""
    dates = tuple(rebalance_dates)
    if not dates:
        raise UniverseConstructionError("at least one rebalance date is required")
    if len(set(dates)) != len(dates):
        raise UniverseConstructionError("rebalance dates must be unique")
    ordered = tuple(sorted(dates))
    rows = tuple(observations)
    return tuple(
        build_universe_snapshot(security_master, rows, as_of=as_of, policy=policy)
        for as_of in ordered
    )
