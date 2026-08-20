"""Tests for point-in-time liquid-universe construction."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from trading_bot.data.security_master import (
    SecurityMaster,
    SecurityRecord,
    SecurityType,
    SymbolPeriod,
)
from trading_bot.data.universe import (
    LiquidityObservation,
    UniverseConstructionError,
    UniversePolicy,
    build_universe_snapshot,
    build_universe_snapshots,
)


def security_master() -> SecurityMaster:
    records = (
        SecurityRecord(
            security_id="a",
            security_type=SecurityType.COMMON_STOCK,
            exchange="NYSE",
            listing_date=date(2010, 1, 1),
        ),
        SecurityRecord(
            security_id="b",
            security_type=SecurityType.COMMON_STOCK,
            exchange="NASDAQ",
            listing_date=date(2010, 1, 1),
        ),
        SecurityRecord(
            security_id="gone",
            security_type=SecurityType.COMMON_STOCK,
            exchange="NYSE",
            listing_date=date(2010, 1, 1),
            delisting_date=date(2020, 12, 31),
        ),
        SecurityRecord(
            security_id="etf",
            security_type=SecurityType.ETF,
            exchange="NYSE",
            listing_date=date(2010, 1, 1),
        ),
        SecurityRecord(
            security_id="future",
            security_type=SecurityType.COMMON_STOCK,
            exchange="NYSE",
            listing_date=date(2025, 1, 1),
        ),
    )
    symbols = tuple(
        SymbolPeriod(
            security_id=record.security_id,
            symbol=record.security_id.upper(),
            start_date=record.listing_date,
            end_date=record.delisting_date,
        )
        for record in records
    )
    return SecurityMaster(version="sm-v1", securities=records, symbols=symbols)


def policy(target_size: int = 3) -> UniversePolicy:
    return UniversePolicy(
        version="liq-v1",
        target_size=target_size,
        trailing_observations=3,
        minimum_history_observations=2,
        minimum_price=5,
        minimum_average_dollar_volume=100,
    )


def obs(security_id: str, when: date, price: float, volume: float) -> LiquidityObservation:
    return LiquidityObservation(security_id, when, price, volume)


def history(as_of: date) -> list[LiquidityObservation]:
    rows: list[LiquidityObservation] = []
    for offset in (3, 2, 1):
        when = as_of - timedelta(days=offset)
        rows.extend(
            [
                obs("a", when, 10, 100),
                obs("b", when, 20, 100),
                obs("gone", when, 15, 100),
                obs("etf", when, 100, 1000),
            ]
        )
    return rows


def test_universe_ranks_by_trailing_average_dollar_volume() -> None:
    as_of = date(2020, 6, 1)
    snapshot = build_universe_snapshot(
        security_master(), history(as_of), as_of=as_of, policy=policy()
    )
    assert snapshot.security_ids == ("b", "gone", "a")
    assert [member.rank for member in snapshot.members] == [1, 2, 3]


def test_future_observations_do_not_change_earlier_snapshot() -> None:
    as_of = date(2020, 6, 1)
    rows = history(as_of)
    baseline = build_universe_snapshot(security_master(), rows, as_of=as_of, policy=policy())
    rows.append(obs("a", as_of + timedelta(days=1), 1000, 1_000_000))
    extended = build_universe_snapshot(security_master(), rows, as_of=as_of, policy=policy())
    assert extended == baseline


def test_observation_on_rebalance_date_is_not_used() -> None:
    as_of = date(2020, 6, 1)
    rows = [*history(as_of), obs("a", as_of, 1000, 1000000)]
    snapshot = build_universe_snapshot(security_master(), rows, as_of=as_of, policy=policy())
    assert snapshot.security_ids[0] == "b"


def test_historical_delisted_security_is_eligible_before_delisting() -> None:
    before = date(2020, 6, 1)
    after = date(2021, 6, 1)
    before_snapshot = build_universe_snapshot(
        security_master(), history(before), as_of=before, policy=policy()
    )
    after_snapshot = build_universe_snapshot(
        security_master(), history(after), as_of=after, policy=policy()
    )
    assert "gone" in before_snapshot.security_ids
    assert "gone" not in after_snapshot.security_ids


def test_non_common_and_future_listings_are_excluded() -> None:
    as_of = date(2020, 6, 1)
    rows = [
        *history(as_of),
        obs("future", as_of - timedelta(days=1), 100, 10000),
        obs("future", as_of - timedelta(days=2), 100, 10000),
    ]
    snapshot = build_universe_snapshot(security_master(), rows, as_of=as_of, policy=policy(10))
    assert "etf" not in snapshot.security_ids
    assert "future" not in snapshot.security_ids


def test_minimum_price_history_and_liquidity_filters_are_enforced() -> None:
    as_of = date(2020, 6, 1)
    rows = [
        obs("a", as_of - timedelta(days=2), 4, 1000),
        obs("a", as_of - timedelta(days=1), 4, 1000),
        obs("b", as_of - timedelta(days=1), 20, 100),
    ]
    snapshot = build_universe_snapshot(security_master(), rows, as_of=as_of, policy=policy(10))
    assert snapshot.members == ()


def test_policy_hash_is_stable_and_snapshots_record_versions() -> None:
    as_of = date(2020, 6, 1)
    first = policy()
    second = policy()
    assert first.policy_sha256() == second.policy_sha256()
    snapshot = build_universe_snapshot(security_master(), history(as_of), as_of=as_of, policy=first)
    assert snapshot.security_master_version == "sm-v1"
    assert snapshot.policy_version == "liq-v1"
    assert snapshot.policy_sha256 == first.policy_sha256()


def test_explicit_rebalance_dates_freeze_multiple_snapshots() -> None:
    first_date = date(2020, 6, 1)
    second_date = date(2020, 6, 8)
    rows = history(first_date) + history(second_date)
    snapshots = build_universe_snapshots(
        security_master(),
        rows,
        rebalance_dates=[second_date, first_date],
        policy=policy(),
    )
    assert [snapshot.as_of for snapshot in snapshots] == [first_date, second_date]


def test_duplicate_or_empty_rebalance_dates_are_rejected() -> None:
    with pytest.raises(UniverseConstructionError, match="at least one"):
        build_universe_snapshots(security_master(), [], rebalance_dates=[], policy=policy())
    target = date(2020, 6, 1)
    with pytest.raises(UniverseConstructionError, match="unique"):
        build_universe_snapshots(
            security_master(), [], rebalance_dates=[target, target], policy=policy()
        )
