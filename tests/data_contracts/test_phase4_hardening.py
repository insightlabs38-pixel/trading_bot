"""Adversarial regressions for Phase 4 leakage protection and dataset audits."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, time, timedelta

import pytest

from trading_bot.data.audit import (
    AuditObservation,
    DatasetAuditError,
    build_dataset_audit_report,
)
from trading_bot.data.canonicalization import canonicalize_bars
from trading_bot.data.features import FeatureObservation, FeaturePolicy, compute_features
from trading_bot.data.raw_validation import RawBar
from trading_bot.data.resampling import SessionSpec, resample_canonical_bars
from trading_bot.data.security_master import (
    SecurityMaster,
    SecurityRecord,
    SecurityType,
    SymbolPeriod,
)
from trading_bot.data.splits import DateRange, SplitManifest, WalkForwardFold
from trading_bot.data.universe import UniverseMember, UniverseSnapshot

START = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)


def _split_manifest() -> SplitManifest:
    return SplitManifest(
        split_version="split-v1",
        dataset_version="dataset-v1",
        folds=(
            WalkForwardFold(
                fold_id="fold-1",
                train=DateRange(start=date(2020, 1, 1), end=date(2020, 12, 31)),
                validation=DateRange(start=date(2021, 1, 1), end=date(2021, 6, 30)),
            ),
        ),
        final_holdout_id="final-v1",
        final_holdout=DateRange(start=date(2022, 1, 1), end=date(2022, 12, 31)),
    )


def _snapshot(day: int, ids: tuple[str, ...]) -> UniverseSnapshot:
    members = tuple(
        UniverseMember(
            security_id=security_id,
            symbol=security_id.upper(),
            average_dollar_volume=1000.0,
            latest_price=10.0,
            history_observations=20,
            rank=rank,
        )
        for rank, security_id in enumerate(ids, start=1)
    )
    return UniverseSnapshot(
        as_of=date(2024, 1, day),
        security_master_version="sm-v1",
        policy_version="u-v1",
        policy_sha256="a" * 64,
        members=members,
    )


def _single_security_master() -> SecurityMaster:
    return SecurityMaster(
        version="sm-v1",
        securities=(
            SecurityRecord(
                security_id="sec-a",
                security_type=SecurityType.COMMON_STOCK,
                exchange="NASDAQ",
                listing_date=date(2010, 1, 1),
            ),
        ),
        symbols=(
            SymbolPeriod(
                security_id="sec-a",
                symbol="AAA",
                start_date=date(2010, 1, 1),
            ),
        ),
    )


def _raw(day: int, minute: int) -> RawBar:
    timestamp = datetime(2024, 1, day, 14, 30, tzinfo=UTC) + timedelta(minutes=minute)
    return RawBar("sec-a", timestamp, 100, 101, 99, 100, 100, 100)


def _feature(security_id: str, minute: int, close: float) -> FeatureObservation:
    return FeatureObservation(
        security_id=security_id,
        symbol=security_id.upper(),
        sector="Technology",
        timestamp=START + timedelta(minutes=minute),
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=100,
        vwap=close,
    )


def test_routine_split_view_physically_omits_final_holdout_dates() -> None:
    manifest = _split_manifest()
    routine = manifest.routine_view()
    assert not hasattr(routine, "final_holdout")
    assert not hasattr(routine, "final_holdout_range")
    assert routine.full_split_sha256 == manifest.split_sha256()
    assert routine.routine_partition(date(2022, 6, 1)) is None
    payload = routine.canonical_json()
    assert "2022-01-01" not in payload
    assert "2022-12-31" not in payload
    assert "final-v1" in payload


def test_routine_dataset_audit_never_serializes_final_holdout_dates() -> None:
    manifest = _split_manifest()
    report = build_dataset_audit_report([], split_manifest=manifest.routine_view())
    assert report.split_sha256 == manifest.split_sha256()
    assert report.final_holdout_id == "final-v1"
    assert "2022-01-01" not in report.canonical_json()
    assert "2022-12-31" not in report.to_markdown()


def test_future_cross_sectional_rows_cannot_change_prior_feature_panel() -> None:
    policy = FeaturePolicy(
        return_horizons=(1,),
        relative_volume_window=2,
        volatility_window=2,
        range_window=2,
        momentum_window=2,
    )
    prefix = [
        _feature("a", 0, 100),
        _feature("b", 0, 100),
        _feature("a", 1, 101),
        _feature("b", 1, 99),
    ]
    baseline = compute_features(prefix, policy=policy)
    extended = compute_features(
        [*prefix, _feature("c", 2, 1000), _feature("a", 2, 500)],
        policy=policy,
    )
    assert extended[: len(baseline)] == baseline


def test_resampling_keeps_adjacent_sessions_separate() -> None:
    master = _single_security_master()
    session = SessionSpec(open_time=time(9, 30), close_time=time(9, 35))
    first = canonicalize_bars([_raw(2, minute) for minute in range(5)], master)
    second = canonicalize_bars([_raw(3, minute) for minute in range(5)], master)
    daily = resample_canonical_bars(first + second, "1d", session=session)
    assert len(daily) == 2
    assert daily[0].window_end.date() != daily[1].window_end.date()
    assert all(item.source_count == 5 for item in daily)


def test_non_overlapping_ticker_reuse_preserves_permanent_security_identity() -> None:
    master = SecurityMaster(
        version="reuse-v1",
        securities=(
            SecurityRecord(
                security_id="old-company",
                security_type=SecurityType.COMMON_STOCK,
                exchange="NYSE",
                listing_date=date(2010, 1, 1),
                delisting_date=date(2020, 12, 31),
            ),
            SecurityRecord(
                security_id="new-company",
                security_type=SecurityType.COMMON_STOCK,
                exchange="NASDAQ",
                listing_date=date(2021, 1, 1),
            ),
        ),
        symbols=(
            SymbolPeriod(
                security_id="old-company",
                symbol="SAME",
                start_date=date(2010, 1, 1),
                end_date=date(2020, 12, 31),
            ),
            SymbolPeriod(
                security_id="new-company",
                symbol="SAME",
                start_date=date(2021, 1, 1),
            ),
        ),
    )
    assert master.security_for_symbol("SAME", date(2020, 6, 1)).security_id == "old-company"
    assert master.security_for_symbol("SAME", date(2021, 6, 1)).security_id == "new-company"


def test_audit_surfaces_duplicates_bad_closes_and_derived_return_overflow() -> None:
    rows = [
        AuditObservation("overflow", START, 1e-308, 1e308),
        AuditObservation("overflow", START + timedelta(minutes=1), 1e308, 1e308),
        AuditObservation("bad", START, -1.0, 10.0),
        AuditObservation("duplicate", START, 100.0, 10.0),
        AuditObservation("duplicate", START, 100.0, 10.0),
    ]
    report = build_dataset_audit_report(rows)
    assert report.duplicate_observations == 1
    assert report.nonpositive_close == 1
    assert report.nonfinite_return == 1
    assert report.volume_summary.count == 5
    assert math.isfinite(report.volume_summary.mean or math.nan)
    assert math.isfinite(report.volume_summary.std or math.nan)


def test_universe_turnover_is_symmetric_and_bounded_when_size_changes() -> None:
    grows = build_dataset_audit_report(
        [],
        universe_snapshots=[_snapshot(2, ("a",)), _snapshot(9, ("a", "b", "c"))],
    ).universe_turnover[0]
    shrinks = build_dataset_audit_report(
        [],
        universe_snapshots=[_snapshot(2, ("a", "b", "c")), _snapshot(9, ("a",))],
    ).universe_turnover[0]
    assert grows.one_way_turnover == pytest.approx(0.5)
    assert shrinks.one_way_turnover == pytest.approx(0.5)
    assert 0.0 <= grows.one_way_turnover <= 1.0
    assert 0.0 <= shrinks.one_way_turnover <= 1.0


def test_audit_rejects_ambiguous_universe_snapshots() -> None:
    with pytest.raises(DatasetAuditError, match="dates must be unique"):
        build_dataset_audit_report(
            [],
            universe_snapshots=[_snapshot(2, ("a",)), _snapshot(2, ("a",))],
        )

    duplicate_members = UniverseSnapshot(
        as_of=date(2024, 1, 2),
        security_master_version="sm-v1",
        policy_version="u-v1",
        policy_sha256="a" * 64,
        members=(
            UniverseMember("a", "A", 1000, 10, 20, 1),
            UniverseMember("a", "A", 900, 10, 20, 2),
        ),
    )
    with pytest.raises(DatasetAuditError, match="security IDs must be unique"):
        build_dataset_audit_report([], universe_snapshots=[duplicate_members])


def test_audit_rejects_noncontiguous_universe_ranks() -> None:
    malformed = UniverseSnapshot(
        as_of=date(2024, 1, 2),
        security_master_version="sm-v1",
        policy_version="u-v1",
        policy_sha256="a" * 64,
        members=(UniverseMember("a", "A", 1000, 10, 20, 2),),
    )
    with pytest.raises(DatasetAuditError, match="ranks must be contiguous"):
        build_dataset_audit_report([], universe_snapshots=[malformed])
