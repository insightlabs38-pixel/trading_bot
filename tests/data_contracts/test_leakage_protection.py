"""Cross-module regression tests that make leakage difficult to introduce accidentally."""

from __future__ import annotations

import inspect
from datetime import UTC, date, datetime, timedelta

import pytest
from pydantic import ValidationError

from trading_bot.data import features as feature_module
from trading_bot.data.canonicalization import canonicalize_bars
from trading_bot.data.features import FeatureObservation, FeaturePolicy, compute_features
from trading_bot.data.labels import LabelObservation, LabelPolicy, generate_labels
from trading_bot.data.raw_validation import RawBar
from trading_bot.data.resampling import resample_canonical_bars
from trading_bot.data.security_master import (
    CorporateAction,
    CorporateActionType,
    SecurityMaster,
    SecurityRecord,
    SecurityType,
    SymbolPeriod,
)
from trading_bot.data.splits import (
    DateRange,
    FinalHoldoutAccessError,
    SplitManifest,
    WalkForwardFold,
)
from trading_bot.data.universe import (
    LiquidityObservation,
    UniversePolicy,
    build_universe_snapshot,
)

START = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)


def _security_master(*, future_split: bool = False) -> SecurityMaster:
    actions = ()
    if future_split:
        actions = (
            CorporateAction(
                security_id="sec-a",
                action_type=CorporateActionType.SPLIT,
                effective_date=date(2030, 1, 1),
                split_ratio=2.0,
            ),
        )
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
        corporate_actions=actions,
    )


def _feature(minute: int, close: float) -> FeatureObservation:
    return FeatureObservation(
        security_id="sec-a",
        symbol="AAA",
        sector="Technology",
        timestamp=START + timedelta(minutes=minute),
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=100,
        vwap=close,
    )


def test_future_rows_cannot_change_existing_features() -> None:
    policy = FeaturePolicy(
        return_horizons=(1, 2),
        relative_volume_window=2,
        volatility_window=2,
        range_window=2,
        momentum_window=2,
    )
    prefix = [_feature(0, 100), _feature(1, 101), _feature(2, 102)]
    baseline = compute_features(prefix, policy=policy)
    extended = compute_features([*prefix, _feature(3, 1000)], policy=policy)
    assert extended[: len(baseline)] == baseline


def test_feature_module_does_not_import_label_pipeline() -> None:
    source = inspect.getsource(feature_module)
    assert "trading_bot.data.labels" not in source
    assert "LabelRow" not in source
    assert "generate_labels" not in source


def test_label_targets_are_future_only_and_require_exact_endpoints() -> None:
    rows = [
        LabelObservation("sec-a", START, 100),
        LabelObservation("sec-a", START + timedelta(minutes=6), 106),
    ]
    labels = generate_labels(rows, policy=LabelPolicy(horizons_minutes=(5, 6)))
    first = next(row for row in labels if row.timestamp == START)
    assert 5 not in first.future_returns
    assert first.future_returns[6] == pytest.approx(0.06)


def test_future_liquidity_cannot_change_prior_universe_snapshot() -> None:
    master = _security_master()
    as_of = date(2024, 1, 10)
    history = [
        LiquidityObservation("sec-a", as_of - timedelta(days=2), 10, 100),
        LiquidityObservation("sec-a", as_of - timedelta(days=1), 10, 100),
    ]
    policy = UniversePolicy(
        version="u-v1",
        target_size=1,
        trailing_observations=2,
        minimum_history_observations=2,
    )
    baseline = build_universe_snapshot(master, history, as_of=as_of, policy=policy)
    changed = build_universe_snapshot(
        master,
        [*history, LiquidityObservation("sec-a", as_of + timedelta(days=1), 1000, 1000000)],
        as_of=as_of,
        policy=policy,
    )
    assert changed == baseline


def test_future_corporate_action_cannot_change_historical_canonical_bar() -> None:
    raw = RawBar(
        asset_id="sec-a",
        timestamp=START,
        open=100,
        high=101,
        low=99,
        close=100,
        volume=100,
        vwap=100,
    )
    baseline = canonicalize_bars([raw], _security_master())[0]
    with_future = canonicalize_bars([raw], _security_master(future_split=True))[0]
    assert with_future == baseline


def test_resampling_never_uses_future_bucket_to_change_prior_bucket() -> None:
    rows = []
    for minute in range(10):
        raw = RawBar(
            asset_id="sec-a",
            timestamp=START + timedelta(minutes=minute),
            open=100 + minute,
            high=101 + minute,
            low=99 + minute,
            close=100 + minute,
            volume=100,
            vwap=100 + minute,
        )
        rows.append(canonicalize_bars([raw], _security_master())[0])
    first = resample_canonical_bars(rows[:5], 5)[0]
    extended = resample_canonical_bars(rows, 5)
    assert extended[0] == first


def test_missing_interval_is_not_silently_filled_during_resampling() -> None:
    canonical = []
    for minute in (0, 1, 3, 4):
        raw = RawBar(
            "sec-a",
            START + timedelta(minutes=minute),
            100,
            101,
            99,
            100,
            100,
            100,
        )
        canonical.append(canonicalize_bars([raw], _security_master())[0])
    assert resample_canonical_bars(canonical, 5) == ()


def test_train_validation_chronology_and_final_holdout_are_enforced() -> None:
    manifest = SplitManifest(
        split_version="split-v1",
        dataset_version="dataset-v1",
        folds=(
            WalkForwardFold(
                fold_id="fold-1",
                train=DateRange(start=date(2018, 1, 1), end=date(2019, 12, 31)),
                validation=DateRange(start=date(2020, 1, 1), end=date(2020, 12, 31)),
            ),
        ),
        final_holdout_id="holdout-v1",
        final_holdout=DateRange(start=date(2021, 1, 1), end=date(2021, 12, 31)),
    )
    assert manifest.routine_partition(date(2021, 6, 1)) is None
    with pytest.raises(FinalHoldoutAccessError):
        manifest.final_holdout_range()


def test_ticker_reuse_cannot_splice_unrelated_securities() -> None:
    with pytest.raises(ValidationError, match="same symbol overlaps"):
        SecurityMaster(
            version="sm-v1",
            securities=(
                SecurityRecord(
                    security_id="one",
                    security_type=SecurityType.COMMON_STOCK,
                    exchange="NYSE",
                    listing_date=date(2010, 1, 1),
                ),
                SecurityRecord(
                    security_id="two",
                    security_type=SecurityType.COMMON_STOCK,
                    exchange="NYSE",
                    listing_date=date(2015, 1, 1),
                ),
            ),
            symbols=(
                SymbolPeriod(security_id="one", symbol="SAME", start_date=date(2010, 1, 1)),
                SymbolPeriod(security_id="two", symbol="SAME", start_date=date(2015, 1, 1)),
            ),
        )
