"""Synthetic deterministic raw-to-packed integration gate for Phase 3."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np

from trading_bot.data.canonicalization import canonicalize_bars
from trading_bot.data.features import FeatureObservation, FeaturePolicy, compute_features
from trading_bot.data.labels import LabelObservation, LabelPolicy, generate_labels
from trading_bot.data.packing import PackedDataset, TrainingSample, pack_training_data
from trading_bot.data.raw_validation import RawBar, validate_raw_bars
from trading_bot.data.resampling import resample_canonical_bars
from trading_bot.data.security_master import (
    SecurityMaster,
    SecurityRecord,
    SecurityType,
    SymbolPeriod,
)
from trading_bot.data.splits import DateRange, SplitManifest, WalkForwardFold
from trading_bot.data.universe import (
    LiquidityObservation,
    UniversePolicy,
    build_universe_snapshot,
)


START = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)


def _master() -> SecurityMaster:
    records = (
        SecurityRecord(
            security_id="sec-a",
            security_type=SecurityType.COMMON_STOCK,
            exchange="NASDAQ",
            listing_date=date(2020, 1, 1),
            sector="Technology",
        ),
        SecurityRecord(
            security_id="sec-b",
            security_type=SecurityType.COMMON_STOCK,
            exchange="NYSE",
            listing_date=date(2020, 1, 1),
            sector="Industrials",
        ),
    )
    return SecurityMaster(
        version="sm-e2e-v1",
        securities=records,
        symbols=tuple(
            SymbolPeriod(
                security_id=record.security_id,
                symbol=record.security_id.upper(),
                start_date=record.listing_date,
            )
            for record in records
        ),
    )


def _raw_rows() -> tuple[RawBar, ...]:
    rows: list[RawBar] = []
    for security_id, base in (("sec-a", 100.0), ("sec-b", 200.0)):
        for minute in range(15):
            price = base + minute * (0.1 if security_id == "sec-a" else -0.05)
            rows.append(
                RawBar(
                    asset_id=security_id,
                    timestamp=START + timedelta(minutes=minute),
                    open=price,
                    high=price + 0.2,
                    low=price - 0.2,
                    close=price + 0.05,
                    volume=1000.0 + minute * 10,
                    vwap=price + 0.02,
                )
            )
    return tuple(rows)


def _split_manifest() -> SplitManifest:
    return SplitManifest(
        split_version="split-e2e-v1",
        dataset_version="dataset-e2e-v1",
        folds=(
            WalkForwardFold(
                fold_id="fold-1",
                train=DateRange(start=date(2020, 1, 1), end=date(2023, 12, 31)),
                validation=DateRange(start=date(2024, 1, 1), end=date(2024, 12, 31)),
            ),
        ),
        final_holdout_id="final-e2e-v1",
        final_holdout=DateRange(start=date(2025, 1, 1), end=date(2025, 12, 31)),
    )


def _build_samples() -> tuple[TrainingSample, ...]:
    master = _master()
    raw = _raw_rows()
    report = validate_raw_bars(
        raw,
        expected_sessions=(date(2024, 1, 2),),
        expected_assets=("sec-a", "sec-b"),
    )
    assert report.is_valid

    canonical = canonicalize_bars(raw, master)
    resampled = resample_canonical_bars(canonical, 5)
    assert len(resampled) == 6

    universe = build_universe_snapshot(
        master,
        (
            LiquidityObservation("sec-a", date(2024, 1, 1), 100.0, 1_000_000.0),
            LiquidityObservation("sec-b", date(2024, 1, 1), 200.0, 800_000.0),
        ),
        as_of=date(2024, 1, 2),
        policy=UniversePolicy(
            version="universe-e2e-v1",
            target_size=2,
            trailing_observations=1,
            minimum_history_observations=1,
        ),
    )
    assert set(universe.security_ids) == {"sec-a", "sec-b"}

    feature_inputs = tuple(
        FeatureObservation(
            security_id=row.security_id,
            symbol=row.symbol,
            sector=master.get_security(row.security_id).sector or "UNKNOWN",
            timestamp=row.timestamp,
            open=row.adjusted_open,
            high=row.adjusted_high,
            low=row.adjusted_low,
            close=row.adjusted_close,
            volume=row.adjusted_volume,
            vwap=row.adjusted_vwap,
        )
        for row in resampled
        if row.security_id in universe.security_ids
    )
    features = compute_features(
        feature_inputs,
        policy=FeaturePolicy(
            return_horizons=(1, 2),
            relative_volume_window=2,
            volatility_window=2,
            range_window=2,
            momentum_window=2,
        ),
    )

    labels = generate_labels(
        (
            LabelObservation(row.security_id, row.timestamp, row.adjusted_close)
            for row in resampled
            if row.security_id in universe.security_ids
        ),
        policy=LabelPolicy(horizons_minutes=(5,)),
    )
    label_by_identity = {(row.security_id, row.timestamp): row for row in labels}
    split = _split_manifest()

    samples: list[TrainingSample] = []
    for row in features:
        target = label_by_identity[(row.security_id, row.timestamp)].future_returns.get(5)
        if target is None:
            continue
        assert split.routine_partition(row.timestamp.date()) == ("fold-1", "validation")
        samples.append(
            TrainingSample(
                security_id=row.security_id,
                timestamp=row.timestamp,
                features=(
                    row.values["close"],
                    row.values["return_1"],
                    row.values["relative_volume"],
                    row.values["market_relative_return_1"],
                ),
                targets=(target,),
            )
        )
    return tuple(samples)


def test_raw_to_packed_flow_is_deterministic_across_repeated_builds(tmp_path: Path) -> None:
    samples = _build_samples()
    assert len(samples) == 4
    kwargs = {
        "feature_names": ("close", "return_1", "relative_volume", "market_relative_return_1"),
        "target_names": ("future_return_5",),
        "dataset_version": "dataset-e2e-v1",
        "split_version": "split-e2e-v1",
    }
    first = pack_training_data(samples, tmp_path / "first", **kwargs)
    second = pack_training_data(reversed(samples), tmp_path / "second", **kwargs)
    assert first.dataset_sha256 == second.dataset_sha256

    first_data = PackedDataset(first.path)
    second_data = PackedDataset(second.path)
    np.testing.assert_array_equal(first_data.features, second_data.features)
    np.testing.assert_array_equal(first_data.targets, second_data.targets)
    np.testing.assert_array_equal(first_data.timestamps_ns, second_data.timestamps_ns)
    np.testing.assert_array_equal(first_data.asset_ids, second_data.asset_ids)
