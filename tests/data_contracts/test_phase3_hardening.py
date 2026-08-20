"""Additional hardening checks for the reference Phase 3 data pipeline."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from trading_bot.data.acquisition import (
    AcquisitionError,
    ProviderBatch,
    ProviderBatchRef,
    acquire_provider_batch,
    raw_batch_object_key,
)
from trading_bot.data.canonicalization import (
    CanonicalizationError,
    canonicalize_bars,
)
from trading_bot.data.features import (
    FeatureGenerationError,
    FeaturePolicy,
    generate_features,
)
from trading_bot.data.labels import LabelGenerationError, LabelObservation, generate_labels
from trading_bot.data.packing import PackedDataset, PackingError, TrainingSample, pack_training_data
from trading_bot.data.raw_validation import RawValidationError, validate_raw_bars
from trading_bot.data.resampling import ResamplingError, SessionSpec, resample_canonical_bars
from trading_bot.data.security_master import (
    CorporateAction,
    CorporateActionType,
    SecurityMaster,
    SecurityMasterError,
    SecurityRecord,
    SecurityType,
    SymbolPeriod,
)
from trading_bot.data.splits import SplitError, SplitPolicy, build_splits
from trading_bot.data.universe import (
    LiquidityObservation,
    UniverseConstructionError,
    UniversePolicy,
    build_universe_snapshot,
)
from trading_bot.data.vendor import RawBar, RawBarBatch
from trading_bot.storage.local import LocalStorageBackend


def _security_master() -> SecurityMaster:
    return SecurityMaster(
        version="v1",
        securities=(
            SecurityRecord(
                security_id="a",
                security_type=SecurityType.COMMON_STOCK,
                exchange="NYSE",
                listing_date=date(2020, 1, 1),
            ),
        ),
        symbols=(
            SymbolPeriod(
                security_id="a",
                symbol="AAA",
                start_date=date(2020, 1, 1),
            ),
        ),
    )


def _raw_bar(timestamp: datetime, *, close: float = 10.0) -> RawBar:
    return RawBar(
        asset_id="a",
        timestamp=timestamp,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=100.0,
        vwap=close,
    )


def test_acquisition_rejects_provider_batch_identity_mismatch(tmp_path: Path) -> None:
    class Adapter:
        name = "fake"

        def fetch_bars(self, batch: ProviderBatchRef) -> ProviderBatch:
            return ProviderBatch(
                batch=replace(batch, batch_id="wrong"),
                rows=(_raw_bar(batch.start),),
            )

    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    batch = ProviderBatchRef(
        provider="fake",
        batch_id="b1",
        start=start,
        end=start + timedelta(minutes=1),
        asset_ids=("a",),
        interval="1m",
    )
    with pytest.raises(AcquisitionError, match="identity mismatch"):
        acquire_provider_batch(Adapter(), batch, tmp_path)


def test_acquisition_object_key_is_stable_and_scoped() -> None:
    batch = ProviderBatchRef(
        provider="fake",
        batch_id="b1",
        start=datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
        end=datetime(2024, 1, 2, 14, 31, tzinfo=UTC),
        asset_ids=("a",),
        interval="1m",
    )
    assert raw_batch_object_key(batch) == "raw/fake/b1.jsonl"


def test_raw_validation_rejects_non_finite_and_bad_ohlc() -> None:
    timestamp = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    with pytest.raises(ValueError, match="finite"):
        RawBar(
            asset_id="a",
            timestamp=timestamp,
            open=10,
            high=10,
            low=10,
            close=math.inf,
            volume=100,
        )

    bad = RawBar(
        asset_id="a",
        timestamp=timestamp,
        open=10,
        high=9,
        low=8,
        close=8.5,
        volume=100,
    )
    with pytest.raises(RawValidationError, match="high is below"):
        validate_raw_bars([bad], batch_start=timestamp, batch_end=timestamp)


def test_raw_validation_rejects_rows_outside_batch_window() -> None:
    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    with pytest.raises(RawValidationError, match="outside declared batch window"):
        validate_raw_bars(
            [_raw_bar(start + timedelta(minutes=2))],
            batch_start=start,
            batch_end=start + timedelta(minutes=1),
        )


def test_security_master_rejects_action_before_listing() -> None:
    with pytest.raises(SecurityMasterError, match="before listing"):
        SecurityMaster(
            version="v1",
            securities=(
                SecurityRecord(
                    security_id="a",
                    security_type=SecurityType.COMMON_STOCK,
                    exchange="NYSE",
                    listing_date=date(2020, 1, 1),
                ),
            ),
            symbols=(
                SymbolPeriod(
                    security_id="a",
                    symbol="AAA",
                    start_date=date(2020, 1, 1),
                ),
            ),
            corporate_actions=(
                CorporateAction(
                    security_id="a",
                    action_type=CorporateActionType.SPLIT,
                    effective_date=date(2019, 12, 31),
                    split_ratio=2.0,
                ),
            ),
        )


def test_security_master_rejects_non_finite_action_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        CorporateAction(
            security_id="a",
            action_type=CorporateActionType.SPLIT,
            effective_date=date(2020, 1, 2),
            split_ratio=math.inf,
        )


def test_canonicalization_rejects_overflowed_adjusted_values() -> None:
    master = SecurityMaster(
        version="v1",
        securities=(
            SecurityRecord(
                security_id="a",
                security_type=SecurityType.COMMON_STOCK,
                exchange="NYSE",
                listing_date=date(2020, 1, 1),
            ),
        ),
        symbols=(
            SymbolPeriod(
                security_id="a",
                symbol="AAA",
                start_date=date(2020, 1, 1),
            ),
        ),
        corporate_actions=tuple(
            CorporateAction(
                security_id="a",
                action_type=CorporateActionType.SPLIT,
                effective_date=date(2020, 1, 2 + offset),
                split_ratio=1e100,
            )
            for offset in range(3)
        ),
    )
    with pytest.raises(CanonicalizationError, match="finite"):
        canonicalize_bars(
            [_raw_bar(datetime(2020, 1, 4, 14, 30, tzinfo=UTC))],
            master,
        )


def test_resampling_rejects_cross_session_and_off_grid_rows() -> None:
    session = SessionSpec(
        session_date=date(2024, 1, 2),
        open_time=datetime(2024, 1, 2, 9, 30, tzinfo=UTC),
        close_time=datetime(2024, 1, 2, 9, 35, tzinfo=UTC),
    )
    master = _security_master()
    cross_session = canonicalize_bars(
        [_raw_bar(datetime(2024, 1, 3, 9, 30, tzinfo=UTC))],
        master,
    )
    with pytest.raises(ResamplingError, match="outside configured session"):
        resample_canonical_bars(cross_session, session=session, intervals_minutes=(5,))

    off_grid = canonicalize_bars(
        [_raw_bar(datetime(2024, 1, 2, 9, 30, 30, tzinfo=UTC))],
        master,
    )
    with pytest.raises(ResamplingError, match="base interval grid"):
        resample_canonical_bars(off_grid, session=session, intervals_minutes=(5,))


def test_universe_rejects_non_finite_liquidity_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        LiquidityObservation("a", date(2024, 1, 2), price=math.inf, volume=100)


def test_universe_rejects_observation_before_listing() -> None:
    policy = UniversePolicy(
        version="u1",
        target_size=10,
        trailing_observations=2,
        minimum_history_observations=1,
        minimum_price=1,
        minimum_average_dollar_volume=1,
    )
    with pytest.raises(UniverseConstructionError, match="outside security lifetime"):
        build_universe_snapshot(
            _security_master(),
            [LiquidityObservation("a", date(2019, 12, 31), 10, 100)],
            as_of=date(2020, 1, 2),
            policy=policy,
        )


def test_feature_generation_rejects_duplicate_bars() -> None:
    master = _security_master()
    timestamp = datetime(2024, 1, 2, 9, 30, tzinfo=UTC)
    bars = canonicalize_bars([_raw_bar(timestamp)], master)
    policy = FeaturePolicy(return_lags=(1,), volatility_windows=(2,), volume_window=2)
    with pytest.raises(FeatureGenerationError, match="duplicate"):
        generate_features([bars[0], bars[0]], policy=policy)


def test_label_generation_rejects_duplicate_observations() -> None:
    timestamp = datetime(2024, 1, 2, 9, 30, tzinfo=UTC)
    row = LabelObservation("a", timestamp, 10.0)
    with pytest.raises(LabelGenerationError, match="duplicate"):
        generate_labels([row, row])


def test_split_policy_rejects_non_monotonic_dates() -> None:
    with pytest.raises(SplitError, match="strictly increasing"):
        SplitPolicy(
            version="s1",
            training_start=date(2020, 1, 1),
            validation_start=date(2020, 6, 1),
            test_start=date(2020, 5, 1),
            final_holdout_start=date(2020, 7, 1),
            end_date=date(2020, 8, 1),
        )


def test_split_builder_rejects_timestamp_outside_declared_window() -> None:
    policy = SplitPolicy(
        version="s1",
        training_start=date(2020, 1, 1),
        validation_start=date(2020, 2, 1),
        test_start=date(2020, 3, 1),
        final_holdout_start=date(2020, 4, 1),
        end_date=date(2020, 5, 1),
    )
    with pytest.raises(SplitError, match="outside split policy window"):
        build_splits(
            [datetime(2019, 12, 31, tzinfo=UTC)],
            policy=policy,
        )


def test_packing_rejects_float32_overflow_and_duplicates(tmp_path: Path) -> None:
    timestamp = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    with pytest.raises(PackingError, match="float32"):
        pack_training_data(
            [TrainingSample("a", timestamp, (float(2**128),), (1.0,))],
            tmp_path / "overflow",
            feature_names=("f",),
            target_names=("t",),
            dataset_version="d1",
            split_version="s1",
        )

    duplicate = TrainingSample("a", timestamp, (1.0,), (2.0,))
    with pytest.raises(PackingError, match="duplicate"):
        pack_training_data(
            [duplicate, duplicate],
            tmp_path / "duplicate",
            feature_names=("f",),
            target_names=("t",),
            dataset_version="d1",
            split_version="s1",
        )


def test_packing_uses_exact_integer_timestamp_conversion_and_shape_checks(tmp_path: Path) -> None:
    timestamp = datetime(2024, 1, 2, 14, 30, 0, 123456, tzinfo=UTC)
    destination = tmp_path / "pack"
    pack_training_data(
        [TrainingSample("a", timestamp, (1.0,), (2.0,))],
        destination,
        feature_names=("f",),
        target_names=("t",),
        dataset_version="d1",
        split_version="s1",
    )
    dataset = PackedDataset(destination)
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    delta = timestamp - epoch
    expected = ((delta.days * 86_400 + delta.seconds) * 1_000_000 + 123456) * 1_000
    assert int(dataset.timestamps_ns[0]) == expected

    metadata_path = destination / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["feature_count"] = 2
    metadata["feature_names"] = ["f", "synthetic-second-feature"]
    metadata_bytes = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    metadata_path.write_bytes(metadata_bytes)
    (destination / "metadata.sha256").write_text(
        f"{hashlib.sha256(metadata_bytes).hexdigest()}\n",
        encoding="ascii",
    )
    with pytest.raises(PackingError, match="feature array shape"):
        PackedDataset(destination)


def test_acquisition_persists_raw_batch_to_storage_backend(tmp_path: Path) -> None:
    class Adapter:
        name = "fake"

        def fetch_bars(self, batch: ProviderBatchRef) -> ProviderBatch:
            return ProviderBatch(batch=batch, rows=(_raw_bar(batch.start),))

    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    batch = ProviderBatchRef(
        provider="fake",
        batch_id="b1",
        start=start,
        end=start + timedelta(minutes=1),
        asset_ids=("a",),
        interval="1m",
    )
    storage = LocalStorageBackend(tmp_path / "objects")
    result = acquire_provider_batch(Adapter(), batch, tmp_path / "raw", storage=storage)
    assert result.object_key == "raw/fake/b1.jsonl"
    assert storage.exists(result.object_key)
    assert storage.verify_checksum(result.object_key, result.checksum_sha256)
