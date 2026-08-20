"""Adversarial regression tests for Phase 3 data-boundary contracts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from trading_bot.data.acquisition import (
    AcquisitionError,
    AcquisitionRunner,
    VendorPayload,
    VendorRequest,
)
from trading_bot.data.canonicalization import (
    CanonicalBar,
    CanonicalizationError,
    canonicalize_bars,
)
from trading_bot.data.features import (
    FeatureObservation,
    FeaturePipelineError,
    compute_features,
)
from trading_bot.data.labels import (
    LabelGenerationError,
    LabelObservation,
    LabelPolicy,
    generate_labels,
)
from trading_bot.data.packing import PackedDataset, PackingError, TrainingSample, pack_training_data
from trading_bot.data.raw_validation import AnomalyCode, RawBar, validate_raw_bars
from trading_bot.data.resampling import ResamplingError, resample_canonical_bars
from trading_bot.data.security_master import (
    CorporateAction,
    CorporateActionType,
    SecurityMaster,
    SecurityRecord,
    SecurityType,
    SymbolPeriod,
)
from trading_bot.data.splits import DateRange, SplitManifest, WalkForwardFold
from trading_bot.data.universe import (
    LiquidityObservation,
    UniverseConstructionError,
    UniversePolicy,
    build_universe_snapshot,
)
from trading_bot.storage import LocalStorageBackend

START = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)


class MetadataAdapter:
    provider_name = "fake"

    def __init__(self, metadata: dict[str, object] | None = None) -> None:
        self.metadata = metadata or {}

    def fetch(self, request: VendorRequest) -> VendorPayload:
        return VendorPayload(b"raw", response_metadata=self.metadata)  # type: ignore[arg-type]


def _master(*, delisting_date: date | None = None) -> SecurityMaster:
    record = SecurityRecord(
        security_id="sec-1",
        security_type=SecurityType.COMMON_STOCK,
        exchange="NASDAQ",
        listing_date=date(2020, 1, 1),
        delisting_date=delisting_date,
    )
    return SecurityMaster(
        version="sm-v1",
        securities=(record,),
        symbols=(
            SymbolPeriod(
                security_id="sec-1",
                symbol="AAA",
                start_date=record.listing_date,
                end_date=record.delisting_date,
            ),
        ),
    )


def _raw(minute: int = 0) -> RawBar:
    return RawBar(
        asset_id="sec-1",
        timestamp=START + timedelta(minutes=minute),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1000.0,
        vwap=100.2,
    )


def _canonical(security_id: str, minute: int) -> CanonicalBar:
    raw = _raw(minute)
    return CanonicalBar(
        security_id=security_id,
        symbol=security_id.upper(),
        timestamp=raw.timestamp,
        raw_open=raw.open,
        raw_high=raw.high,
        raw_low=raw.low,
        raw_close=raw.close,
        raw_volume=raw.volume,
        raw_vwap=raw.vwap,
        adjusted_open=raw.open,
        adjusted_high=raw.high,
        adjusted_low=raw.low,
        adjusted_close=raw.close,
        adjusted_volume=raw.volume,
        adjusted_vwap=raw.vwap,
        cumulative_split_factor=1.0,
        cash_dividend_per_share=0.0,
    )


def _split_manifest(**overrides: object) -> SplitManifest:
    payload: dict[str, object] = {
        "split_version": "split-v1",
        "dataset_version": "dataset-v1",
        "folds": (
            WalkForwardFold(
                fold_id="fold-1",
                train=DateRange(start=date(2020, 1, 1), end=date(2020, 12, 31)),
                validation=DateRange(start=date(2021, 1, 1), end=date(2021, 6, 30)),
            ),
        ),
        "final_holdout_id": "final-v1",
        "final_holdout": DateRange(start=date(2022, 1, 1), end=date(2022, 12, 31)),
    }
    payload.update(overrides)
    return SplitManifest.model_validate(payload)


def test_vendor_request_rejects_nested_runtime_secrets() -> None:
    with pytest.raises(ValidationError, match="runtime secret"):
        VendorRequest(
            provider="fake",
            dataset="bars",
            parameters={"auth": {"api_key": "do-not-persist"}},
        )


def test_acquisition_rejects_secret_response_metadata(tmp_path: Path) -> None:
    runner = AcquisitionRunner(LocalStorageBackend(tmp_path / "store"))
    request = VendorRequest(provider="fake", dataset="bars")
    with pytest.raises(AcquisitionError, match="runtime secret"):
        runner.acquire(MetadataAdapter({"authorization": "secret"}), request)


def test_acquisition_requires_aware_clock_and_same_time_records_do_not_collide(
    tmp_path: Path,
) -> None:
    request = VendorRequest(provider="fake", dataset="bars")
    backend = LocalStorageBackend(tmp_path / "store")
    naive = AcquisitionRunner(backend, now=lambda: datetime(2026, 8, 18, 10, 0))
    with pytest.raises(AcquisitionError, match="timezone-aware"):
        naive.acquire(MetadataAdapter(), request)

    timestamp = datetime(2026, 8, 18, 10, 0, tzinfo=UTC)
    runner = AcquisitionRunner(backend, now=lambda: timestamp)
    runner.acquire(MetadataAdapter(), request)
    runner.acquire(MetadataAdapter(), request)
    prefix = f"00_raw/fake/{request.request_sha256()}/acquisitions/"
    assert len(backend.list(prefix)) == 2


def test_raw_validation_can_detect_an_asset_with_no_rows_at_all() -> None:
    report = validate_raw_bars(
        [_raw()],
        expected_sessions=(date(2024, 1, 2),),
        expected_assets=("sec-1", "sec-2"),
    )
    missing = [item for item in report.anomalies if item.code == AnomalyCode.MISSING_SESSION]
    assert [(item.asset_id, item.message) for item in missing] == [
        ("sec-2", "expected session 2024-01-02 has no raw bars")
    ]


def test_expected_assets_require_sessions_and_unique_nonblank_ids() -> None:
    with pytest.raises(ValueError, match="requires expected_sessions"):
        validate_raw_bars([_raw()], expected_assets=("sec-1",))
    with pytest.raises(ValueError, match="duplicate identifiers"):
        validate_raw_bars(
            [_raw()],
            expected_sessions=(date(2024, 1, 2),),
            expected_assets=("sec-1", "sec-1"),
        )


def test_security_master_symbol_history_must_cover_listing_lifetime() -> None:
    record = SecurityRecord(
        security_id="sec-1",
        security_type=SecurityType.COMMON_STOCK,
        exchange="NASDAQ",
        listing_date=date(2020, 1, 1),
    )
    with pytest.raises(ValidationError, match="begin on the security listing_date"):
        SecurityMaster(
            version="sm-v1",
            securities=(record,),
            symbols=(
                SymbolPeriod(
                    security_id="sec-1",
                    symbol="AAA",
                    start_date=date(2020, 1, 2),
                ),
            ),
        )
    with pytest.raises(ValidationError, match="remain open-ended"):
        SecurityMaster(
            version="sm-v1",
            securities=(record,),
            symbols=(
                SymbolPeriod(
                    security_id="sec-1",
                    symbol="AAA",
                    start_date=date(2020, 1, 1),
                    end_date=date(2021, 1, 1),
                ),
            ),
        )


def test_security_master_rejects_nonfinite_and_duplicate_source_actions() -> None:
    with pytest.raises(ValidationError, match="finite"):
        CorporateAction(
            security_id="sec-1",
            action_type=CorporateActionType.SPLIT,
            effective_date=date(2021, 1, 1),
            split_ratio=math.inf,
        )

    value = _master()
    payload = value.model_dump(mode="python")
    payload["corporate_actions"] = (
        CorporateAction(
            security_id="sec-1",
            action_type=CorporateActionType.SPLIT,
            effective_date=date(2021, 1, 1),
            split_ratio=2.0,
            source_id="source-1",
        ),
        CorporateAction(
            security_id="sec-1",
            action_type=CorporateActionType.CASH_DIVIDEND,
            effective_date=date(2021, 2, 1),
            cash_amount=0.5,
            source_id="source-1",
        ),
    )
    with pytest.raises(ValidationError, match="source IDs must be unique"):
        SecurityMaster.model_validate(payload)


def test_symbol_lookup_rejects_dates_outside_security_lifetime() -> None:
    value = _master(delisting_date=date(2021, 12, 31))
    with pytest.raises(KeyError, match="not listed"):
        value.symbol_for("sec-1", date(2022, 1, 1))


def test_canonicalization_rejects_invalid_and_duplicate_raw_bars() -> None:
    invalid = replace(_raw(), close=math.nan)
    with pytest.raises(CanonicalizationError, match="finite and positive"):
        canonicalize_bars([invalid], _master())
    row = _raw()
    with pytest.raises(CanonicalizationError, match="duplicate"):
        canonicalize_bars([row, row], _master())


def test_resampling_is_input_order_independent_for_equal_time_assets() -> None:
    rows = [
        *[_canonical("b", minute) for minute in range(5)],
        *[_canonical("a", minute) for minute in range(5)],
    ]
    forward = resample_canonical_bars(rows, 5)
    reverse = resample_canonical_bars(reversed(rows), 5)
    assert forward == reverse
    assert [item.security_id for item in forward] == ["a", "b"]


def test_resampling_rejects_nonfinite_input_and_unsupported_frequency() -> None:
    invalid = replace(_canonical("a", 0), adjusted_close=math.nan)
    with pytest.raises(ResamplingError, match="finite and positive"):
        resample_canonical_bars([invalid], 5, require_complete=False)
    with pytest.raises(ValueError, match="unsupported"):
        resample_canonical_bars([], 3)  # type: ignore[arg-type]


def test_universe_rejects_nonfinite_and_duplicate_daily_liquidity() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        LiquidityObservation("sec-1", date(2024, 1, 1), math.nan, 100)

    policy = UniversePolicy(
        version="v1",
        target_size=1,
        trailing_observations=2,
        minimum_history_observations=1,
    )
    duplicate = LiquidityObservation("sec-1", date(2023, 12, 31), 100, 10)
    with pytest.raises(UniverseConstructionError, match="duplicate security/date"):
        build_universe_snapshot(
            _master(),
            [duplicate, duplicate],
            as_of=date(2024, 1, 2),
            policy=policy,
        )


def test_feature_boundary_rejects_nonfinite_input_and_derived_overflow() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        FeatureObservation("a", "A", "Tech", START, 1, 1, 1, math.nan, 1)

    huge = FeatureObservation(
        "a",
        "A",
        "Tech",
        START,
        1e308,
        1e308,
        1e308,
        1e308,
        1e308,
        1e308,
    )
    with pytest.raises(FeaturePipelineError, match="non-finite feature"):
        compute_features([huge])


def test_label_generation_rejects_nonfinite_derived_return() -> None:
    rows = [
        LabelObservation("a", START, 1e-308),
        LabelObservation("a", START + timedelta(minutes=5), 1e308),
    ]
    with pytest.raises(LabelGenerationError, match="non-finite future_return"):
        generate_labels(rows, policy=LabelPolicy(horizons_minutes=(5,)))


def test_split_schema_and_final_holdout_identity_are_strict() -> None:
    payload = _split_manifest().model_dump(mode="python")
    payload["schema_version"] = 2
    with pytest.raises(ValidationError, match="literal_error"):
        SplitManifest.model_validate(payload)
    with pytest.raises(ValidationError, match="distinct from routine fold IDs"):
        _split_manifest(final_holdout_id="fold-1")


def test_packing_rejects_nonfinite_duplicate_and_unrepresentable_samples(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite values"):
        TrainingSample("a", START, (math.nan,), (1.0,))

    row = TrainingSample("a", START, (1.0,), (2.0,))
    with pytest.raises(PackingError, match="duplicate security/timestamp"):
        pack_training_data(
            [row, row],
            tmp_path / "duplicate",
            feature_names=("f",),
            target_names=("t",),
            dataset_version="d1",
            split_version="s1",
        )
    huge = TrainingSample("a", START, (1e100,), (1.0,))
    with pytest.raises(PackingError, match="representable"):
        pack_training_data(
            [huge],
            tmp_path / "huge",
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
