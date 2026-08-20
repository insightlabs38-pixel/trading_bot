"""CPU verification for the Parquet + Zstd Phase 3 reference representation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import polars as pl
import pyarrow.parquet as pq
import pytest

from trading_bot.data.columnar import (
    ColumnarDataset,
    ColumnarDatasetError,
    write_training_parquet,
)
from trading_bot.data.packing import TrainingSample

START = datetime(2024, 1, 2, 14, 30, 0, 123456, tzinfo=UTC)


def _samples() -> tuple[TrainingSample, ...]:
    return (
        TrainingSample("b", START + timedelta(minutes=1), (2.0, 2.5), (0.5,)),
        TrainingSample("a", START, (1.0, 1.5), (-0.25,)),
        TrainingSample("b", START, (3.0, 3.5), (0.75,)),
    )


def _write(path: Path):
    return write_training_parquet(
        _samples(),
        path,
        feature_names=("return_5m", "realized_volatility"),
        target_names=("future_return_15m",),
        dataset_version="dataset-v1",
        split_version="split-v1",
    )


def test_parquet_zstd_round_trip_preserves_logical_samples_and_exact_timestamps(
    tmp_path: Path,
) -> None:
    result = _write(tmp_path / "columnar")
    dataset = ColumnarDataset(result.path)

    assert dataset.sample_count == 3
    assert dataset.feature_names == ("return_5m", "realized_volatility")
    assert dataset.target_names == ("future_return_15m",)
    assert dataset.to_training_samples() == tuple(
        sorted(_samples(), key=lambda row: (row.timestamp, row.security_id))
    )

    table = pq.read_table(dataset.parquet_path)
    timestamp_ns = table.column("timestamp_ns").to_pylist()[0]
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    delta = START - epoch
    expected = ((delta.days * 86_400 + delta.seconds) * 1_000_000 + 123456) * 1_000
    assert timestamp_ns == expected


def test_parquet_zstd_is_deterministic_and_compressed_in_every_column(tmp_path: Path) -> None:
    first = _write(tmp_path / "first")
    second = _write(tmp_path / "second")

    assert first.data_sha256 == second.data_sha256
    assert first.manifest_sha256 == second.manifest_sha256
    assert (first.path / "data.parquet").read_bytes() == (second.path / "data.parquet").read_bytes()
    assert (first.path / "manifest.json").read_bytes() == (
        second.path / "manifest.json"
    ).read_bytes()

    metadata = pq.read_metadata(first.path / "data.parquet")
    assert metadata.num_row_groups > 0
    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        assert {
            row_group.column(column_index).compression
            for column_index in range(metadata.num_columns)
        } == {"ZSTD"}


def test_parquet_reference_is_readable_by_polars_and_duckdb(tmp_path: Path) -> None:
    result = _write(tmp_path / "columnar")
    parquet_path = result.path / "data.parquet"

    polars_frame = pl.read_parquet(parquet_path)
    assert polars_frame.columns == [
        "security_id",
        "timestamp_ns",
        "feature_0000",
        "feature_0001",
        "target_0000",
    ]
    assert polars_frame.height == 3
    assert polars_frame.get_column("security_id").to_list() == ["a", "b", "b"]

    escaped_path = str(parquet_path).replace("'", "''")
    with duckdb.connect(database=":memory:") as connection:
        count, first_asset = connection.execute(
            f"SELECT COUNT(*), MIN(security_id) FROM read_parquet('{escaped_path}')"
        ).fetchone()
    assert count == 3
    assert first_asset == "a"


def test_columnar_reader_rejects_tampered_parquet_bytes(tmp_path: Path) -> None:
    result = _write(tmp_path / "columnar")
    parquet_path = result.path / "data.parquet"
    payload = bytearray(parquet_path.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    parquet_path.write_bytes(payload)

    with pytest.raises(ColumnarDatasetError, match="checksum mismatch"):
        ColumnarDataset(result.path)


def test_columnar_reader_rejects_tampered_manifest(tmp_path: Path) -> None:
    result = _write(tmp_path / "columnar")
    manifest_path = result.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split_version"] = "tampered"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ColumnarDatasetError, match="manifest checksum mismatch"):
        ColumnarDataset(result.path)


def test_columnar_reader_cross_checks_manifest_against_embedded_parquet_metadata(
    tmp_path: Path,
) -> None:
    result = _write(tmp_path / "columnar")
    manifest_path = result.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_version"] = "different-but-checksummed"
    manifest_bytes = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    (result.path / "manifest.sha256").write_text(
        f"{hashlib.sha256(manifest_bytes).hexdigest()}\n",
        encoding="ascii",
    )

    with pytest.raises(ColumnarDatasetError, match="semantic metadata mismatch"):
        ColumnarDataset(result.path)


def test_columnar_writer_rejects_duplicate_overflow_and_existing_destination(
    tmp_path: Path,
) -> None:
    row = TrainingSample("a", START, (1.0,), (2.0,))
    with pytest.raises(ColumnarDatasetError, match="duplicate"):
        write_training_parquet(
            (row, row),
            tmp_path / "duplicate",
            feature_names=("f",),
            target_names=("t",),
            dataset_version="d1",
            split_version="s1",
        )

    huge = TrainingSample("a", START, (1e100,), (1.0,))
    with pytest.raises(ColumnarDatasetError, match="float32"):
        write_training_parquet(
            (huge,),
            tmp_path / "overflow",
            feature_names=("f",),
            target_names=("t",),
            dataset_version="d1",
            split_version="s1",
        )

    destination = tmp_path / "existing"
    _write(destination)
    with pytest.raises(ColumnarDatasetError, match="already exists"):
        _write(destination)
