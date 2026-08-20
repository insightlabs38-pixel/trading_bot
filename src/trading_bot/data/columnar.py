"""Checksummed Parquet + Zstd reference representation for model-ready training data."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable, cast

from trading_bot.data.packing import TrainingSample
from trading_bot.storage.base import fsync_directory, fsync_file


class ColumnarDatasetError(RuntimeError):
    """Raised when a columnar dataset violates its immutable representation contract."""


_DATA_FILE = "data.parquet"
_MANIFEST_FILE = "manifest.json"
_MANIFEST_SHA256_FILE = "manifest.sha256"
_FORMAT = "parquet_zstd_reference"
_SCHEMA_VERSION = 1
_FLOAT32_MAX = 3.4028234663852886e38


@dataclass(frozen=True, slots=True)
class ColumnarWriteResult:
    path: Path
    sample_count: int
    feature_count: int
    target_count: int
    data_sha256: str
    manifest_sha256: str


class ColumnarDataset:
    """Validated immutable Parquet dataset with a checksummed semantic manifest."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.parquet_path = self.path / _DATA_FILE
        manifest_bytes = _read_verified_manifest(self.path)
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        try:
            parsed = json.loads(manifest_bytes)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ColumnarDatasetError(f"invalid columnar manifest JSON: {exc}") from exc
        self.manifest = _validate_manifest(parsed)
        _verify_data_file(self.parquet_path, self.manifest["data_file"])
        _, parquet = _arrow_modules()
        try:
            metadata = parquet.read_metadata(str(self.parquet_path), memory_map=True)
            schema = parquet.read_schema(str(self.parquet_path), memory_map=True)
        except (OSError, ValueError) as exc:
            raise ColumnarDatasetError(f"invalid Parquet dataset: {exc}") from exc
        _validate_parquet_contract(metadata, schema, self.manifest)

    @property
    def sample_count(self) -> int:
        return int(self.manifest["sample_count"])

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(cast(list[str], self.manifest["feature_names"]))

    @property
    def target_names(self) -> tuple[str, ...]:
        return tuple(cast(list[str], self.manifest["target_names"]))

    def to_training_samples(self) -> tuple[TrainingSample, ...]:
        """Materialize logical samples for reference validation and CPU-side tooling."""
        _, parquet = _arrow_modules()
        try:
            table = parquet.read_table(str(self.parquet_path), memory_map=True)
        except (OSError, ValueError) as exc:
            raise ColumnarDatasetError(f"unable to read Parquet dataset: {exc}") from exc
        columns = cast(dict[str, list[object]], table.to_pydict())
        feature_columns = _feature_columns(len(self.feature_names))
        target_columns = _target_columns(len(self.target_names))
        rows: list[TrainingSample] = []
        for index in range(self.sample_count):
            security_id = columns["security_id"][index]
            timestamp_ns = columns["timestamp_ns"][index]
            if not isinstance(security_id, str) or not isinstance(timestamp_ns, int):
                raise ColumnarDatasetError("Parquet identity columns contain invalid values")
            features = tuple(float(columns[name][index]) for name in feature_columns)
            targets = tuple(float(columns[name][index]) for name in target_columns)
            rows.append(
                TrainingSample(
                    security_id=security_id,
                    timestamp=_ns_to_timestamp(timestamp_ns),
                    features=features,
                    targets=targets,
                )
            )
        return tuple(rows)


def write_training_parquet(
    samples: Iterable[TrainingSample],
    destination: str | Path,
    *,
    feature_names: tuple[str, ...],
    target_names: tuple[str, ...],
    dataset_version: str,
    split_version: str,
) -> ColumnarWriteResult:
    """Write an immutable deterministic-order Parquet + Zstd reference dataset."""
    rows = tuple(samples)
    _validate_write_contract(
        rows,
        feature_names=feature_names,
        target_names=target_names,
        dataset_version=dataset_version,
        split_version=split_version,
    )
    ordered = tuple(sorted(rows, key=lambda row: (row.timestamp, row.security_id)))
    destination = Path(destination)
    if destination.exists():
        raise ColumnarDatasetError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=str(destination.parent))
    )
    try:
        parquet_path = temporary / _DATA_FILE
        _write_parquet(
            ordered,
            parquet_path,
            feature_names=feature_names,
            target_names=target_names,
            dataset_version=dataset_version,
            split_version=split_version,
        )
        fsync_file(parquet_path)
        data_sha = _sha256_file(parquet_path)
        data_size = parquet_path.stat().st_size
        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "format": _FORMAT,
            "dataset_version": dataset_version,
            "split_version": split_version,
            "sample_count": len(ordered),
            "feature_count": len(feature_names),
            "target_count": len(target_names),
            "feature_names": list(feature_names),
            "target_names": list(target_names),
            "physical_columns": _physical_columns(len(feature_names), len(target_names)),
            "data_file": {
                "name": _DATA_FILE,
                "size": data_size,
                "sha256": data_sha,
                "compression": "zstd",
                "parquet_version": "2.6",
            },
        }
        manifest_bytes = _canonical_json_bytes(manifest)
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        manifest_path = temporary / _MANIFEST_FILE
        checksum_path = temporary / _MANIFEST_SHA256_FILE
        manifest_path.write_bytes(manifest_bytes)
        checksum_path.write_text(f"{manifest_sha}\n", encoding="ascii")
        fsync_file(manifest_path)
        fsync_file(checksum_path)
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return ColumnarWriteResult(
        path=destination,
        sample_count=len(ordered),
        feature_count=len(feature_names),
        target_count=len(target_names),
        data_sha256=data_sha,
        manifest_sha256=manifest_sha,
    )


def _write_parquet(
    rows: tuple[TrainingSample, ...],
    path: Path,
    *,
    feature_names: tuple[str, ...],
    target_names: tuple[str, ...],
    dataset_version: str,
    split_version: str,
) -> None:
    arrow, parquet = _arrow_modules()
    feature_columns = _feature_columns(len(feature_names))
    target_columns = _target_columns(len(target_names))
    physical_columns = ["security_id", "timestamp_ns", *feature_columns, *target_columns]
    arrays = [
        arrow.array([row.security_id for row in rows], type=arrow.string()),
        arrow.array([_timestamp_to_ns(row.timestamp) for row in rows], type=arrow.int64()),
    ]
    arrays.extend(
        arrow.array([row.features[index] for row in rows], type=arrow.float32())
        for index in range(len(feature_names))
    )
    arrays.extend(
        arrow.array([row.targets[index] for row in rows], type=arrow.float32())
        for index in range(len(target_names))
    )
    schema_metadata = {
        b"trading_bot.schema_version": str(_SCHEMA_VERSION).encode("ascii"),
        b"trading_bot.format": _FORMAT.encode("ascii"),
        b"trading_bot.dataset_version": dataset_version.encode("utf-8"),
        b"trading_bot.split_version": split_version.encode("utf-8"),
        b"trading_bot.feature_names": _canonical_json_bytes(list(feature_names)),
        b"trading_bot.target_names": _canonical_json_bytes(list(target_names)),
    }
    schema = arrow.schema(
        [
            arrow.field("security_id", arrow.string(), nullable=False),
            arrow.field("timestamp_ns", arrow.int64(), nullable=False),
            *[
                arrow.field(name, arrow.float32(), nullable=False)
                for name in feature_columns + target_columns
            ],
        ],
        metadata=schema_metadata,
    )
    table = arrow.Table.from_arrays(arrays, schema=schema)
    if table.column_names != physical_columns:
        raise ColumnarDatasetError("internal Parquet column construction mismatch")
    parquet.write_table(
        table,
        str(path),
        version="2.6",
        compression="zstd",
        use_dictionary=["security_id"],
        write_statistics=True,
        write_page_checksum=True,
        row_group_size=min(len(rows), 65_536),
    )


def _validate_write_contract(
    rows: tuple[TrainingSample, ...],
    *,
    feature_names: tuple[str, ...],
    target_names: tuple[str, ...],
    dataset_version: str,
    split_version: str,
) -> None:
    if not rows:
        raise ColumnarDatasetError("at least one training sample is required")
    _validate_names(feature_names, field_name="feature_names")
    _validate_names(target_names, field_name="target_names")
    if not dataset_version.strip() or not split_version.strip():
        raise ColumnarDatasetError("dataset_version and split_version must not be blank")
    if any(len(row.features) != len(feature_names) for row in rows):
        raise ColumnarDatasetError("sample feature width does not match feature_names")
    if any(len(row.targets) != len(target_names) for row in rows):
        raise ColumnarDatasetError("sample target width does not match target_names")
    identities = [(row.security_id, row.timestamp) for row in rows]
    if len(set(identities)) != len(identities):
        raise ColumnarDatasetError("duplicate security/timestamp training samples are not allowed")
    for row in rows:
        if any(abs(value) > _FLOAT32_MAX for value in row.features + row.targets):
            raise ColumnarDatasetError("training values must be representable as finite float32 values")


def _read_verified_manifest(path: Path) -> bytes:
    try:
        manifest_bytes = (path / _MANIFEST_FILE).read_bytes()
        expected = (path / _MANIFEST_SHA256_FILE).read_text(encoding="ascii").strip().lower()
    except (OSError, UnicodeError) as exc:
        raise ColumnarDatasetError(f"invalid columnar manifest integrity files: {exc}") from exc
    _require_sha256(expected, field_name=_MANIFEST_SHA256_FILE)
    actual = hashlib.sha256(manifest_bytes).hexdigest()
    if actual != expected:
        raise ColumnarDatasetError("columnar manifest checksum mismatch")
    return manifest_bytes


def _validate_manifest(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ColumnarDatasetError("columnar manifest must be a JSON object")
    manifest = cast(dict[str, Any], value)
    if manifest.get("schema_version") != _SCHEMA_VERSION:
        raise ColumnarDatasetError("unsupported columnar schema version")
    if manifest.get("format") != _FORMAT:
        raise ColumnarDatasetError("unsupported columnar dataset format")
    for key in ("dataset_version", "split_version"):
        item = manifest.get(key)
        if not isinstance(item, str) or not item.strip():
            raise ColumnarDatasetError(f"columnar manifest field {key} must be a non-blank string")
    for key in ("sample_count", "feature_count", "target_count"):
        item = manifest.get(key)
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            raise ColumnarDatasetError(f"columnar manifest field {key} must be a positive integer")
    feature_count = int(manifest["feature_count"])
    target_count = int(manifest["target_count"])
    _validate_manifest_names(manifest.get("feature_names"), feature_count, "feature_names")
    _validate_manifest_names(manifest.get("target_names"), target_count, "target_names")
    expected_columns = _physical_columns(feature_count, target_count)
    if manifest.get("physical_columns") != expected_columns:
        raise ColumnarDatasetError("columnar physical column manifest is inconsistent")
    record = manifest.get("data_file")
    if not isinstance(record, dict):
        raise ColumnarDatasetError("columnar manifest data_file record is invalid")
    if record.get("name") != _DATA_FILE:
        raise ColumnarDatasetError("columnar manifest data_file name is invalid")
    if record.get("compression") != "zstd" or record.get("parquet_version") != "2.6":
        raise ColumnarDatasetError("columnar manifest Parquet format settings are invalid")
    return manifest


def _verify_data_file(path: Path, record_value: object) -> None:
    if not isinstance(record_value, dict):
        raise ColumnarDatasetError("columnar manifest data_file record is invalid")
    record = cast(dict[str, Any], record_value)
    size = record.get("size")
    checksum = record.get("sha256")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ColumnarDatasetError("columnar data_file size is invalid")
    if not isinstance(checksum, str):
        raise ColumnarDatasetError("columnar data_file SHA-256 is invalid")
    _require_sha256(checksum, field_name="data_file.sha256")
    if not path.is_file():
        raise ColumnarDatasetError(f"columnar Parquet file is missing: {path}")
    if path.stat().st_size != size:
        raise ColumnarDatasetError("columnar Parquet file size mismatch")
    if _sha256_file(path) != checksum.lower():
        raise ColumnarDatasetError("columnar Parquet file checksum mismatch")


def _validate_parquet_contract(metadata: Any, schema: Any, manifest: dict[str, Any]) -> None:
    expected_columns = cast(list[str], manifest["physical_columns"])
    if list(schema.names) != expected_columns:
        raise ColumnarDatasetError("Parquet physical columns do not match manifest")
    if int(metadata.num_rows) != int(manifest["sample_count"]):
        raise ColumnarDatasetError("Parquet row count does not match manifest")
    if int(metadata.num_columns) != len(expected_columns):
        raise ColumnarDatasetError("Parquet column count does not match manifest")
    if int(metadata.num_row_groups) <= 0:
        raise ColumnarDatasetError("Parquet dataset must contain at least one row group")
    for row_group_index in range(int(metadata.num_row_groups)):
        row_group = metadata.row_group(row_group_index)
        for column_index in range(int(metadata.num_columns)):
            compression = str(row_group.column(column_index).compression).upper()
            if compression != "ZSTD":
                raise ColumnarDatasetError("Parquet column is not Zstd-compressed")
    file_metadata = metadata.metadata or {}
    expected_metadata = {
        b"trading_bot.schema_version": str(_SCHEMA_VERSION).encode("ascii"),
        b"trading_bot.format": _FORMAT.encode("ascii"),
        b"trading_bot.dataset_version": str(manifest["dataset_version"]).encode("utf-8"),
        b"trading_bot.split_version": str(manifest["split_version"]).encode("utf-8"),
        b"trading_bot.feature_names": _canonical_json_bytes(manifest["feature_names"]),
        b"trading_bot.target_names": _canonical_json_bytes(manifest["target_names"]),
    }
    for key, expected in expected_metadata.items():
        if file_metadata.get(key) != expected:
            raise ColumnarDatasetError(f"Parquet semantic metadata mismatch for {key.decode()}")


def _validate_names(names: tuple[str, ...], *, field_name: str) -> None:
    if not names or any(not name.strip() for name in names):
        raise ColumnarDatasetError(f"{field_name} must contain non-blank names")
    if len(set(names)) != len(names):
        raise ColumnarDatasetError(f"{field_name} must be unique")


def _validate_manifest_names(value: object, expected_count: int, field_name: str) -> None:
    if not isinstance(value, list) or len(value) != expected_count:
        raise ColumnarDatasetError(f"columnar manifest {field_name} width is invalid")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ColumnarDatasetError(f"columnar manifest {field_name} must contain non-blank strings")
    if len(set(value)) != len(value):
        raise ColumnarDatasetError(f"columnar manifest {field_name} must be unique")


def _feature_columns(count: int) -> list[str]:
    return [f"feature_{index:04d}" for index in range(count)]


def _target_columns(count: int) -> list[str]:
    return [f"target_{index:04d}" for index in range(count)]


def _physical_columns(feature_count: int, target_count: int) -> list[str]:
    return [
        "security_id",
        "timestamp_ns",
        *_feature_columns(feature_count),
        *_target_columns(target_count),
    ]


def _timestamp_to_ns(value: datetime) -> int:
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    delta = value.astimezone(UTC) - epoch
    microseconds = (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds
    nanoseconds = microseconds * 1_000
    if not -(2**63) <= nanoseconds < 2**63:
        raise ColumnarDatasetError("timestamp is outside the signed int64 nanosecond range")
    return nanoseconds


def _ns_to_timestamp(value: int) -> datetime:
    if value % 1_000 != 0:
        raise ColumnarDatasetError("timestamp_ns cannot be represented exactly by Python datetime")
    try:
        return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=value // 1_000)
    except OverflowError as exc:
        raise ColumnarDatasetError("timestamp_ns is outside Python datetime range") from exc


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, *, field_name: str) -> None:
    normalized = value.lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ColumnarDatasetError(f"{field_name} must be 64 hexadecimal characters")


def _arrow_modules() -> tuple[Any, Any]:
    try:
        return import_module("pyarrow"), import_module("pyarrow.parquet")
    except ImportError as exc:
        raise ColumnarDatasetError(
            "Parquet support requires the CPU dependency group with pyarrow installed"
        ) from exc
