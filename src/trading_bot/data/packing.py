"""Deterministic memory-mapped reference packing for model-ready training samples."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


class PackingError(RuntimeError):
    """Raised when packed dataset inputs or integrity checks fail."""


@dataclass(frozen=True, slots=True)
class TrainingSample:
    security_id: str
    timestamp: datetime
    features: tuple[float, ...]
    targets: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.security_id.strip():
            raise ValueError("security_id must not be blank")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        if not self.features:
            raise ValueError("features must not be empty")
        if not self.targets:
            raise ValueError("targets must not be empty")
        if any(not math.isfinite(value) for value in self.features):
            raise ValueError("features must contain only finite values")
        if any(not math.isfinite(value) for value in self.targets):
            raise ValueError("targets must contain only finite values")


@dataclass(frozen=True, slots=True)
class PackingResult:
    path: Path
    sample_count: int
    feature_count: int
    target_count: int
    dataset_sha256: str


@dataclass(frozen=True, slots=True)
class LoaderBenchmark:
    sample_count: int
    bytes_read: int
    elapsed_seconds: float

    @property
    def samples_per_second(self) -> float:
        return 0.0 if self.elapsed_seconds <= 0 else self.sample_count / self.elapsed_seconds

    @property
    def mib_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.bytes_read / (1024 * 1024) / self.elapsed_seconds


class PackedDataset:
    """Validated memory-mapped dataset opened without copying full arrays into RAM."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        try:
            self.metadata = json.loads((self.path / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PackingError(f"invalid packed dataset metadata: {exc}") from exc
        _validate_metadata(self.metadata)
        _verify_files(self.path, self.metadata["files"])
        try:
            self.features = np.load(self.path / "features.npy", mmap_mode="r")
            self.targets = np.load(self.path / "targets.npy", mmap_mode="r")
            self.timestamps_ns = np.load(self.path / "timestamps_ns.npy", mmap_mode="r")
            self.asset_ids = np.load(self.path / "asset_ids.npy", mmap_mode="r")
        except (OSError, ValueError) as exc:
            raise PackingError(f"invalid packed array: {exc}") from exc
        _validate_array_shapes(self)

    def iter_batches(
        self,
        batch_size: int,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(self.features), batch_size):
            stop = min(len(self.features), start + batch_size)
            yield (
                self.features[start:stop],
                self.targets[start:stop],
                self.timestamps_ns[start:stop],
                self.asset_ids[start:stop],
            )


def pack_training_data(
    samples: Iterable[TrainingSample],
    destination: str | Path,
    *,
    feature_names: tuple[str, ...],
    target_names: tuple[str, ...],
    dataset_version: str,
    split_version: str,
    overwrite: bool = False,
) -> PackingResult:
    """Write a deterministic NPY/memmap training pack with per-file integrity metadata."""
    rows = tuple(samples)
    if not rows:
        raise PackingError("at least one training sample is required")
    if not feature_names or not target_names:
        raise PackingError("feature_names and target_names must not be empty")
    if any(not name.strip() for name in feature_names + target_names):
        raise PackingError("feature and target names must not contain blanks")
    if not dataset_version.strip() or not split_version.strip():
        raise PackingError("dataset_version and split_version must not be blank")

    feature_count = len(feature_names)
    target_count = len(target_names)
    if any(len(row.features) != feature_count for row in rows):
        raise PackingError("sample feature width does not match feature_names")
    if any(len(row.targets) != target_count for row in rows):
        raise PackingError("sample target width does not match target_names")
    if len(set(feature_names)) != feature_count or len(set(target_names)) != target_count:
        raise PackingError("feature and target names must be unique")
    identities = [(row.security_id, row.timestamp) for row in rows]
    if len(set(identities)) != len(identities):
        raise PackingError("duplicate security/timestamp training samples are not allowed")
    _validate_float32_representable(rows)

    ordered = tuple(sorted(rows, key=lambda row: (row.timestamp, row.security_id)))
    destination = Path(destination)
    if destination.exists() and not overwrite:
        raise PackingError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=str(destination.parent))
    )
    try:
        _write_arrays(temporary, ordered, feature_count, target_count)
        files = {
            name: _file_record(temporary / name)
            for name in ("features.npy", "targets.npy", "timestamps_ns.npy", "asset_ids.npy")
        }
        metadata = {
            "schema_version": 1,
            "format": "numpy_npy_memmap_reference",
            "dataset_version": dataset_version,
            "split_version": split_version,
            "sample_count": len(ordered),
            "feature_count": feature_count,
            "target_count": target_count,
            "feature_names": list(feature_names),
            "target_names": list(target_names),
            "files": files,
        }
        metadata_bytes = json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        (temporary / "metadata.json").write_bytes(metadata_bytes)
        dataset_sha = hashlib.sha256(metadata_bytes).hexdigest()
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return PackingResult(destination, len(ordered), feature_count, target_count, dataset_sha)


def benchmark_loader(dataset: PackedDataset, *, batch_size: int = 1024) -> LoaderBenchmark:
    """Read every batch and touch each array so measurements include memory-map page access."""
    start = time.perf_counter()
    count = 0
    bytes_read = 0
    checksum = 0.0
    for features, targets, timestamps, assets in dataset.iter_batches(batch_size):
        count += len(features)
        bytes_read += features.nbytes + targets.nbytes + timestamps.nbytes + assets.nbytes
        checksum += float(np.sum(features)) + float(np.sum(targets))
        checksum += float(np.sum(timestamps.astype(np.float64)))
        checksum += float(sum(len(str(value)) for value in assets))
    if not np.isfinite(checksum):
        raise PackingError("loader benchmark encountered non-finite data")
    return LoaderBenchmark(count, bytes_read, max(0.0, time.perf_counter() - start))


def _validate_float32_representable(rows: tuple[TrainingSample, ...]) -> None:
    limit = float(np.finfo(np.float32).max)
    for row in rows:
        if any(abs(value) > limit for value in row.features + row.targets):
            raise PackingError("training values must be representable as finite float32 values")


def _write_arrays(
    path: Path,
    rows: tuple[TrainingSample, ...],
    feature_count: int,
    target_count: int,
) -> None:
    sample_count = len(rows)
    max_asset_length = max(len(row.security_id) for row in rows)
    features = np.lib.format.open_memmap(
        path / "features.npy", mode="w+", dtype=np.float32, shape=(sample_count, feature_count)
    )
    targets = np.lib.format.open_memmap(
        path / "targets.npy", mode="w+", dtype=np.float32, shape=(sample_count, target_count)
    )
    timestamps = np.lib.format.open_memmap(
        path / "timestamps_ns.npy", mode="w+", dtype=np.int64, shape=(sample_count,)
    )
    assets = np.lib.format.open_memmap(
        path / "asset_ids.npy", mode="w+", dtype=f"U{max_asset_length}", shape=(sample_count,)
    )
    for index, row in enumerate(rows):
        features[index] = row.features
        targets[index] = row.targets
        timestamps[index] = _timestamp_to_ns(row.timestamp)
        assets[index] = row.security_id
    for array in (features, targets, timestamps, assets):
        array.flush()


def _timestamp_to_ns(value: datetime) -> int:
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    delta = value.astimezone(UTC) - epoch
    microseconds = (
        (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds
    )
    nanoseconds = microseconds * 1_000
    limits = np.iinfo(np.int64)
    if not limits.min <= nanoseconds <= limits.max:
        raise PackingError("timestamp is outside the signed int64 nanosecond range")
    return nanoseconds


def _validate_metadata(metadata: object) -> None:
    if not isinstance(metadata, dict):
        raise PackingError("packed dataset metadata must be a JSON object")
    if metadata.get("schema_version") != 1:
        raise PackingError("unsupported packed dataset schema version")
    if metadata.get("format") != "numpy_npy_memmap_reference":
        raise PackingError("unsupported packed dataset format")
    required_files = {"features.npy", "targets.npy", "timestamps_ns.npy", "asset_ids.npy"}
    records = metadata.get("files")
    if not isinstance(records, dict) or set(records) != required_files:
        raise PackingError("packed dataset metadata must describe all required array files")
    for key in ("sample_count", "feature_count", "target_count"):
        value = metadata.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise PackingError(f"packed dataset metadata field {key} must be a non-negative integer")


def _validate_array_shapes(dataset: PackedDataset) -> None:
    samples = int(dataset.metadata["sample_count"])
    features = int(dataset.metadata["feature_count"])
    targets = int(dataset.metadata["target_count"])
    if dataset.features.shape != (samples, features):
        raise PackingError("feature array shape does not match metadata")
    if dataset.targets.shape != (samples, targets):
        raise PackingError("target array shape does not match metadata")
    if dataset.timestamps_ns.shape != (samples,):
        raise PackingError("timestamp array shape does not match metadata")
    if dataset.asset_ids.shape != (samples,):
        raise PackingError("asset ID array shape does not match metadata")
    if dataset.features.dtype != np.float32 or dataset.targets.dtype != np.float32:
        raise PackingError("feature and target arrays must use float32")
    if dataset.timestamps_ns.dtype != np.int64:
        raise PackingError("timestamp array must use int64 nanoseconds")


def _file_record(path: Path) -> dict[str, int | str]:
    return {"size": path.stat().st_size, "sha256": _sha256_file(path)}


def _verify_files(path: Path, records: dict[str, dict[str, int | str]]) -> None:
    for name, record in records.items():
        if not isinstance(record, dict) or "size" not in record or "sha256" not in record:
            raise PackingError(f"invalid packed file metadata: {name}")
        file_path = path / name
        if not file_path.is_file():
            raise PackingError(f"packed file missing: {name}")
        if file_path.stat().st_size != int(record["size"]):
            raise PackingError(f"packed file size mismatch: {name}")
        if _sha256_file(file_path) != str(record["sha256"]):
            raise PackingError(f"packed file checksum mismatch: {name}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
