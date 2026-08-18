"""Deterministic memory-mapped reference packing for model-ready training samples."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
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
        if self.metadata.get("schema_version") != 1:
            raise PackingError("unsupported packed dataset schema version")
        _verify_files(self.path, self.metadata["files"])
        self.features = np.load(self.path / "features.npy", mmap_mode="r")
        self.targets = np.load(self.path / "targets.npy", mmap_mode="r")
        self.timestamps_ns = np.load(self.path / "timestamps_ns.npy", mmap_mode="r")
        self.asset_ids = np.load(self.path / "asset_ids.npy", mmap_mode="r")
        expected_samples = int(self.metadata["sample_count"])
        if not all(
            len(array) == expected_samples
            for array in (self.features, self.targets, self.timestamps_ns, self.asset_ids)
        ):
            raise PackingError("packed arrays do not share the metadata sample count")

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
    feature_count = len(feature_names)
    target_count = len(target_names)
    if any(len(row.features) != feature_count for row in rows):
        raise PackingError("sample feature width does not match feature_names")
    if any(len(row.targets) != target_count for row in rows):
        raise PackingError("sample target width does not match target_names")
    if len(set(feature_names)) != feature_count or len(set(target_names)) != target_count:
        raise PackingError("feature and target names must be unique")

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
        timestamps[index] = int(row.timestamp.timestamp() * 1_000_000_000)
        assets[index] = row.security_id
    for array in (features, targets, timestamps, assets):
        array.flush()


def _file_record(path: Path) -> dict[str, int | str]:
    return {"size": path.stat().st_size, "sha256": _sha256_file(path)}


def _verify_files(path: Path, records: dict[str, dict[str, int | str]]) -> None:
    for name, record in records.items():
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
