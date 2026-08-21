"""CPU-safe disk, dataset, and storage health checks used by the circuit breaker."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from trading_bot.recovery.types import GateResult
from trading_bot.storage.base import StorageBackend


def check_disk(path: str | Path, *, minimum_free_bytes: int) -> GateResult:
    if minimum_free_bytes < 0:
        raise ValueError("minimum_free_bytes must be non-negative")
    free = shutil.disk_usage(Path(path)).free
    return GateResult(
        name="disk",
        passed=free >= minimum_free_bytes,
        detail=f"free_bytes={free}; minimum={minimum_free_bytes}",
    )


def check_dataset_sample(path: str | Path, *, expected_sha256: str) -> GateResult:
    source = Path(path)
    if not source.is_file():
        return GateResult(name="dataset", passed=False, detail=f"missing dataset sample: {source}")
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    return GateResult(
        name="dataset",
        passed=actual == expected_sha256,
        detail=f"sha256={actual}",
    )


def check_storage_object(
    backend: StorageBackend,
    *,
    key: str,
    expected_sha256: str,
) -> GateResult:
    try:
        exists = backend.exists(key)
        verified = exists and backend.verify_checksum(key, expected_sha256)
    except Exception as exc:
        return GateResult(name="storage", passed=False, detail=f"{type(exc).__name__}: {exc}")
    return GateResult(
        name="storage",
        passed=verified,
        detail=f"key={key}; exists={exists}; checksum_verified={verified}",
    )
