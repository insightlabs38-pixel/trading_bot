"""Known-good CPU canary used by CI and non-GPU health checks."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from time import perf_counter

from pydantic import Field

from trading_bot.config.base import FrozenConfigModel
from trading_bot.storage.base import StorageBackend


class GoldenCanarySpec(FrozenConfigModel):
    schema_version: int = 1
    model_id: str = "linear-y-equals-2x-plus-1"
    expected_mse_max: float = Field(default=1.0e-12, ge=0.0)
    min_predictions_per_second: float = Field(default=100.0, gt=0.0)
    benchmark_iterations: int = Field(default=10_000, gt=0)


class GoldenCanaryResult(FrozenConfigModel):
    passed: bool
    mse: float = Field(ge=0.0)
    predictions_per_second: float = Field(gt=0.0)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    storage_verified: bool


def run_cpu_golden_canary(
    backend: StorageBackend,
    *,
    storage_key: str,
    spec: GoldenCanarySpec | None = None,
) -> GoldenCanaryResult:
    """Run a tiny deterministic model/data/save-load/eval/storage/throughput canary."""
    active = spec or GoldenCanarySpec()
    xs = (0.0, 1.0, 2.0, 3.0)
    expected = (1.0, 3.0, 5.0, 7.0)
    slope = 2.0
    intercept = 1.0
    model: dict[str, float | str] = {
        "slope": slope,
        "intercept": intercept,
        "model_id": active.model_id,
    }

    predictions = tuple(slope * value + intercept for value in xs)
    mse = sum((actual - target) ** 2 for actual, target in zip(predictions, expected, strict=True)) / len(xs)

    started = perf_counter()
    sink = 0.0
    for _ in range(active.benchmark_iterations):
        for value in xs:
            sink += slope * value + intercept
    elapsed = perf_counter() - started
    if sink <= 0.0 or elapsed <= 0.0:
        raise AssertionError("canary benchmark produced an invalid timing result")
    predictions_per_second = active.benchmark_iterations * len(xs) / elapsed

    with tempfile.TemporaryDirectory(prefix="trading-bot-golden-canary-") as directory:
        artifact = Path(directory) / "canary.json"
        artifact.write_text(json.dumps(model, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        restored = json.loads(artifact.read_text(encoding="utf-8"))
        if restored != model:
            raise ValueError("golden canary save/load round-trip failed")
        checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
        backend.upload(artifact, storage_key, expected_sha256=checksum)
        storage_verified = backend.verify_checksum(storage_key, checksum)

    return GoldenCanaryResult(
        passed=(
            mse <= active.expected_mse_max
            and predictions_per_second >= active.min_predictions_per_second
            and storage_verified
        ),
        mse=mse,
        predictions_per_second=predictions_per_second,
        artifact_sha256=checksum,
        storage_verified=storage_verified,
    )
