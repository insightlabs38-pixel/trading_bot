"""Immutable prediction artifacts that can be evaluated without importing training code."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable, cast

import torch
from torch import Tensor

from trading_bot.storage.base import fsync_directory, fsync_file
from trading_bot.training.contracts import ModelOutput, TradingModel, TrainingBatch

_DATA_FILE = "predictions.parquet"
_MANIFEST_FILE = "manifest.json"
_MANIFEST_SHA256_FILE = "manifest.sha256"
_FORMAT = "phase5_prediction_parquet"
_SCHEMA_VERSION = 1


class PredictionArtifactError(RuntimeError):
    """Raised when a prediction artifact violates its immutable contract."""


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """One evaluation-ready prediction with exact sample identity."""

    asset_id: str
    timestamp_ns: int
    target: float
    expected_return: float | None = None
    rank_score: float | None = None
    direction_probability: float | None = None
    volatility: float | None = None
    uncertainty: float | None = None
    quantiles: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.asset_id.strip():
            raise ValueError("prediction asset_id must not be blank")
        values = [self.target]
        values.extend(
            value
            for value in (
                self.expected_return,
                self.rank_score,
                self.direction_probability,
                self.volatility,
                self.uncertainty,
            )
            if value is not None
        )
        if self.quantiles is not None:
            values.extend(self.quantiles)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("prediction values must be finite")


@dataclass(frozen=True, slots=True)
class PredictionWriteResult:
    path: Path
    record_count: int
    data_sha256: str
    manifest_sha256: str


class PredictionArtifact:
    """Validated reader used by Phase 6 without importing the trainer implementation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        manifest_bytes = _read_verified_manifest(self.path)
        try:
            value = json.loads(manifest_bytes)
        except json.JSONDecodeError as exc:
            raise PredictionArtifactError("invalid prediction manifest JSON") from exc
        self.manifest = _validate_manifest(value)
        self.parquet_path = self.path / _DATA_FILE
        _verify_data_file(self.parquet_path, self.manifest["data_file"])
        _, parquet = _arrow_modules()
        try:
            table = parquet.read_table(str(self.parquet_path), memory_map=True)
        except (OSError, ValueError) as exc:
            raise PredictionArtifactError(f"unable to read prediction Parquet file: {exc}") from exc
        _validate_table_contract(table, self.manifest)

    @property
    def record_count(self) -> int:
        return cast(int, self.manifest["record_count"])

    def records(self) -> tuple[PredictionRecord, ...]:
        """Materialize logical records for evaluator and audit code."""
        _, parquet = _arrow_modules()
        table = parquet.read_table(str(self.parquet_path), memory_map=True)
        columns = cast(dict[str, list[object]], table.to_pydict())
        records: list[PredictionRecord] = []
        for index in range(self.record_count):
            asset_id = columns["asset_id"][index]
            timestamp_ns = columns["timestamp_ns"][index]
            if not isinstance(asset_id, str) or not isinstance(timestamp_ns, int):
                raise PredictionArtifactError("invalid prediction identity columns")
            records.append(
                PredictionRecord(
                    asset_id=asset_id,
                    timestamp_ns=timestamp_ns,
                    target=_numeric(columns["target"][index], "target"),
                    expected_return=_optional_numeric(
                        columns["expected_return"][index], "expected_return"
                    ),
                    rank_score=_optional_numeric(columns["rank_score"][index], "rank_score"),
                    direction_probability=_optional_numeric(
                        columns["direction_probability"][index], "direction_probability"
                    ),
                    volatility=_optional_numeric(columns["volatility"][index], "volatility"),
                    uncertainty=_optional_numeric(columns["uncertainty"][index], "uncertainty"),
                    quantiles=_optional_numeric_tuple(columns["quantiles"][index]),
                )
            )
        return tuple(records)


def predict_records(
    model: TradingModel,
    batches: Iterable[TrainingBatch],
    *,
    target_name: str,
) -> tuple[PredictionRecord, ...]:
    """Run architecture-neutral inference and retain evaluator-required identity/targets."""
    if not target_name.strip():
        raise ValueError("target_name must not be blank")
    device = next(model.parameters()).device
    model.eval()
    records: list[PredictionRecord] = []
    with torch.inference_mode():
        for original_batch in batches:
            batch = original_batch.to(device)
            target = batch.targets.get(target_name)
            if target is None:
                raise KeyError(f"batch does not provide target {target_name!r}")
            output = cast(ModelOutput, model(batch))
            output.validate(batch.batch_size)
            target_values = _scalar_tensor_values(target, batch.batch_size, target_name)
            scalar_heads = {
                "expected_return": _optional_scalar_tensor_values(
                    output.expected_return, batch.batch_size, "expected_return"
                ),
                "rank_score": _optional_scalar_tensor_values(
                    output.rank_score, batch.batch_size, "rank_score"
                ),
                "direction_probability": _optional_scalar_tensor_values(
                    output.direction_probability,
                    batch.batch_size,
                    "direction_probability",
                ),
                "volatility": _optional_scalar_tensor_values(
                    output.volatility, batch.batch_size, "volatility"
                ),
                "uncertainty": _optional_scalar_tensor_values(
                    output.uncertainty, batch.batch_size, "uncertainty"
                ),
            }
            quantiles = _quantile_tensor_values(output.quantiles, batch.batch_size)
            timestamps = batch.timestamps_ns.detach().cpu().tolist()
            for index in range(batch.batch_size):
                records.append(
                    PredictionRecord(
                        asset_id=batch.asset_ids[index],
                        timestamp_ns=int(timestamps[index]),
                        target=target_values[index],
                        expected_return=_head_value(scalar_heads["expected_return"], index),
                        rank_score=_head_value(scalar_heads["rank_score"], index),
                        direction_probability=_head_value(
                            scalar_heads["direction_probability"], index
                        ),
                        volatility=_head_value(scalar_heads["volatility"], index),
                        uncertainty=_head_value(scalar_heads["uncertainty"], index),
                        quantiles=_head_value(quantiles, index),
                    )
                )
    return tuple(records)


def write_prediction_artifact(
    records: Iterable[PredictionRecord],
    destination: str | Path,
    *,
    dataset_id: str,
    split_id: str,
    model_config_hash: str,
    checkpoint_id: str,
    target_name: str,
) -> PredictionWriteResult:
    """Atomically publish a checksummed Parquet/Zstd prediction artifact."""
    rows = tuple(sorted(records, key=lambda row: (row.timestamp_ns, row.asset_id)))
    if not rows:
        raise PredictionArtifactError("at least one prediction record is required")
    identifiers = (dataset_id, split_id, model_config_hash, checkpoint_id, target_name)
    if any(not value.strip() for value in identifiers):
        raise PredictionArtifactError("prediction artifact identity fields must not be blank")
    identities = [(row.asset_id, row.timestamp_ns) for row in rows]
    if len(set(identities)) != len(identities):
        raise PredictionArtifactError("duplicate asset/timestamp predictions are not allowed")
    destination = Path(destination)
    if destination.exists():
        raise PredictionArtifactError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=str(destination.parent))
    )
    try:
        data_path = temporary / _DATA_FILE
        _write_parquet(
            rows,
            data_path,
            dataset_id=dataset_id,
            split_id=split_id,
            model_config_hash=model_config_hash,
            checkpoint_id=checkpoint_id,
            target_name=target_name,
        )
        fsync_file(data_path)
        data_sha256 = _sha256_file(data_path)
        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "format": _FORMAT,
            "record_count": len(rows),
            "dataset_id": dataset_id,
            "split_id": split_id,
            "model_config_hash": model_config_hash,
            "checkpoint_id": checkpoint_id,
            "target_name": target_name,
            "data_file": {
                "name": _DATA_FILE,
                "size": data_path.stat().st_size,
                "sha256": data_sha256,
                "compression": "zstd",
            },
        }
        manifest_bytes = _canonical_json_bytes(manifest)
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        (temporary / _MANIFEST_FILE).write_bytes(manifest_bytes)
        (temporary / _MANIFEST_SHA256_FILE).write_text(
            f"{manifest_sha256}\n",
            encoding="ascii",
        )
        fsync_file(temporary / _MANIFEST_FILE)
        fsync_file(temporary / _MANIFEST_SHA256_FILE)
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return PredictionWriteResult(
        path=destination,
        record_count=len(rows),
        data_sha256=data_sha256,
        manifest_sha256=manifest_sha256,
    )


def _write_parquet(
    rows: tuple[PredictionRecord, ...],
    path: Path,
    *,
    dataset_id: str,
    split_id: str,
    model_config_hash: str,
    checkpoint_id: str,
    target_name: str,
) -> None:
    arrow, parquet = _arrow_modules()
    fields = [
        arrow.field("asset_id", arrow.string(), nullable=False),
        arrow.field("timestamp_ns", arrow.int64(), nullable=False),
        arrow.field("target", arrow.float32(), nullable=False),
        arrow.field("expected_return", arrow.float32(), nullable=True),
        arrow.field("rank_score", arrow.float32(), nullable=True),
        arrow.field("direction_probability", arrow.float32(), nullable=True),
        arrow.field("volatility", arrow.float32(), nullable=True),
        arrow.field("uncertainty", arrow.float32(), nullable=True),
        arrow.field("quantiles", arrow.list_(arrow.float32()), nullable=True),
    ]
    metadata = {
        b"trading_bot.schema_version": str(_SCHEMA_VERSION).encode("ascii"),
        b"trading_bot.format": _FORMAT.encode("ascii"),
        b"trading_bot.dataset_id": dataset_id.encode("utf-8"),
        b"trading_bot.split_id": split_id.encode("utf-8"),
        b"trading_bot.model_config_hash": model_config_hash.encode("ascii"),
        b"trading_bot.checkpoint_id": checkpoint_id.encode("utf-8"),
        b"trading_bot.target_name": target_name.encode("utf-8"),
    }
    schema = arrow.schema(fields, metadata=metadata)
    columns = [
        arrow.array([row.asset_id for row in rows], type=arrow.string()),
        arrow.array([row.timestamp_ns for row in rows], type=arrow.int64()),
        arrow.array([row.target for row in rows], type=arrow.float32()),
        arrow.array([row.expected_return for row in rows], type=arrow.float32()),
        arrow.array([row.rank_score for row in rows], type=arrow.float32()),
        arrow.array([row.direction_probability for row in rows], type=arrow.float32()),
        arrow.array([row.volatility for row in rows], type=arrow.float32()),
        arrow.array([row.uncertainty for row in rows], type=arrow.float32()),
        arrow.array([row.quantiles for row in rows], type=arrow.list_(arrow.float32())),
    ]
    table = arrow.Table.from_arrays(columns, schema=schema)
    parquet.write_table(
        table,
        str(path),
        version="2.6",
        compression="zstd",
        use_dictionary=["asset_id"],
        write_statistics=True,
        write_page_checksum=True,
    )


def _validate_manifest(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PredictionArtifactError("prediction manifest must be a JSON object")
    manifest = cast(dict[str, Any], value)
    if manifest.get("schema_version") != _SCHEMA_VERSION or manifest.get("format") != _FORMAT:
        raise PredictionArtifactError("unsupported prediction artifact schema")
    count = manifest.get("record_count")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise PredictionArtifactError("prediction record_count must be a positive integer")
    for key in ("dataset_id", "split_id", "model_config_hash", "checkpoint_id", "target_name"):
        item = manifest.get(key)
        if not isinstance(item, str) or not item.strip():
            raise PredictionArtifactError(f"prediction manifest field {key} is invalid")
    return manifest


def _validate_table_contract(table: Any, manifest: dict[str, Any]) -> None:
    expected_columns = [
        "asset_id",
        "timestamp_ns",
        "target",
        "expected_return",
        "rank_score",
        "direction_probability",
        "volatility",
        "uncertainty",
        "quantiles",
    ]
    if (
        list(table.column_names) != expected_columns
        or int(table.num_rows) != manifest["record_count"]
    ):
        raise PredictionArtifactError("prediction Parquet shape/schema does not match manifest")
    metadata = table.schema.metadata or {}
    expected_metadata = {
        b"trading_bot.schema_version": str(_SCHEMA_VERSION).encode("ascii"),
        b"trading_bot.format": _FORMAT.encode("ascii"),
        b"trading_bot.dataset_id": str(manifest["dataset_id"]).encode("utf-8"),
        b"trading_bot.split_id": str(manifest["split_id"]).encode("utf-8"),
        b"trading_bot.model_config_hash": str(manifest["model_config_hash"]).encode("ascii"),
        b"trading_bot.checkpoint_id": str(manifest["checkpoint_id"]).encode("utf-8"),
        b"trading_bot.target_name": str(manifest["target_name"]).encode("utf-8"),
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise PredictionArtifactError(
                f"prediction semantic metadata mismatch for {key.decode()}"
            )


def _read_verified_manifest(path: Path) -> bytes:
    try:
        manifest_bytes = (path / _MANIFEST_FILE).read_bytes()
        expected = (path / _MANIFEST_SHA256_FILE).read_text(encoding="ascii").strip().lower()
    except (OSError, UnicodeError) as exc:
        raise PredictionArtifactError("invalid prediction manifest integrity files") from exc
    _require_sha256(expected, "manifest.sha256")
    if hashlib.sha256(manifest_bytes).hexdigest() != expected:
        raise PredictionArtifactError("prediction manifest checksum mismatch")
    return manifest_bytes


def _verify_data_file(path: Path, value: object) -> None:
    if not isinstance(value, dict):
        raise PredictionArtifactError("prediction data_file manifest record is invalid")
    size = value.get("size")
    checksum = value.get("sha256")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise PredictionArtifactError("prediction data_file size is invalid")
    if not isinstance(checksum, str):
        raise PredictionArtifactError("prediction data_file checksum is invalid")
    _require_sha256(checksum, "data_file.sha256")
    if not path.is_file() or path.stat().st_size != size:
        raise PredictionArtifactError("prediction Parquet file size mismatch")
    if _sha256_file(path) != checksum.lower():
        raise PredictionArtifactError("prediction Parquet file checksum mismatch")


def _scalar_tensor_values(tensor: Tensor, batch_size: int, name: str) -> list[float]:
    value = tensor.detach().float().cpu()
    if value.shape == (batch_size, 1):
        value = value.squeeze(-1)
    if value.shape != (batch_size,):
        raise PredictionArtifactError(f"prediction head {name} must be scalar per sample")
    return [float(item) for item in value.tolist()]


def _optional_scalar_tensor_values(
    tensor: Tensor | None,
    batch_size: int,
    name: str,
) -> list[float] | None:
    if tensor is None:
        return None
    return _scalar_tensor_values(tensor, batch_size, name)


def _quantile_tensor_values(
    tensor: Tensor | None,
    batch_size: int,
) -> list[tuple[float, ...]] | None:
    if tensor is None:
        return None
    value = tensor.detach().float().cpu()
    if value.ndim != 2 or int(value.shape[0]) != batch_size:
        raise PredictionArtifactError("quantile predictions must have shape [batch, quantile]")
    return [tuple(float(item) for item in row) for row in value.tolist()]


def _head_value[T](values: list[T] | None, index: int) -> T | None:
    return None if values is None else values[index]


def _numeric(value: object, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise PredictionArtifactError(f"prediction column {name} contains a non-numeric value")
    number = float(value)
    if not math.isfinite(number):
        raise PredictionArtifactError(f"prediction column {name} contains NaN/Inf")
    return number


def _optional_numeric(value: object, name: str) -> float | None:
    return None if value is None else _numeric(value, name)


def _optional_numeric_tuple(value: object) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise PredictionArtifactError("prediction quantiles contain an invalid value")
    return tuple(_numeric(item, "quantiles") for item in value)


def _require_sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise PredictionArtifactError(f"prediction {field_name} is not a valid SHA-256")


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


def _arrow_modules() -> tuple[Any, Any]:
    try:
        arrow = import_module("pyarrow")
        parquet = import_module("pyarrow.parquet")
    except ModuleNotFoundError as exc:
        raise PredictionArtifactError("PyArrow is required for prediction artifacts") from exc
    return arrow, parquet
