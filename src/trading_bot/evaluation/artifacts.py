"""Independent reader for durable saved predictions produced by Phase 5."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, cast

from trading_bot.evaluation.contracts import PredictionPoint

_DATA_FILE = "predictions.parquet"
_MANIFEST_FILE = "manifest.json"
_MANIFEST_SHA256_FILE = "manifest.sha256"
_FORMAT = "phase5_prediction_parquet"
_SCHEMA_VERSION = 1

ScoreField = Literal["expected_return", "rank_score"]


class SavedPredictionError(RuntimeError):
    """Raised when a saved prediction artifact violates the evaluator contract."""


@dataclass(frozen=True, slots=True)
class SavedPrediction:
    asset_id: str
    timestamp_ns: int
    target: float
    expected_return: float | None
    rank_score: float | None
    direction_probability: float | None
    volatility: float | None
    uncertainty: float | None
    quantiles: tuple[float, ...] | None


@dataclass(frozen=True, slots=True)
class PredictionDataset:
    dataset_id: str
    split_id: str
    model_config_hash: str
    checkpoint_id: str
    target_name: str
    records: tuple[SavedPrediction, ...]

    def prediction_points(
        self,
        *,
        score_field: ScoreField = "rank_score",
        regime_by_identity: Mapping[tuple[str, int], str] | None = None,
        sector_by_asset: Mapping[str, str] | None = None,
        seed: int | None = None,
    ) -> tuple[PredictionPoint, ...]:
        """Convert saved predictions into the canonical pure-prediction metric input."""
        result: list[PredictionPoint] = []
        for record in self.records:
            score = getattr(record, score_field)
            if score is None:
                raise SavedPredictionError(
                    f"saved prediction does not provide required score field {score_field}"
                )
            identity = (record.asset_id, record.timestamp_ns)
            result.append(
                PredictionPoint(
                    asset_id=record.asset_id,
                    timestamp_ns=record.timestamp_ns,
                    target=record.target,
                    score=score,
                    fold_id=self.split_id,
                    regime=(
                        regime_by_identity.get(identity, "unlabeled")
                        if regime_by_identity is not None
                        else "unlabeled"
                    ),
                    horizon=self.target_name,
                    sector=(
                        sector_by_asset.get(record.asset_id, "unlabeled")
                        if sector_by_asset is not None
                        else "unlabeled"
                    ),
                    seed=seed,
                )
            )
        return tuple(result)


def read_prediction_artifact(path: str | Path) -> PredictionDataset:
    """Read and independently validate the Phase 5 prediction artifact."""
    root = Path(path)
    manifest_bytes = _verified_manifest_bytes(root)
    try:
        raw_manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise SavedPredictionError("invalid prediction manifest JSON") from exc
    manifest = _validate_manifest(raw_manifest)
    data_path = root / _DATA_FILE
    _verify_data_file(data_path, manifest["data_file"])

    _, parquet = _arrow_modules()
    try:
        table = parquet.read_table(str(data_path), memory_map=True)
    except (OSError, ValueError) as exc:
        raise SavedPredictionError(f"unable to read prediction Parquet file: {exc}") from exc
    _validate_table(table, manifest, data_path)
    columns = cast(dict[str, list[object]], table.to_pydict())
    count = cast(int, manifest["record_count"])
    records: list[SavedPrediction] = []
    identities: set[tuple[str, int]] = set()
    for index in range(count):
        asset_id = columns["asset_id"][index]
        timestamp_ns = columns["timestamp_ns"][index]
        if (
            not isinstance(asset_id, str)
            or not asset_id.strip()
            or not isinstance(timestamp_ns, int)
        ):
            raise SavedPredictionError("invalid prediction identity columns")
        identity = (asset_id, timestamp_ns)
        if identity in identities:
            raise SavedPredictionError("duplicate saved prediction identity")
        identities.add(identity)
        records.append(
            SavedPrediction(
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

    return PredictionDataset(
        dataset_id=cast(str, manifest["dataset_id"]),
        split_id=cast(str, manifest["split_id"]),
        model_config_hash=cast(str, manifest["model_config_hash"]),
        checkpoint_id=cast(str, manifest["checkpoint_id"]),
        target_name=cast(str, manifest["target_name"]),
        records=tuple(records),
    )


def _verified_manifest_bytes(root: Path) -> bytes:
    try:
        manifest_bytes = (root / _MANIFEST_FILE).read_bytes()
        expected = (root / _MANIFEST_SHA256_FILE).read_text(encoding="ascii").strip().lower()
    except OSError as exc:
        raise SavedPredictionError("prediction manifest files are missing") from exc
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise SavedPredictionError("invalid prediction manifest checksum")
    if hashlib.sha256(manifest_bytes).hexdigest() != expected:
        raise SavedPredictionError("prediction manifest checksum mismatch")
    return manifest_bytes


def _validate_manifest(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SavedPredictionError("prediction manifest must be a JSON object")
    manifest = cast(dict[str, Any], value)
    if manifest.get("schema_version") != _SCHEMA_VERSION or manifest.get("format") != _FORMAT:
        raise SavedPredictionError("unsupported prediction artifact schema")
    count = manifest.get("record_count")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise SavedPredictionError("prediction record_count must be a positive integer")
    for key in ("dataset_id", "split_id", "model_config_hash", "checkpoint_id", "target_name"):
        item = manifest.get(key)
        if not isinstance(item, str) or not item.strip():
            raise SavedPredictionError(f"prediction manifest field {key} is invalid")
    data_file = manifest.get("data_file")
    if not isinstance(data_file, dict):
        raise SavedPredictionError("prediction manifest data_file is invalid")
    return manifest


def _verify_data_file(path: Path, value: object) -> None:
    if not isinstance(value, dict):
        raise SavedPredictionError("prediction data_file manifest is invalid")
    record = cast(dict[str, object], value)
    name = record.get("name")
    size = record.get("size")
    digest = record.get("sha256")
    compression = record.get("compression")
    if (
        name != _DATA_FILE
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or compression != "zstd"
    ):
        raise SavedPredictionError("prediction data_file manifest fields are invalid")
    try:
        actual_size = path.stat().st_size
    except OSError as exc:
        raise SavedPredictionError("prediction data file is missing") from exc
    if actual_size != size:
        raise SavedPredictionError("prediction data file size mismatch")
    if _sha256_file(path) != digest:
        raise SavedPredictionError("prediction data file checksum mismatch")


def _validate_table(table: Any, manifest: dict[str, Any], path: Path) -> None:
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
        raise SavedPredictionError("prediction Parquet shape/schema does not match manifest")
    metadata = table.schema.metadata or {}
    expected_metadata = {
        b"trading_bot.schema_version": str(_SCHEMA_VERSION).encode("ascii"),
        b"trading_bot.format": _FORMAT.encode("ascii"),
        b"trading_bot.dataset_id": cast(str, manifest["dataset_id"]).encode("utf-8"),
        b"trading_bot.split_id": cast(str, manifest["split_id"]).encode("utf-8"),
        b"trading_bot.model_config_hash": cast(str, manifest["model_config_hash"]).encode("ascii"),
        b"trading_bot.checkpoint_id": cast(str, manifest["checkpoint_id"]).encode("utf-8"),
        b"trading_bot.target_name": cast(str, manifest["target_name"]).encode("utf-8"),
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise SavedPredictionError("prediction Parquet metadata does not match manifest")
    _, parquet = _arrow_modules()
    parquet_file = parquet.ParquetFile(str(path))
    for row_group_index in range(parquet_file.metadata.num_row_groups):
        row_group = parquet_file.metadata.row_group(row_group_index)
        for column_index in range(row_group.num_columns):
            if row_group.column(column_index).compression != "ZSTD":
                raise SavedPredictionError("prediction Parquet columns must use ZSTD compression")


def _numeric(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SavedPredictionError(f"prediction column {name} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SavedPredictionError(f"prediction column {name} contains non-finite values")
    return result


def _optional_numeric(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _numeric(value, name)


def _optional_numeric_tuple(value: object) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise SavedPredictionError("prediction quantiles column is invalid")
    return tuple(_numeric(item, "quantiles") for item in value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _arrow_modules() -> tuple[Any, Any]:
    try:
        return import_module("pyarrow"), import_module("pyarrow.parquet")
    except ImportError as exc:
        raise SavedPredictionError("PyArrow is required to read prediction artifacts") from exc
