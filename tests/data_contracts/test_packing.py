"""Tests for deterministic memory-mapped reference training packs."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from trading_bot.data.packing import (
    PackedDataset,
    PackingError,
    TrainingSample,
    benchmark_loader,
    pack_training_data,
)


START = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)


def sample(index: int, security_id: str = "sec-a") -> TrainingSample:
    return TrainingSample(
        security_id=security_id,
        timestamp=START + timedelta(minutes=index),
        features=(float(index), float(index + 1)),
        targets=(float(index) / 10,),
    )


def test_pack_preserves_features_targets_timestamps_and_asset_ids(tmp_path: Path) -> None:
    destination = tmp_path / "pack"
    result = pack_training_data(
        [sample(1, "sec-b"), sample(0, "sec-a")],
        destination,
        feature_names=("f1", "f2"),
        target_names=("return_15",),
        dataset_version="dataset-v1",
        split_version="split-v1",
    )
    dataset = PackedDataset(destination)
    assert isinstance(dataset.features, np.memmap)
    assert dataset.features.shape == (2, 2)
    assert dataset.targets.shape == (2, 1)
    assert list(dataset.asset_ids) == ["sec-a", "sec-b"]
    assert dataset.timestamps_ns[0] < dataset.timestamps_ns[1]
    assert dataset.dataset_sha256 == result.dataset_sha256
    assert (destination / "metadata.sha256").read_text(encoding="ascii").strip() == (
        result.dataset_sha256
    )


def test_metadata_is_deterministic_for_equivalent_inputs(tmp_path: Path) -> None:
    kwargs = {
        "feature_names": ("f1", "f2"),
        "target_names": ("target",),
        "dataset_version": "dataset-v1",
        "split_version": "split-v1",
    }
    first = pack_training_data([sample(0), sample(1)], tmp_path / "first", **kwargs)
    second = pack_training_data([sample(1), sample(0)], tmp_path / "second", **kwargs)
    assert first.dataset_sha256 == second.dataset_sha256
    first_metadata = json.loads((first.path / "metadata.json").read_text())
    second_metadata = json.loads((second.path / "metadata.json").read_text())
    assert first_metadata == second_metadata
    assert (first.path / "metadata.sha256").read_bytes() == (
        second.path / "metadata.sha256"
    ).read_bytes()


def test_integrity_check_rejects_tampered_array(tmp_path: Path) -> None:
    destination = tmp_path / "pack"
    pack_training_data(
        [sample(0)],
        destination,
        feature_names=("f1", "f2"),
        target_names=("target",),
        dataset_version="dataset-v1",
        split_version="split-v1",
    )
    path = destination / "features.npy"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(PackingError, match="checksum mismatch"):
        PackedDataset(destination)


def test_integrity_check_rejects_tampered_metadata(tmp_path: Path) -> None:
    destination = tmp_path / "pack"
    pack_training_data(
        [sample(0)],
        destination,
        feature_names=("f1", "f2"),
        target_names=("target",),
        dataset_version="dataset-v1",
        split_version="split-v1",
    )
    metadata_path = destination / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["dataset_version"] = "tampered"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(PackingError, match="metadata checksum mismatch"):
        PackedDataset(destination)


def test_integrity_check_rejects_malformed_metadata_checksum(tmp_path: Path) -> None:
    destination = tmp_path / "pack"
    pack_training_data(
        [sample(0)],
        destination,
        feature_names=("f1", "f2"),
        target_names=("target",),
        dataset_version="dataset-v1",
        split_version="split-v1",
    )
    (destination / "metadata.sha256").write_text("not-a-checksum\n", encoding="ascii")
    with pytest.raises(PackingError, match="64-character hexadecimal SHA-256"):
        PackedDataset(destination)


def test_integrity_sidecar_matches_metadata_bytes(tmp_path: Path) -> None:
    destination = tmp_path / "pack"
    result = pack_training_data(
        [sample(0)],
        destination,
        feature_names=("f1", "f2"),
        target_names=("target",),
        dataset_version="dataset-v1",
        split_version="split-v1",
    )
    expected = hashlib.sha256((destination / "metadata.json").read_bytes()).hexdigest()
    assert expected == result.dataset_sha256


def test_iter_batches_and_loader_benchmark_touch_complete_dataset(tmp_path: Path) -> None:
    destination = tmp_path / "pack"
    pack_training_data(
        [sample(index) for index in range(10)],
        destination,
        feature_names=("f1", "f2"),
        target_names=("target",),
        dataset_version="dataset-v1",
        split_version="split-v1",
    )
    dataset = PackedDataset(destination)
    batches = list(dataset.iter_batches(4))
    assert [len(batch[0]) for batch in batches] == [4, 4, 2]
    benchmark = benchmark_loader(dataset, batch_size=3)
    assert benchmark.sample_count == 10
    assert benchmark.bytes_read > 0
    assert benchmark.samples_per_second > 0
    assert benchmark.mib_per_second > 0


def test_width_mismatch_duplicate_names_and_existing_destination_are_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(PackingError, match="feature width"):
        pack_training_data(
            [sample(0)],
            tmp_path / "bad",
            feature_names=("only-one",),
            target_names=("target",),
            dataset_version="v1",
            split_version="s1",
        )
    with pytest.raises(PackingError, match="unique"):
        pack_training_data(
            [sample(0)],
            tmp_path / "duplicates",
            feature_names=("f", "f"),
            target_names=("target",),
            dataset_version="v1",
            split_version="s1",
        )
    destination = tmp_path / "existing"
    pack_training_data(
        [sample(0)],
        destination,
        feature_names=("f1", "f2"),
        target_names=("target",),
        dataset_version="v1",
        split_version="s1",
    )
    with pytest.raises(PackingError, match="already exists"):
        pack_training_data(
            [sample(0)],
            destination,
            feature_names=("f1", "f2"),
            target_names=("target",),
            dataset_version="v1",
            split_version="s1",
        )
