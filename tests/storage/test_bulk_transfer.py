"""Tests for resumable backend-native bulk transfer jobs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trading_bot.storage import ChecksumMismatchError, LocalStorageBackend, sha256_file
from trading_bot.storage.bulk import (
    BulkTransferError,
    BulkTransferManager,
    DownloadItem,
    UploadItem,
)
from trading_bot.storage.manifests import publish_artifact_with_manifest, verify_artifact_manifest


class CountingBackend(LocalStorageBackend):
    def __init__(self, root: Path, *, fail_once_key: str | None = None) -> None:
        super().__init__(root)
        self.upload_counts: dict[str, int] = {}
        self.fail_once_key = fail_once_key
        self.failed = False

    def upload(
        self,
        source: str | Path,
        key: str,
        *,
        expected_sha256: str | None = None,
    ):
        self.upload_counts[key] = self.upload_counts.get(key, 0) + 1
        if key == self.fail_once_key and not self.failed:
            self.failed = True
            raise RuntimeError("injected transfer interruption")
        return super().upload(source, key, expected_sha256=expected_sha256)


def make_source(root: Path, name: str, payload: bytes) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_bulk_upload_records_throughput_and_journal(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path / "store")
    manager = BulkTransferManager(backend)
    a = make_source(tmp_path, "a.bin", b"a" * 100)
    b = make_source(tmp_path, "b.bin", b"b" * 200)
    journal = tmp_path / "upload.json"
    stats = manager.upload(
        [UploadItem(a, "data/a.bin"), UploadItem(b, "data/b.bin")],
        journal_path=journal,
    )
    assert stats.transferred_items == 2
    assert stats.bytes_transferred == 300
    assert stats.failed_items == 0
    assert stats.throughput_mib_per_second >= 0
    stored = json.loads(journal.read_text(encoding="utf-8"))
    assert set(stored["completed"]) == {"data/a.bin", "data/b.bin"}
    stats_records = journal.with_suffix(".json.stats.jsonl").read_text().splitlines()
    assert json.loads(stats_records[-1])["bytes_transferred"] == 300


def test_interrupted_upload_recovers_without_retransferring_completed_files(
    tmp_path: Path,
) -> None:
    backend = CountingBackend(tmp_path / "store", fail_once_key="data/b.bin")
    manager = BulkTransferManager(backend)
    a = make_source(tmp_path, "a.bin", b"first")
    b = make_source(tmp_path, "b.bin", b"second")
    journal = tmp_path / "resume.json"
    items = [UploadItem(a, "data/a.bin"), UploadItem(b, "data/b.bin")]
    with pytest.raises(BulkTransferError, match="interruption") as failure:
        manager.upload(items, journal_path=journal)
    assert failure.value.stats is not None
    assert failure.value.stats.transferred_items == 1
    assert backend.upload_counts["data/a.bin"] == 1
    recovered = manager.upload(items, journal_path=journal)
    assert recovered.skipped_items == 1
    assert recovered.transferred_items == 1
    assert backend.upload_counts["data/a.bin"] == 1
    assert backend.upload_counts["data/b.bin"] == 2
    assert backend.verify_checksum("data/a.bin", sha256_file(a))
    assert backend.verify_checksum("data/b.bin", sha256_file(b))


def test_existing_verified_object_is_adopted_when_journal_is_new(tmp_path: Path) -> None:
    backend = CountingBackend(tmp_path / "store")
    manager = BulkTransferManager(backend)
    source = make_source(tmp_path, "a.bin", b"existing")
    backend.upload(source, "data/a.bin")
    assert backend.upload_counts["data/a.bin"] == 1
    stats = manager.upload(
        [UploadItem(source, "data/a.bin")],
        journal_path=tmp_path / "new-journal.json",
    )
    assert stats.skipped_items == 1
    assert backend.upload_counts["data/a.bin"] == 1


def test_journal_rejects_a_different_transfer_plan(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path / "store")
    manager = BulkTransferManager(backend)
    first = make_source(tmp_path, "a.bin", b"a")
    second = make_source(tmp_path, "b.bin", b"b")
    journal = tmp_path / "journal.json"
    manager.upload([UploadItem(first, "a.bin")], journal_path=journal)
    with pytest.raises(BulkTransferError, match="does not match"):
        manager.upload([UploadItem(second, "b.bin")], journal_path=journal)


def test_bulk_download_restores_and_then_skips_verified_files(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path / "store")
    source = make_source(tmp_path, "source.bin", b"restore-me")
    checksum = sha256_file(source)
    backend.upload(source, "durable/source.bin", expected_sha256=checksum)
    destination = tmp_path / "restored" / "source.bin"
    manager = BulkTransferManager(backend)
    item = DownloadItem("durable/source.bin", destination, checksum)
    journal = tmp_path / "download.json"
    first = manager.download([item], journal_path=journal)
    assert first.transferred_items == 1
    assert destination.read_bytes() == b"restore-me"
    second = manager.download([item], journal_path=journal)
    assert second.skipped_items == 1
    assert second.bytes_transferred == 0


def test_bulk_upload_rejects_wrong_expected_checksum_before_transfer(tmp_path: Path) -> None:
    backend = CountingBackend(tmp_path / "store")
    manager = BulkTransferManager(backend)
    source = make_source(tmp_path, "source.bin", b"source")
    with pytest.raises(ChecksumMismatchError, match="SHA-256 mismatch"):
        manager.upload(
            [UploadItem(source, "source.bin", "0" * 64)],
            journal_path=tmp_path / "journal.json",
        )
    assert backend.upload_counts == {}


def test_phase2_local_gate_round_trip_with_manifest(tmp_path: Path) -> None:
    durable = LocalStorageBackend(tmp_path / "durable")
    source = make_source(tmp_path, "generated.bin", b"generated artifact")
    manifest = publish_artifact_with_manifest(
        durable,
        source,
        "artifacts/generated.bin",
        artifact_schema="test-artifact-v1",
        artifact_version="1",
        producer_git_sha="a" * 40,
        producer_config_sha256="b" * 64,
    )
    source.unlink()
    restored = tmp_path / "restored.bin"
    manager = BulkTransferManager(durable)
    stats = manager.download(
        [DownloadItem(manifest.artifact_key, restored, manifest.checksum)],
        journal_path=tmp_path / "restore.json",
    )
    assert stats.transferred_items == 1
    assert restored.read_bytes() == b"generated artifact"
    assert verify_artifact_manifest(durable, manifest) == manifest
