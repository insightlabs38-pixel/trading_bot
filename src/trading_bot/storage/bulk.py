"""Resumable checksum-aware bulk transfers over the common storage protocol."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from trading_bot.storage.base import (
    ChecksumMismatchError,
    ObjectNotFoundError,
    StorageBackend,
    StorageError,
    fsync_file,
    normalize_storage_key,
    sha256_file,
    temporary_local_path,
)


class BulkTransferError(StorageError):
    """Raised when a resumable bulk transfer cannot complete."""

    def __init__(self, message: str, *, stats: BulkTransferStats | None = None) -> None:
        super().__init__(message)
        self.stats = stats


@dataclass(frozen=True, slots=True)
class UploadItem:
    """One local file to publish under a storage object key."""

    source: Path
    key: str
    expected_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", Path(self.source))
        object.__setattr__(self, "key", normalize_storage_key(self.key))
        if self.expected_sha256 is not None:
            object.__setattr__(self, "expected_sha256", _normalize_sha256(self.expected_sha256))


@dataclass(frozen=True, slots=True)
class DownloadItem:
    """One storage object to restore to a local path."""

    key: str
    destination: Path
    expected_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", normalize_storage_key(self.key))
        object.__setattr__(self, "destination", Path(self.destination))
        if self.expected_sha256 is not None:
            object.__setattr__(self, "expected_sha256", _normalize_sha256(self.expected_sha256))


@dataclass(frozen=True, slots=True)
class BulkTransferStats:
    """Auditable counters and measured throughput for one bulk transfer invocation."""

    direction: Literal["upload", "download"]
    total_items: int
    transferred_items: int
    skipped_items: int
    failed_items: int
    bytes_transferred: int
    elapsed_seconds: float
    started_at_utc: str
    finished_at_utc: str

    @property
    def throughput_mib_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.bytes_transferred / (1024 * 1024) / self.elapsed_seconds

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["throughput_mib_per_second"] = self.throughput_mib_per_second
        return payload


@dataclass(frozen=True, slots=True)
class _PreparedUpload:
    source: Path
    key: str
    size: int
    checksum: str


@dataclass(frozen=True, slots=True)
class _PreparedDownload:
    key: str
    destination: Path
    size: int
    checksum: str | None


class BulkTransferManager:
    """Backend-native equivalent to bulk copy tools with durable file-level resume."""

    def __init__(self, backend: StorageBackend) -> None:
        self.backend = backend

    def upload(
        self,
        items: Iterable[UploadItem],
        *,
        journal_path: str | Path,
        stats_path: str | Path | None = None,
    ) -> BulkTransferStats:
        prepared = tuple(self._prepare_upload(item) for item in items)
        plan_hash = _plan_hash(
            "upload",
            [
                {
                    "source": str(item.source.resolve()),
                    "key": item.key,
                    "size": item.size,
                    "checksum": item.checksum,
                }
                for item in prepared
            ],
        )
        journal = _load_or_create_journal(journal_path, "upload", plan_hash)
        start_wall = datetime.now(UTC)
        start = time.monotonic()
        transferred = skipped = bytes_transferred = 0
        try:
            for item in prepared:
                if self._upload_is_complete(item, journal):
                    skipped += 1
                    continue
                metadata = self.backend.upload(
                    item.source,
                    item.key,
                    expected_sha256=item.checksum,
                )
                if metadata.size != item.size or not self.backend.verify_checksum(
                    item.key, item.checksum
                ):
                    raise BulkTransferError(f"post-upload verification failed for {item.key}")
                journal["completed"][item.key] = {
                    "size": item.size,
                    "checksum": item.checksum,
                }
                _write_journal(journal_path, journal)
                transferred += 1
                bytes_transferred += item.size
        except Exception as exc:
            stats = _finish_stats(
                "upload",
                len(prepared),
                transferred,
                skipped,
                1,
                bytes_transferred,
                start,
                start_wall,
            )
            _record_stats(stats_path, journal_path, stats)
            if isinstance(exc, BulkTransferError):
                exc.stats = stats
                raise
            raise BulkTransferError(f"bulk upload failed: {exc}", stats=stats) from exc

        stats = _finish_stats(
            "upload",
            len(prepared),
            transferred,
            skipped,
            0,
            bytes_transferred,
            start,
            start_wall,
        )
        _record_stats(stats_path, journal_path, stats)
        return stats

    def download(
        self,
        items: Iterable[DownloadItem],
        *,
        journal_path: str | Path,
        stats_path: str | Path | None = None,
    ) -> BulkTransferStats:
        prepared = tuple(self._prepare_download(item) for item in items)
        plan_hash = _plan_hash(
            "download",
            [
                {
                    "key": item.key,
                    "destination": str(item.destination.resolve()),
                    "size": item.size,
                    "checksum": item.checksum,
                }
                for item in prepared
            ],
        )
        journal = _load_or_create_journal(journal_path, "download", plan_hash)
        start_wall = datetime.now(UTC)
        start = time.monotonic()
        transferred = skipped = bytes_transferred = 0
        try:
            for item in prepared:
                identity = str(item.destination.resolve())
                if self._download_is_complete(item, identity, journal):
                    skipped += 1
                    continue
                result = self.backend.download(
                    item.key,
                    item.destination,
                    expected_sha256=item.checksum,
                )
                checksum = sha256_file(result)
                if item.checksum is not None and checksum != item.checksum:
                    raise ChecksumMismatchError(
                        f"SHA-256 mismatch after restoring {item.key} to {item.destination}"
                    )
                if result.stat().st_size != item.size:
                    raise BulkTransferError(f"post-download size mismatch for {item.key}")
                journal["completed"][identity] = {
                    "size": item.size,
                    "checksum": checksum,
                    "key": item.key,
                }
                _write_journal(journal_path, journal)
                transferred += 1
                bytes_transferred += item.size
        except Exception as exc:
            stats = _finish_stats(
                "download",
                len(prepared),
                transferred,
                skipped,
                1,
                bytes_transferred,
                start,
                start_wall,
            )
            _record_stats(stats_path, journal_path, stats)
            if isinstance(exc, BulkTransferError):
                exc.stats = stats
                raise
            raise BulkTransferError(f"bulk download failed: {exc}", stats=stats) from exc

        stats = _finish_stats(
            "download",
            len(prepared),
            transferred,
            skipped,
            0,
            bytes_transferred,
            start,
            start_wall,
        )
        _record_stats(stats_path, journal_path, stats)
        return stats

    def _prepare_upload(self, item: UploadItem) -> _PreparedUpload:
        if not item.source.is_file():
            raise ObjectNotFoundError(f"bulk upload source does not exist: {item.source}")
        checksum = sha256_file(item.source)
        if item.expected_sha256 is not None and checksum != item.expected_sha256:
            raise ChecksumMismatchError(
                f"SHA-256 mismatch for bulk upload source {item.source}: "
                f"expected {item.expected_sha256}, got {checksum}"
            )
        return _PreparedUpload(item.source, item.key, item.source.stat().st_size, checksum)

    def _prepare_download(self, item: DownloadItem) -> _PreparedDownload:
        metadata = self.backend.head(item.key)
        checksum = item.expected_sha256 or metadata.checksum_sha256
        return _PreparedDownload(
            item.key,
            item.destination,
            metadata.size,
            None if checksum is None else _normalize_sha256(checksum),
        )

    def _upload_is_complete(self, item: _PreparedUpload, journal: dict[str, Any]) -> bool:
        record = journal["completed"].get(item.key)
        record_matches = record == {"size": item.size, "checksum": item.checksum}
        if not record_matches and not self.backend.exists(item.key):
            return False
        try:
            metadata = self.backend.head(item.key)
            verified = metadata.size == item.size and self.backend.verify_checksum(
                item.key, item.checksum
            )
        except StorageError:
            return False
        if verified and not record_matches:
            journal["completed"][item.key] = {
                "size": item.size,
                "checksum": item.checksum,
            }
        return verified

    @staticmethod
    def _download_is_complete(
        item: _PreparedDownload,
        identity: str,
        journal: dict[str, Any],
    ) -> bool:
        if not item.destination.is_file() or item.destination.stat().st_size != item.size:
            return False
        checksum = sha256_file(item.destination)
        if item.checksum is not None and checksum != item.checksum:
            return False
        record = journal["completed"].get(identity)
        if record is not None and record.get("key") != item.key:
            return False
        journal["completed"][identity] = {
            "size": item.size,
            "checksum": checksum,
            "key": item.key,
        }
        return True


def _normalize_sha256(value: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError("SHA-256 values must be 64 hexadecimal characters")
    return normalized


def _plan_hash(direction: str, items: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        {"direction": direction, "items": items},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_or_create_journal(
    path: str | Path,
    direction: Literal["upload", "download"],
    plan_hash: str,
) -> dict[str, Any]:
    journal_path = Path(path)
    if journal_path.exists():
        try:
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BulkTransferError(f"invalid transfer journal {journal_path}: {exc}") from exc
        if journal.get("schema_version") != 1:
            raise BulkTransferError(f"unsupported transfer journal version: {journal_path}")
        if journal.get("direction") != direction or journal.get("plan_sha256") != plan_hash:
            raise BulkTransferError(
                "transfer journal does not match the requested direction/plan; "
                "use a new journal path"
            )
        if not isinstance(journal.get("completed"), dict):
            raise BulkTransferError(f"invalid completed map in transfer journal {journal_path}")
        return journal
    return {
        "schema_version": 1,
        "direction": direction,
        "plan_sha256": plan_hash,
        "completed": {},
    }


def _write_journal(path: str | Path, journal: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(journal)
    payload["updated_at_utc"] = datetime.now(UTC).isoformat()
    temporary = temporary_local_path(destination)
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        fsync_file(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _finish_stats(
    direction: Literal["upload", "download"],
    total: int,
    transferred: int,
    skipped: int,
    failed: int,
    bytes_transferred: int,
    start: float,
    start_wall: datetime,
) -> BulkTransferStats:
    finished = datetime.now(UTC)
    return BulkTransferStats(
        direction=direction,
        total_items=total,
        transferred_items=transferred,
        skipped_items=skipped,
        failed_items=failed,
        bytes_transferred=bytes_transferred,
        elapsed_seconds=max(0.0, time.monotonic() - start),
        started_at_utc=start_wall.isoformat(),
        finished_at_utc=finished.isoformat(),
    )


def _record_stats(
    stats_path: str | Path | None,
    journal_path: str | Path,
    stats: BulkTransferStats,
) -> None:
    destination = (
        Path(stats_path)
        if stats_path is not None
        else Path(journal_path).with_suffix(Path(journal_path).suffix + ".stats.jsonl")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(stats.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
