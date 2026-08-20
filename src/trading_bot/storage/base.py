"""Backend-independent storage contracts and safety helpers."""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol, TypeVar, runtime_checkable

_T = TypeVar("_T")
_TEMP_MARKER = ".trading-bot-tmp-"


class StorageError(RuntimeError):
    """Base error raised by storage backends."""


class ObjectNotFoundError(StorageError):
    """Raised when a requested storage object does not exist."""


class ChecksumMismatchError(StorageError):
    """Raised when transferred bytes do not match the expected SHA-256 digest."""


class UnsafeStorageKeyError(StorageError):
    """Raised when an object key can escape or ambiguously address the storage root."""


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded exponential retry settings for transient remote failures."""

    max_attempts: int = 4
    initial_delay_seconds: float = 0.25
    multiplier: float = 2.0
    max_delay_seconds: float = 4.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_delay_seconds < 0:
            raise ValueError("initial_delay_seconds must be non-negative")
        if self.multiplier < 1:
            raise ValueError("multiplier must be at least 1")
        if self.max_delay_seconds < self.initial_delay_seconds:
            raise ValueError("max_delay_seconds must not be less than initial_delay_seconds")


@dataclass(frozen=True, slots=True)
class TransferTimeoutPolicy:
    """Connection and socket-read timeouts for remote storage operations."""

    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.connect_timeout_seconds <= 0 or self.read_timeout_seconds <= 0:
            raise ValueError("storage timeouts must be positive")


@dataclass(frozen=True, slots=True)
class StorageObjectMetadata:
    """Portable object metadata returned by all storage backends."""

    key: str
    size: int
    checksum_sha256: str | None = None
    etag: str | None = None
    last_modified_epoch_seconds: float | None = None


@runtime_checkable
class StorageBackend(Protocol):
    """Common local/S3 storage surface used by later artifact code."""

    def list(self, prefix: str = "") -> list[StorageObjectMetadata]: ...

    def exists(self, key: str) -> bool: ...

    def upload(
        self,
        source: str | Path,
        key: str,
        *,
        expected_sha256: str | None = None,
    ) -> StorageObjectMetadata: ...

    def multipart_upload(
        self,
        source: str | Path,
        key: str,
        *,
        expected_sha256: str | None = None,
    ) -> StorageObjectMetadata: ...

    def download(
        self,
        key: str,
        destination: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> Path: ...

    def copy(
        self,
        source_key: str,
        destination_key: str,
        *,
        expected_sha256: str | None = None,
    ) -> StorageObjectMetadata: ...

    def delete(self, key: str) -> None: ...

    def head(self, key: str) -> StorageObjectMetadata: ...

    def verify_checksum(self, key: str, expected_sha256: str) -> bool: ...


def normalize_storage_key(key: str, *, allow_empty: bool = False) -> str:
    """Return one canonical relative POSIX object key or reject unsafe input."""
    if not isinstance(key, str):
        raise TypeError("storage key must be a string")
    candidate = key.replace("\\", "/").strip()
    if not candidate:
        if allow_empty:
            return ""
        raise UnsafeStorageKeyError("storage key must not be empty")
    path = PurePosixPath(candidate)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise UnsafeStorageKeyError(f"unsafe storage key: {key!r}")
    normalized = path.as_posix()
    if normalized.startswith("../") or "/../" in normalized:
        raise UnsafeStorageKeyError(f"unsafe storage key: {key!r}")
    return normalized


def temporary_storage_key(key: str) -> str:
    """Return an unpublished temporary sibling key used for atomic publication."""
    normalized = normalize_storage_key(key)
    path = PurePosixPath(normalized)
    token = uuid.uuid4().hex
    temporary_name = f"{path.name}{_TEMP_MARKER}{token}"
    return (path.parent / temporary_name).as_posix()


def is_temporary_storage_key(key: str) -> bool:
    """Identify temporary keys so normal listings never treat them as durable objects."""
    return _TEMP_MARKER in PurePosixPath(key).name


def temporary_local_path(destination: str | Path) -> Path:
    """Return a unique temporary sibling path for atomic local publication."""
    path = Path(destination)
    return path.with_name(f"{path.name}{_TEMP_MARKER}{uuid.uuid4().hex}")


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute a streaming SHA-256 digest for a local file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_chunks(chunks: Iterator[bytes]) -> str:
    """Compute SHA-256 for a byte iterator."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def require_checksum(actual: str, expected: str | None, *, context: str) -> None:
    """Raise a clear error when an expected checksum does not match."""
    if expected is not None and actual.lower() != expected.lower():
        raise ChecksumMismatchError(
            f"SHA-256 mismatch for {context}: expected {expected.lower()}, got {actual.lower()}"
        )


def retry_call[T](
    operation: Callable[[], T],
    *,
    policy: RetryPolicy,
    is_retryable: Callable[[BaseException], bool],
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run an operation under bounded exponential retry without hidden infinite loops."""
    delay = policy.initial_delay_seconds
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return operation()
        except BaseException as exc:
            if attempt >= policy.max_attempts or not is_retryable(exc):
                raise
            if delay > 0:
                sleep(delay)
            delay = min(delay * policy.multiplier, policy.max_delay_seconds)
    raise AssertionError("unreachable retry loop")


def fsync_file(path: Path) -> None:
    """Flush file contents before an atomic local rename publishes them."""
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    """Flush directory metadata after rename/unlink so publication survives a crash."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
