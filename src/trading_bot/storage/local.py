"""Atomic local-filesystem implementation of the common storage protocol."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from trading_bot.storage.base import (
    ObjectNotFoundError,
    StorageObjectMetadata,
    fsync_directory,
    fsync_file,
    is_temporary_storage_key,
    normalize_storage_key,
    require_checksum,
    sha256_file,
    temporary_local_path,
    temporary_storage_key,
)


class LocalStorageBackend:
    """Durable local storage rooted at one directory with path-escape protection."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        normalized = normalize_storage_key(key)
        candidate = (self.root / normalized).resolve()
        if self.root not in candidate.parents:
            raise ValueError(f"storage key escaped local root: {key!r}")
        return candidate

    def list(self, prefix: str = "") -> list[StorageObjectMetadata]:
        normalized_prefix = normalize_storage_key(prefix, allow_empty=True)
        objects: list[StorageObjectMetadata] = []
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            key = path.relative_to(self.root).as_posix()
            if is_temporary_storage_key(key) or not key.startswith(normalized_prefix):
                continue
            objects.append(self.head(key))
        return sorted(objects, key=lambda item: item.key)

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def upload(
        self,
        source: str | Path,
        key: str,
        *,
        expected_sha256: str | None = None,
    ) -> StorageObjectMetadata:
        source_path = Path(source)
        if not source_path.is_file():
            raise ObjectNotFoundError(f"local upload source does not exist: {source_path}")
        actual = sha256_file(source_path)
        require_checksum(actual, expected_sha256, context=str(source_path))
        destination = self._path(key)
        temporary = self._path(temporary_storage_key(key))
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(source_path, temporary)
            require_checksum(sha256_file(temporary), actual, context=str(temporary))
            fsync_file(temporary)
            os.replace(temporary, destination)
            fsync_directory(destination.parent)
        finally:
            temporary.unlink(missing_ok=True)
        return self.head(key)

    def multipart_upload(
        self,
        source: str | Path,
        key: str,
        *,
        expected_sha256: str | None = None,
    ) -> StorageObjectMetadata:
        # Local filesystems do not need S3 multipart semantics, but this preserves the common API
        # and performs a chunked atomic transfer suitable for large files.
        source_path = Path(source)
        if not source_path.is_file():
            raise ObjectNotFoundError(f"local upload source does not exist: {source_path}")
        actual = sha256_file(source_path)
        require_checksum(actual, expected_sha256, context=str(source_path))
        destination = self._path(key)
        temporary = self._path(temporary_storage_key(key))
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with source_path.open("rb") as source_handle, temporary.open("wb") as target_handle:
                while chunk := source_handle.read(8 * 1024 * 1024):
                    target_handle.write(chunk)
                target_handle.flush()
                os.fsync(target_handle.fileno())
            require_checksum(sha256_file(temporary), actual, context=str(temporary))
            os.replace(temporary, destination)
            fsync_directory(destination.parent)
        finally:
            temporary.unlink(missing_ok=True)
        return self.head(key)

    def download(
        self,
        key: str,
        destination: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> Path:
        source = self._path(key)
        if not source.is_file():
            raise ObjectNotFoundError(f"storage object does not exist: {key}")
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = temporary_local_path(destination_path)
        try:
            shutil.copyfile(source, temporary)
            actual = sha256_file(temporary)
            require_checksum(actual, expected_sha256, context=key)
            fsync_file(temporary)
            os.replace(temporary, destination_path)
            fsync_directory(destination_path.parent)
        finally:
            temporary.unlink(missing_ok=True)
        return destination_path

    def copy(
        self,
        source_key: str,
        destination_key: str,
        *,
        expected_sha256: str | None = None,
    ) -> StorageObjectMetadata:
        source = self._path(source_key)
        if not source.is_file():
            raise ObjectNotFoundError(f"storage object does not exist: {source_key}")
        actual = sha256_file(source)
        require_checksum(actual, expected_sha256, context=source_key)
        destination = self._path(destination_key)
        temporary = self._path(temporary_storage_key(destination_key))
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(source, temporary)
            require_checksum(sha256_file(temporary), actual, context=str(temporary))
            fsync_file(temporary)
            os.replace(temporary, destination)
            fsync_directory(destination.parent)
        finally:
            temporary.unlink(missing_ok=True)
        return self.head(destination_key)

    def delete(self, key: str) -> None:
        path = self._path(key)
        if not path.exists():
            return
        path.unlink()
        fsync_directory(path.parent)

    def head(self, key: str) -> StorageObjectMetadata:
        path = self._path(key)
        if not path.is_file():
            raise ObjectNotFoundError(f"storage object does not exist: {key}")
        stat = path.stat()
        return StorageObjectMetadata(
            key=normalize_storage_key(key),
            size=stat.st_size,
            checksum_sha256=sha256_file(path),
            last_modified_epoch_seconds=stat.st_mtime,
        )

    def verify_checksum(self, key: str, expected_sha256: str) -> bool:
        path = self._path(key)
        if not path.is_file():
            raise ObjectNotFoundError(f"storage object does not exist: {key}")
        return sha256_file(path).lower() == expected_sha256.lower()
