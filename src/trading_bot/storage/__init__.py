"""Interchangeable local and S3-compatible storage backends."""

from trading_bot.storage.base import (
    ChecksumMismatchError,
    ObjectNotFoundError,
    RetryPolicy,
    StorageBackend,
    StorageError,
    StorageObjectMetadata,
    TransferTimeoutPolicy,
    UnsafeStorageKeyError,
    normalize_storage_key,
    sha256_file,
    temporary_storage_key,
)
from trading_bot.storage.factory import create_storage_backend
from trading_bot.storage.local import LocalStorageBackend
from trading_bot.storage.s3 import S3StorageBackend

__all__ = [
    "ChecksumMismatchError",
    "LocalStorageBackend",
    "ObjectNotFoundError",
    "RetryPolicy",
    "S3StorageBackend",
    "StorageBackend",
    "StorageError",
    "StorageObjectMetadata",
    "TransferTimeoutPolicy",
    "UnsafeStorageKeyError",
    "create_storage_backend",
    "normalize_storage_key",
    "sha256_file",
    "temporary_storage_key",
]
