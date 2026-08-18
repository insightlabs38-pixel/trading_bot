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
from trading_bot.storage.bulk import (
    BulkTransferError,
    BulkTransferManager,
    BulkTransferStats,
    DownloadItem,
    UploadItem,
)
from trading_bot.storage.factory import create_storage_backend
from trading_bot.storage.local import LocalStorageBackend
from trading_bot.storage.manifests import (
    ArtifactManifest,
    ArtifactManifestError,
    ArtifactVerificationError,
    build_artifact_manifest,
    load_artifact_manifest,
    manifest_key_for,
    publish_artifact_with_manifest,
    verify_artifact_manifest,
    write_artifact_manifest,
)
from trading_bot.storage.s3 import S3StorageBackend

__all__ = [
    "BulkTransferError",
    "BulkTransferManager",
    "BulkTransferStats",
    "DownloadItem",
    "ArtifactManifest",
    "ArtifactManifestError",
    "ArtifactVerificationError",
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
    "UploadItem",
    "build_artifact_manifest",
    "load_artifact_manifest",
    "manifest_key_for",
    "publish_artifact_with_manifest",
    "verify_artifact_manifest",
    "write_artifact_manifest",
    "create_storage_backend",
    "normalize_storage_key",
    "sha256_file",
    "temporary_storage_key",
]
