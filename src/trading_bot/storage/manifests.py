"""Versioned artifact manifests and backend-independent verification helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from trading_bot.storage.base import (
    ChecksumMismatchError,
    StorageBackend,
    StorageObjectMetadata,
    normalize_storage_key,
    sha256_file,
)
from trading_bot.storage.local import LocalStorageBackend


class ArtifactManifestError(RuntimeError):
    """Base error for invalid or unverifiable artifact manifests."""


class ArtifactVerificationError(ArtifactManifestError):
    """Raised when stored artifact metadata does not match its manifest."""


class ArtifactManifest(BaseModel):
    """Immutable provenance and integrity record for one stored artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    schema_version: Literal[1] = 1
    artifact_key: str
    size_bytes: int = Field(ge=0)
    checksum_algorithm: Literal["sha256"] = "sha256"
    checksum: str
    artifact_schema: str = Field(min_length=1)
    artifact_version: str = Field(min_length=1)
    created_at_utc: datetime
    producer_git_sha: str | None = None
    producer_config_sha256: str | None = None
    row_count: int | None = Field(default=None, ge=0)
    tensor_shape: tuple[int, ...] | None = None
    generation_stage: str | None = None
    upstream_ids: tuple[str, ...] = ()
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("artifact_key")
    @classmethod
    def validate_artifact_key(cls, value: str) -> str:
        return normalize_storage_key(value)

    @field_validator("checksum", "producer_config_sha256")
    @classmethod
    def validate_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.lower()
        if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
            raise ValueError("SHA-256 values must be 64 lowercase hexadecimal characters")
        return normalized

    @field_validator("producer_git_sha")
    @classmethod
    def validate_git_sha(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.lower()
        if not 7 <= len(normalized) <= 64 or any(
            char not in "0123456789abcdef" for char in normalized
        ):
            raise ValueError("producer_git_sha must be a hexadecimal Git object ID")
        return normalized

    @field_validator("created_at_utc")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("created_at_utc must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("tensor_shape")
    @classmethod
    def validate_tensor_shape(cls, value: tuple[int, ...] | None) -> tuple[int, ...] | None:
        if value is not None and any(dimension < 0 for dimension in value):
            raise ValueError("tensor_shape dimensions must be non-negative")
        return value

    @field_validator("generation_stage")
    @classmethod
    def validate_generation_stage(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("generation_stage must not be blank")
        return stripped

    @field_validator("upstream_ids")
    @classmethod
    def validate_upstream_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("upstream_ids must not contain blank identifiers")
        if len(set(normalized)) != len(normalized):
            raise ValueError("upstream_ids must be unique")
        return normalized

    def canonical_json(self) -> str:
        """Return deterministic JSON suitable for durable storage and hashing."""
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def manifest_sha256(self) -> str:
        """Hash the canonical manifest document itself for audit references."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def manifest_key_for(artifact_key: str) -> str:
    """Return the conventional adjacent manifest key for an artifact."""
    return f"{normalize_storage_key(artifact_key)}.manifest.json"


def _checksum_for_backend_object(backend: StorageBackend, key: str) -> str:
    metadata = backend.head(key)
    if metadata.checksum_sha256 is not None:
        return metadata.checksum_sha256.lower()
    with tempfile.TemporaryDirectory(prefix="trading-bot-manifest-") as directory:
        path = Path(directory) / "artifact"
        backend.download(key, path)
        return sha256_file(path)


def build_artifact_manifest(
    backend: StorageBackend,
    artifact_key: str,
    *,
    artifact_schema: str,
    artifact_version: str,
    producer_git_sha: str | None = None,
    producer_config_sha256: str | None = None,
    row_count: int | None = None,
    tensor_shape: tuple[int, ...] | None = None,
    generation_stage: str | None = None,
    upstream_ids: tuple[str, ...] = (),
    metadata: dict[str, JsonValue] | None = None,
    created_at_utc: datetime | None = None,
) -> ArtifactManifest:
    """Build a manifest from the bytes currently stored at ``artifact_key``."""
    key = normalize_storage_key(artifact_key)
    object_metadata = backend.head(key)
    checksum = _checksum_for_backend_object(backend, key)
    if not backend.verify_checksum(key, checksum):
        raise ArtifactVerificationError(f"backend checksum verification failed for {key}")
    return ArtifactManifest(
        artifact_key=key,
        size_bytes=object_metadata.size,
        checksum=checksum,
        artifact_schema=artifact_schema,
        artifact_version=artifact_version,
        created_at_utc=datetime.now(UTC) if created_at_utc is None else created_at_utc,
        producer_git_sha=producer_git_sha,
        producer_config_sha256=producer_config_sha256,
        row_count=row_count,
        tensor_shape=tensor_shape,
        generation_stage=generation_stage,
        upstream_ids=upstream_ids,
        metadata={} if metadata is None else metadata,
    )


def write_artifact_manifest(
    backend: StorageBackend,
    manifest: ArtifactManifest,
    *,
    manifest_key: str | None = None,
) -> StorageObjectMetadata:
    """Atomically publish a canonical JSON manifest through the selected backend."""
    key = manifest_key_for(manifest.artifact_key) if manifest_key is None else manifest_key
    normalized_key = normalize_storage_key(key)
    payload = manifest.canonical_json().encode("utf-8")
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    with tempfile.TemporaryDirectory(prefix="trading-bot-manifest-") as directory:
        path = Path(directory) / "manifest.json"
        path.write_bytes(payload)
        return backend.upload(path, normalized_key, expected_sha256=expected_sha256)


def load_artifact_manifest(backend: StorageBackend, manifest_key: str) -> ArtifactManifest:
    """Load and strongly validate one manifest document from storage."""
    normalized_key = normalize_storage_key(manifest_key)
    with tempfile.TemporaryDirectory(prefix="trading-bot-manifest-") as directory:
        path = Path(directory) / "manifest.json"
        backend.download(normalized_key, path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            message = f"invalid manifest document {normalized_key}: {exc}"
            raise ArtifactManifestError(message) from exc
    try:
        return ArtifactManifest.model_validate(payload)
    except ValueError as exc:
        raise ArtifactManifestError(f"invalid manifest schema {normalized_key}: {exc}") from exc


def verify_artifact_manifest(
    backend: StorageBackend,
    manifest: ArtifactManifest | str,
) -> ArtifactManifest:
    """Verify artifact existence, byte size, and checksum against its manifest."""
    resolved = load_artifact_manifest(backend, manifest) if isinstance(manifest, str) else manifest
    actual = backend.head(resolved.artifact_key)
    if actual.size != resolved.size_bytes:
        raise ArtifactVerificationError(
            f"size mismatch for {resolved.artifact_key}: "
            f"expected {resolved.size_bytes}, got {actual.size}"
        )
    try:
        matches = backend.verify_checksum(resolved.artifact_key, resolved.checksum)
    except ChecksumMismatchError as exc:
        raise ArtifactVerificationError(str(exc)) from exc
    if not matches:
        raise ArtifactVerificationError(f"checksum mismatch for {resolved.artifact_key}")
    return resolved


def publish_artifact_with_manifest(
    backend: StorageBackend,
    source: str | Path,
    artifact_key: str,
    *,
    artifact_schema: str,
    artifact_version: str,
    producer_git_sha: str | None = None,
    producer_config_sha256: str | None = None,
    row_count: int | None = None,
    tensor_shape: tuple[int, ...] | None = None,
    generation_stage: str | None = None,
    upstream_ids: tuple[str, ...] = (),
    metadata: dict[str, JsonValue] | None = None,
) -> ArtifactManifest:
    """Publish verified bytes first, then publish their manifest only after verification."""
    backend.upload(source, artifact_key)
    manifest = build_artifact_manifest(
        backend,
        artifact_key,
        artifact_schema=artifact_schema,
        artifact_version=artifact_version,
        producer_git_sha=producer_git_sha,
        producer_config_sha256=producer_config_sha256,
        row_count=row_count,
        tensor_shape=tensor_shape,
        generation_stage=generation_stage,
        upstream_ids=upstream_ids,
        metadata=metadata,
    )
    verify_artifact_manifest(backend, manifest)
    write_artifact_manifest(backend, manifest)
    return manifest


def _verify_local_command(args: argparse.Namespace) -> int:
    backend = LocalStorageBackend(args.root)
    manifest = verify_artifact_manifest(backend, args.manifest_key)
    print(
        json.dumps(
            {
                "artifact_key": manifest.artifact_key,
                "manifest_key": normalize_storage_key(args.manifest_key),
                "manifest_sha256": manifest.manifest_sha256(),
                "verified": True,
            },
            sort_keys=True,
        )
    )
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Artifact manifest utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_local = subparsers.add_parser("verify-local", help="verify an artifact in local storage")
    verify_local.add_argument("--root", required=True, help="local storage root")
    verify_local.add_argument("--manifest-key", required=True, help="manifest object key")
    verify_local.set_defaults(handler=_verify_local_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
