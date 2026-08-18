"""Tests for immutable storage artifact manifests and verification."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from trading_bot.storage import LocalStorageBackend
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


def test_manifest_schema_contains_required_and_lineage_fields(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path / "store")
    source = tmp_path / "part.parquet"
    source.write_bytes(b"parquet-ish")
    backend.upload(source, "canonical/part.parquet")
    manifest = build_artifact_manifest(
        backend,
        "canonical/part.parquet",
        artifact_schema="minute-bars-v1",
        artifact_version="dataset-2026-08-18",
        producer_git_sha="a" * 40,
        producer_config_sha256="b" * 64,
        row_count=123,
        generation_stage="03_adjusted/canonical",
        upstream_ids=("raw-v7", "security-master-v3"),
        metadata={"partition": "2020-01"},
        created_at_utc=datetime(2026, 8, 18, tzinfo=UTC),
    )
    assert manifest.artifact_key == "canonical/part.parquet"
    assert manifest.size_bytes == len(b"parquet-ish")
    assert manifest.checksum == hashlib.sha256(b"parquet-ish").hexdigest()
    assert manifest.row_count == 123
    assert manifest.upstream_ids == ("raw-v7", "security-master-v3")


def test_manifest_is_immutable_and_rejects_unknown_fields() -> None:
    manifest = ArtifactManifest(
        artifact_key="x.bin",
        size_bytes=1,
        checksum="0" * 64,
        artifact_schema="bytes-v1",
        artifact_version="v1",
        created_at_utc=datetime.now(UTC),
    )
    with pytest.raises(ValidationError):
        manifest.size_bytes = 2  # type: ignore[misc]
    payload = manifest.model_dump(mode="python")
    payload["mystery"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ArtifactManifest.model_validate(payload)


def test_manifest_canonical_json_and_hash_are_stable() -> None:
    manifest = ArtifactManifest(
        artifact_key="x.bin",
        size_bytes=1,
        checksum="0" * 64,
        artifact_schema="bytes-v1",
        artifact_version="v1",
        created_at_utc=datetime(2026, 8, 18, tzinfo=UTC),
        metadata={"z": 1, "a": 2},
    )
    reconstructed = ArtifactManifest.model_validate(json.loads(manifest.canonical_json()))
    assert reconstructed == manifest
    assert reconstructed.manifest_sha256() == manifest.manifest_sha256()
    assert manifest.canonical_json().index('"a"') < manifest.canonical_json().index('"z"')


def test_write_load_and_verify_manifest_round_trip(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path / "store")
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"verified content")
    backend.upload(source, "artifacts/result.bin")
    manifest = build_artifact_manifest(
        backend,
        "artifacts/result.bin",
        artifact_schema="result-v1",
        artifact_version="1",
    )
    metadata = write_artifact_manifest(backend, manifest)
    assert metadata.key == "artifacts/result.bin.manifest.json"
    loaded = load_artifact_manifest(backend, metadata.key)
    assert loaded == manifest
    assert verify_artifact_manifest(backend, metadata.key) == manifest


def test_verification_detects_artifact_tampering(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path / "store")
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"before")
    manifest = publish_artifact_with_manifest(
        backend,
        source,
        "artifact.bin",
        artifact_schema="bytes-v1",
        artifact_version="1",
    )
    tampered = tmp_path / "tampered.bin"
    tampered.write_bytes(b"after!")
    backend.upload(tampered, manifest.artifact_key)
    with pytest.raises(ArtifactVerificationError, match="checksum mismatch"):
        verify_artifact_manifest(backend, manifest)


def test_verification_detects_size_mismatch_before_checksum(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path / "store")
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"abc")
    backend.upload(source, "artifact.bin")
    manifest = build_artifact_manifest(
        backend,
        "artifact.bin",
        artifact_schema="bytes-v1",
        artifact_version="1",
    )
    payload = manifest.model_dump(mode="python")
    payload["size_bytes"] = 999
    wrong_size = ArtifactManifest.model_validate(payload)
    with pytest.raises(ArtifactVerificationError, match="size mismatch"):
        verify_artifact_manifest(backend, wrong_size)


def test_invalid_manifest_document_is_rejected(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path / "store")
    bad = tmp_path / "bad.json"
    bad.write_text("{not-json", encoding="utf-8")
    backend.upload(bad, "bad.manifest.json")
    with pytest.raises(ArtifactManifestError, match="invalid manifest document"):
        load_artifact_manifest(backend, "bad.manifest.json")


def test_publish_writes_manifest_only_after_artifact_is_verified(tmp_path: Path) -> None:
    class FailingVerificationBackend(LocalStorageBackend):
        def verify_checksum(self, key: str, expected_sha256: str) -> bool:
            if key == "artifact.bin":
                return False
            return super().verify_checksum(key, expected_sha256)

    backend = FailingVerificationBackend(tmp_path / "store")
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"content")
    with pytest.raises(ArtifactVerificationError):
        publish_artifact_with_manifest(
            backend,
            source,
            "artifact.bin",
            artifact_schema="bytes-v1",
            artifact_version="1",
        )
    assert not backend.exists(manifest_key_for("artifact.bin"))


def test_tensor_shape_and_upstream_ids_are_validated() -> None:
    base = {
        "artifact_key": "x.bin",
        "size_bytes": 1,
        "checksum": "0" * 64,
        "artifact_schema": "tensor-v1",
        "artifact_version": "1",
        "created_at_utc": datetime.now(UTC),
    }
    with pytest.raises(ValidationError, match="tensor_shape"):
        ArtifactManifest(**base, tensor_shape=(4, -1))
    with pytest.raises(ValidationError, match="upstream_ids"):
        ArtifactManifest(**base, upstream_ids=("same", "same"))


def test_cli_verifies_local_manifest(tmp_path: Path) -> None:
    store = tmp_path / "store"
    backend = LocalStorageBackend(store)
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"cli")
    publish_artifact_with_manifest(
        backend,
        source,
        "artifact.bin",
        artifact_schema="bytes-v1",
        artifact_version="1",
    )
    environment = dict(__import__("os").environ)
    src = str(Path(__file__).parents[2] / "src")
    environment["PYTHONPATH"] = src + (
        __import__("os").pathsep + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH")
        else ""
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "trading_bot.storage.manifests",
            "verify-local",
            "--root",
            str(store),
            "--manifest-key",
            "artifact.bin.manifest.json",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    output = json.loads(completed.stdout)
    assert output["verified"] is True
    assert output["artifact_key"] == "artifact.bin"
