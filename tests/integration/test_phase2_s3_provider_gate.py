"""Opt-in Phase 2 gate against a real S3-compatible provider."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

_ENABLED = os.getenv("TRADING_BOT_S3_TEST_ENABLED") == "1"
_BUCKET = os.getenv("TRADING_BOT_S3_TEST_BUCKET")
_ENDPOINT = os.getenv("TRADING_BOT_S3_TEST_ENDPOINT_URL")

if not (_ENABLED and _BUCKET and _ENDPOINT):
    pytest.skip(
        "real S3 provider gate requires TRADING_BOT_S3_TEST_ENABLED=1, "
        "TRADING_BOT_S3_TEST_BUCKET, and TRADING_BOT_S3_TEST_ENDPOINT_URL",
        allow_module_level=True,
    )

from trading_bot.storage import (  # noqa: E402
    BulkTransferManager,
    DownloadItem,
    S3StorageBackend,
    manifest_key_for,
    publish_artifact_with_manifest,
    verify_artifact_manifest,
)


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    return value if value else None


def test_phase2_real_s3_round_trip(tmp_path: Path) -> None:
    """Publish, delete local source, restore, and verify bytes + manifest on real S3."""
    run_id = uuid.uuid4().hex
    base_prefix = os.getenv("TRADING_BOT_S3_TEST_PREFIX", "trading-bot-phase2-integration")
    prefix = f"{base_prefix.rstrip('/')}/{run_id}"
    backend = S3StorageBackend(
        bucket=_BUCKET,
        prefix=prefix,
        endpoint_url=_ENDPOINT,
        region=_optional_env("TRADING_BOT_S3_TEST_REGION"),
        access_key=_optional_env("TRADING_BOT_S3_TEST_ACCESS_KEY"),
        secret_key=_optional_env("TRADING_BOT_S3_TEST_SECRET_KEY"),
        session_token=_optional_env("TRADING_BOT_S3_TEST_SESSION_TOKEN"),
    )
    artifact_key = "artifacts/phase2-provider-gate.bin"
    manifest_key = manifest_key_for(artifact_key)
    source = tmp_path / "generated.bin"
    restored = tmp_path / "restored.bin"
    journal = tmp_path / "restore.json"
    payload = (f"phase2-provider-gate:{run_id}\n".encode("utf-8")) * 1024
    source.write_bytes(payload)

    try:
        manifest = publish_artifact_with_manifest(
            backend,
            source,
            artifact_key,
            artifact_schema="phase2-provider-gate-v1",
            artifact_version="1",
            metadata={"test_run_id": run_id},
        )
        source.unlink()

        stats = BulkTransferManager(backend).download(
            [DownloadItem(artifact_key, restored, manifest.checksum)],
            journal_path=journal,
        )

        assert stats.transferred_items == 1
        assert restored.read_bytes() == payload
        assert verify_artifact_manifest(backend, manifest_key) == manifest
    finally:
        backend.delete(manifest_key)
        backend.delete(artifact_key)
