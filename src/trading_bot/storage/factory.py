"""Construction helpers that keep callers independent of concrete storage backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from trading_bot.storage.base import StorageBackend
from trading_bot.storage.local import LocalStorageBackend
from trading_bot.storage.s3 import S3StorageBackend

if TYPE_CHECKING:
    from trading_bot.config.schemas import StorageConfig


def create_storage_backend(config: StorageConfig, *, client: Any | None = None) -> StorageBackend:
    """Build the configured backend; the same S3 implementation can serve staging or durable use."""
    if config.backend == "local":
        assert config.root_path is not None
        return LocalStorageBackend(config.root_path)

    assert config.bucket is not None
    threshold_bytes = int(config.multipart_threshold_mb) * 1024 * 1024
    return S3StorageBackend(
        bucket=config.bucket,
        prefix=config.prefix,
        endpoint_url=config.endpoint_url,
        region=config.region,
        access_key=_secret(config.access_key),
        secret_key=_secret(config.secret_key),
        session_token=_secret(config.session_token),
        multipart_threshold_bytes=threshold_bytes,
        client=client,
    )


def _secret(value: Any | None) -> str | None:
    if value is None:
        return None
    get_secret_value = getattr(value, "get_secret_value", None)
    return str(get_secret_value()) if callable(get_secret_value) else str(value)
