"""Provider-independent market-data acquisition with immutable raw preservation."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from trading_bot.storage import StorageBackend, normalize_storage_key


class AcquisitionError(RuntimeError):
    """Base error for vendor acquisition failures."""


class TransientAcquisitionError(AcquisitionError):
    """Retryable vendor or transport failure."""


class PermanentAcquisitionError(AcquisitionError):
    """Non-retryable vendor request failure."""


class VendorRequest(BaseModel):
    """Canonical provider request whose exact semantics are persisted for auditability."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    symbols: tuple[str, ...] = ()
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("provider", "dataset")
    @classmethod
    def strip_nonblank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized

    @field_validator("symbols")
    @classmethod
    def normalize_symbols(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(symbol.strip() for symbol in value)
        if any(not symbol for symbol in normalized):
            raise ValueError("symbols must not contain blanks")
        if len(set(normalized)) != len(normalized):
            raise ValueError("symbols must be unique")
        return normalized

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def request_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class VendorPayload:
    """Raw payload returned by a vendor adapter without transformation."""

    content: bytes
    content_type: str = "application/octet-stream"
    source_id: str | None = None
    response_metadata: Mapping[str, JsonValue] | None = None


class VendorAdapter(Protocol):
    """Minimal interface implemented by each finalized data vendor integration."""

    provider_name: str

    def fetch(self, request: VendorRequest) -> VendorPayload: ...


class AcquisitionRecord(BaseModel):
    """Immutable audit record for one successful vendor download."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    request: VendorRequest
    request_sha256: str
    downloaded_at_utc: datetime
    raw_object_key: str
    payload_size_bytes: int = Field(ge=0)
    payload_sha256: str
    content_type: str
    vendor_source_id: str | None = None
    response_metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("downloaded_at_utc")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("downloaded_at_utc must be timezone-aware")
        return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class AcquisitionRetryPolicy:
    max_attempts: int = 4
    initial_delay_seconds: float = 0.5
    multiplier: float = 2.0
    max_delay_seconds: float = 8.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.initial_delay_seconds < 0 or self.multiplier < 1 or self.max_delay_seconds < 0:
            raise ValueError("retry delays must be non-negative and multiplier >= 1")

    def delay_after_failure(self, attempt: int) -> float:
        return min(
            self.initial_delay_seconds * (self.multiplier ** max(0, attempt - 1)),
            self.max_delay_seconds,
        )


class RequestRateLimiter:
    """Simple minimum-interval limiter with injectable clock/sleep for deterministic tests."""

    def __init__(
        self,
        requests_per_second: float,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self._interval = 1.0 / requests_per_second
        self._monotonic = monotonic
        self._sleep = sleep
        self._last_request_at: float | None = None

    def wait(self) -> None:
        now = self._monotonic()
        if self._last_request_at is not None:
            remaining = self._interval - (now - self._last_request_at)
            if remaining > 0:
                self._sleep(remaining)
                now = self._monotonic()
        self._last_request_at = now


class AcquisitionRunner:
    """Fetch raw vendor bytes, verify them, and preserve request/download provenance."""

    def __init__(
        self,
        backend: StorageBackend,
        *,
        raw_prefix: str = "00_raw",
        retry_policy: AcquisitionRetryPolicy | None = None,
        rate_limiter: RequestRateLimiter | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.backend = backend
        self.raw_prefix = normalize_storage_key(raw_prefix)
        self.retry_policy = retry_policy or AcquisitionRetryPolicy()
        self.rate_limiter = rate_limiter
        self.sleep = sleep
        self.now = now

    def acquire(self, adapter: VendorAdapter, request: VendorRequest) -> AcquisitionRecord:
        if adapter.provider_name != request.provider:
            raise PermanentAcquisitionError(
                f"adapter provider {adapter.provider_name!r} does not match request provider "
                f"{request.provider!r}"
            )

        payload: VendorPayload | None = None
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            if self.rate_limiter is not None:
                self.rate_limiter.wait()
            try:
                payload = adapter.fetch(request)
                break
            except TransientAcquisitionError:
                if attempt >= self.retry_policy.max_attempts:
                    raise
                self.sleep(self.retry_policy.delay_after_failure(attempt))
        assert payload is not None
        return self._preserve(request, payload)

    def _preserve(self, request: VendorRequest, payload: VendorPayload) -> AcquisitionRecord:
        payload_sha = hashlib.sha256(payload.content).hexdigest()
        request_sha = request.request_sha256()
        provider = _safe_component(request.provider)
        raw_key = normalize_storage_key(
            f"{self.raw_prefix}/{provider}/{request_sha}/{payload_sha}.bin"
        )

        with tempfile.TemporaryDirectory(prefix="trading-bot-acquire-") as directory:
            raw_path = Path(directory) / "payload.bin"
            raw_path.write_bytes(payload.content)
            if self.backend.exists(raw_key):
                if not self.backend.verify_checksum(raw_key, payload_sha):
                    raise AcquisitionError(f"immutable raw-object checksum conflict at {raw_key}")
            else:
                self.backend.upload(raw_path, raw_key, expected_sha256=payload_sha)

            downloaded_at = self.now().astimezone(UTC)
            record = AcquisitionRecord(
                request=request,
                request_sha256=request_sha,
                downloaded_at_utc=downloaded_at,
                raw_object_key=raw_key,
                payload_size_bytes=len(payload.content),
                payload_sha256=payload_sha,
                content_type=payload.content_type,
                vendor_source_id=payload.source_id,
                response_metadata=dict(payload.response_metadata or {}),
            )
            record_payload = json.dumps(
                record.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            stamp = downloaded_at.strftime("%Y%m%dT%H%M%S.%fZ")
            record_name = f"{stamp}-{payload_sha[:12]}.json"
            record_key = normalize_storage_key(
                f"{self.raw_prefix}/{provider}/{request_sha}/acquisitions/{record_name}"
            )
            record_path = Path(directory) / "acquisition.json"
            record_path.write_bytes(record_payload)
            record_sha = hashlib.sha256(record_payload).hexdigest()
            self.backend.upload(record_path, record_key, expected_sha256=record_sha)
        return record


def _safe_component(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "-")
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789-_."
    if not normalized or any(character not in allowed for character in normalized):
        raise PermanentAcquisitionError(f"provider name is not safe for storage paths: {value!r}")
    return normalized
