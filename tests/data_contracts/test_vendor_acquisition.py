"""Tests for provider-independent raw market-data acquisition."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trading_bot.data.acquisition import (
    AcquisitionRetryPolicy,
    AcquisitionRunner,
    PermanentAcquisitionError,
    RequestRateLimiter,
    TransientAcquisitionError,
    VendorPayload,
    VendorRequest,
)
from trading_bot.storage import LocalStorageBackend


class FakeAdapter:
    provider_name = "fake"

    def __init__(self, payload: bytes = b"raw", *, failures: int = 0) -> None:
        self.payload = payload
        self.failures = failures
        self.calls = 0

    def fetch(self, request: VendorRequest) -> VendorPayload:
        self.calls += 1
        if self.calls <= self.failures:
            raise TransientAcquisitionError("temporary vendor outage")
        return VendorPayload(
            self.payload,
            content_type="application/x-test",
            source_id="vendor-object-7",
            response_metadata={"request_id": "abc"},
        )


def request() -> VendorRequest:
    return VendorRequest(
        provider="fake",
        dataset="us-equities-1m",
        symbols=("AAPL", "MSFT"),
        parameters={"start": "2024-01-01", "adjusted": False},
    )


def test_vendor_request_hash_is_stable_and_canonical() -> None:
    first = request()
    second = VendorRequest.model_validate(first.model_dump(mode="python"))
    assert first.request_sha256() == second.request_sha256()
    assert json.loads(first.canonical_json())["parameters"]["adjusted"] is False


def test_acquisition_preserves_exact_raw_bytes_and_request_record(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path / "raw-store")
    timestamp = datetime(2026, 8, 18, 8, 0, tzinfo=UTC)
    runner = AcquisitionRunner(backend, now=lambda: timestamp)
    record = runner.acquire(FakeAdapter(b"vendor bytes\x00\x01"), request())
    raw_path = tmp_path / "restored.bin"
    backend.download(record.raw_object_key, raw_path, expected_sha256=record.payload_sha256)
    assert raw_path.read_bytes() == b"vendor bytes\x00\x01"
    record_keys = backend.list(
        f"00_raw/fake/{record.request_sha256}/acquisitions/"
    )
    assert len(record_keys) == 1
    record_path = tmp_path / "record.json"
    backend.download(record_keys[0].key, record_path)
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["request"]["parameters"] == {"adjusted": False, "start": "2024-01-01"}
    assert payload["downloaded_at_utc"] == "2026-08-18T08:00:00Z"
    assert payload["vendor_source_id"] == "vendor-object-7"


def test_repeated_same_payload_does_not_overwrite_raw_object(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path / "store")
    times = iter(
        [
            datetime(2026, 8, 18, 8, 0, tzinfo=UTC),
            datetime(2026, 8, 18, 8, 1, tzinfo=UTC),
        ]
    )
    runner = AcquisitionRunner(backend, now=lambda: next(times))
    first = runner.acquire(FakeAdapter(b"same"), request())
    second = runner.acquire(FakeAdapter(b"same"), request())
    assert first.raw_object_key == second.raw_object_key
    assert len(backend.list(f"00_raw/fake/{first.request_sha256}/acquisitions/")) == 2


def test_transient_failure_retries_with_bounded_backoff(tmp_path: Path) -> None:
    delays: list[float] = []
    adapter = FakeAdapter(failures=2)
    runner = AcquisitionRunner(
        LocalStorageBackend(tmp_path / "store"),
        retry_policy=AcquisitionRetryPolicy(
            max_attempts=4,
            initial_delay_seconds=0.25,
            multiplier=2,
            max_delay_seconds=1,
        ),
        sleep=delays.append,
    )
    runner.acquire(adapter, request())
    assert adapter.calls == 3
    assert delays == [0.25, 0.5]


def test_transient_failure_stops_at_retry_limit(tmp_path: Path) -> None:
    adapter = FakeAdapter(failures=5)
    runner = AcquisitionRunner(
        LocalStorageBackend(tmp_path / "store"),
        retry_policy=AcquisitionRetryPolicy(max_attempts=2, initial_delay_seconds=0),
        sleep=lambda _: None,
    )
    with pytest.raises(TransientAcquisitionError):
        runner.acquire(adapter, request())
    assert adapter.calls == 2


def test_rate_limiter_sleeps_for_remaining_interval() -> None:
    times = iter([0.0, 0.1, 0.5])
    sleeps: list[float] = []
    limiter = RequestRateLimiter(
        2.0,
        monotonic=lambda: next(times),
        sleep=sleeps.append,
    )
    limiter.wait()
    limiter.wait()
    assert sleeps == pytest.approx([0.4])


def test_adapter_provider_must_match_request(tmp_path: Path) -> None:
    class WrongAdapter(FakeAdapter):
        provider_name = "other"

    runner = AcquisitionRunner(LocalStorageBackend(tmp_path / "store"))
    with pytest.raises(PermanentAcquisitionError, match="does not match"):
        runner.acquire(WrongAdapter(), request())


def test_provider_component_rejects_unsafe_storage_name(tmp_path: Path) -> None:
    class UnsafeAdapter(FakeAdapter):
        provider_name = "bad/provider"

    unsafe_request = VendorRequest(provider="bad/provider", dataset="x")
    runner = AcquisitionRunner(LocalStorageBackend(tmp_path / "store"))
    with pytest.raises(PermanentAcquisitionError, match="not safe"):
        runner.acquire(UnsafeAdapter(), unsafe_request)
