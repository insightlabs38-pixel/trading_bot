"""Tests for the reusable HTTPS vendor transport adapter."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

from trading_bot.data import (
    AcquisitionRunner,
    HttpGetVendorAdapter,
    PermanentAcquisitionError,
    TransientAcquisitionError,
    VendorRequest,
)
from trading_bot.storage import LocalStorageBackend


class FakeResponse:
    def __init__(
        self,
        content: bytes = b"payload",
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.content = content
        self.status = status
        self.headers = headers or {}
        self.closed = False

    def read(self) -> bytes:
        return self.content

    def close(self) -> None:
        self.closed = True


class RecordingOpener:
    def __init__(self, response: FakeResponse | BaseException) -> None:
        self.response = response
        self.requests: list[tuple[Any, float]] = []

    def __call__(self, request: Any, *, timeout: float) -> FakeResponse:
        self.requests.append((request, timeout))
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


def vendor_request() -> VendorRequest:
    return VendorRequest(
        provider="example",
        dataset="us-equities-1m",
        symbols=("AAPL",),
        parameters={"start": "2024-01-02", "end": "2024-01-03"},
    )


def test_header_api_key_is_runtime_only_and_metadata_is_sanitized() -> None:
    response = FakeResponse(
        b'{"ok":true}',
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "ETag": '"vendor-etag"',
            "X-Request-Id": "request-7",
            "Authorization": "vendor-response-secret",
        },
    )
    opener = RecordingOpener(response)
    adapter = HttpGetVendorAdapter(
        "example",
        lambda request: "https://data.example.test/bars?symbol=" + request.symbols[0],
        api_key="runtime-secret",
        api_key_header="X-Api-Key",
        timeout_seconds=17,
        opener=opener,
    )

    payload = adapter.fetch(vendor_request())

    request, timeout = opener.requests[0]
    headers = {name.lower(): value for name, value in request.header_items()}
    assert request.full_url == "https://data.example.test/bars?symbol=AAPL"
    assert headers["x-api-key"] == "runtime-secret"
    assert timeout == 17
    assert payload.content == b'{"ok":true}'
    assert payload.content_type == "application/json"
    assert payload.source_id == "request-7"
    assert payload.response_metadata == {
        "http_status": 200,
        "etag": '"vendor-etag"',
        "request_id": "request-7",
    }
    assert response.closed


def test_query_api_key_is_injected_after_public_url_validation() -> None:
    opener = RecordingOpener(FakeResponse())
    adapter = HttpGetVendorAdapter(
        "example",
        lambda _request: "https://data.example.test/bars?symbol=AAPL",
        api_key="runtime-secret",
        api_key_query_parameter="apiKey",
        opener=opener,
    )

    adapter.fetch(vendor_request())

    request, _timeout = opener.requests[0]
    assert request.full_url.endswith("symbol=AAPL&apiKey=runtime-secret")
    assert "runtime-secret" not in vendor_request().canonical_json()


@pytest.mark.parametrize(
    "url",
    [
        "http://data.example.test/bars",
        "https://user:password@data.example.test/bars",
        "https://data.example.test/bars#fragment",
        "https://data.example.test/bars?token=committed-secret",
    ],
)
def test_public_url_rejects_insecure_or_credential_bearing_forms(url: str) -> None:
    adapter = HttpGetVendorAdapter(
        "example", lambda _request: url, opener=RecordingOpener(FakeResponse())
    )
    with pytest.raises(PermanentAcquisitionError):
        adapter.fetch(vendor_request())


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (400, PermanentAcquisitionError),
        (401, PermanentAcquisitionError),
        (404, PermanentAcquisitionError),
        (408, TransientAcquisitionError),
        (425, TransientAcquisitionError),
        (429, TransientAcquisitionError),
        (500, TransientAcquisitionError),
        (503, TransientAcquisitionError),
    ],
)
def test_response_status_is_classified_for_runner_retry_policy(
    status: int,
    error_type: type[RuntimeError],
) -> None:
    adapter = HttpGetVendorAdapter(
        "example",
        lambda _request: "https://data.example.test/bars",
        opener=RecordingOpener(FakeResponse(status=status)),
    )
    with pytest.raises(error_type):
        adapter.fetch(vendor_request())


def test_http_error_and_transport_error_are_classified_without_leaking_url() -> None:
    http_failure = RecordingOpener(
        HTTPError(
            "https://data.example.test/bars?apiKey=secret",
            503,
            "unavailable",
            None,
            None,
        )
    )
    adapter = HttpGetVendorAdapter(
        "example",
        lambda _request: "https://data.example.test/bars",
        opener=http_failure,
    )
    with pytest.raises(TransientAcquisitionError) as http_error:
        adapter.fetch(vendor_request())
    assert "secret" not in str(http_error.value)

    transport_failure = RecordingOpener(URLError("socket unavailable"))
    adapter = HttpGetVendorAdapter(
        "example",
        lambda _request: "https://data.example.test/bars",
        opener=transport_failure,
    )
    with pytest.raises(TransientAcquisitionError, match="temporary HTTP transport failure"):
        adapter.fetch(vendor_request())


def test_sensitive_public_header_must_use_runtime_secret_channel() -> None:
    with pytest.raises(ValueError, match="runtime API-key injection"):
        HttpGetVendorAdapter(
            "example",
            lambda _request: "https://data.example.test/bars",
            public_headers={"Authorization": "committed-secret"},
        )


def test_runner_preserves_raw_bytes_without_persisting_runtime_secret(tmp_path: Path) -> None:
    opener = RecordingOpener(
        FakeResponse(
            b"exact-vendor-bytes",
            headers={"Content-Type": "application/octet-stream", "ETag": '"raw-17"'},
        )
    )
    adapter = HttpGetVendorAdapter(
        "example",
        lambda _request: "https://data.example.test/bars?symbol=AAPL",
        api_key="runtime-secret",
        api_key_header="X-Api-Key",
        opener=opener,
    )
    backend = LocalStorageBackend(tmp_path / "store")
    runner = AcquisitionRunner(
        backend,
        now=lambda: datetime(2026, 8, 18, 14, 0, tzinfo=UTC),
    )

    record = runner.acquire(adapter, vendor_request())

    restored = tmp_path / "raw.bin"
    backend.download(record.raw_object_key, restored, expected_sha256=record.payload_sha256)
    assert restored.read_bytes() == b"exact-vendor-bytes"
    assert "runtime-secret" not in record.request.canonical_json()
    assert "runtime-secret" not in str(record.response_metadata)
    assert record.vendor_source_id == '"raw-17"'
