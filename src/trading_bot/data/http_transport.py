"""Reusable HTTPS transport adapter for vendor-specific market-data integrations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from trading_bot.data.acquisition import (
    PermanentAcquisitionError,
    TransientAcquisitionError,
    VendorPayload,
    VendorRequest,
)

_SENSITIVE_HEADER_NAMES = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "api-key",
        "apikey",
    }
)
_SENSITIVE_QUERY_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "api-key",
        "access_token",
        "token",
        "secret",
        "password",
    }
)
_RESPONSE_METADATA_HEADERS = {
    "etag": "etag",
    "last-modified": "last_modified",
    "content-length": "content_length",
    "x-request-id": "request_id",
    "x-ratelimit-limit": "rate_limit",
    "x-ratelimit-remaining": "rate_limit_remaining",
    "x-ratelimit-reset": "rate_limit_reset",
}


class HttpGetVendorAdapter:
    """Provider-neutral GET transport with runtime-only authentication injection.

    The adapter intentionally does not encode any vendor-specific URL shape. A future finalized
    provider adapter supplies a ``url_builder`` that translates the canonical ``VendorRequest``
    into a public, non-secret HTTPS URL. Authentication is injected separately at runtime so API
    keys never need to appear in persisted request parameters or acquisition records.
    """

    def __init__(
        self,
        provider_name: str,
        url_builder: Callable[[VendorRequest], str],
        *,
        api_key: str | None = None,
        api_key_header: str | None = None,
        api_key_query_parameter: str | None = None,
        timeout_seconds: float = 60.0,
        public_headers: Mapping[str, str] | None = None,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        provider = provider_name.strip()
        if not provider:
            raise ValueError("provider_name must not be blank")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if api_key_header is not None and api_key_query_parameter is not None:
            raise ValueError("configure at most one API-key injection location")
        if api_key is None and (api_key_header is not None or api_key_query_parameter is not None):
            raise ValueError("API-key injection location requires api_key")
        if api_key is not None and api_key_header is None and api_key_query_parameter is None:
            raise ValueError("api_key requires a header or query-parameter injection location")
        if api_key_header is not None and not api_key_header.strip():
            raise ValueError("api_key_header must not be blank")
        if api_key_query_parameter is not None and not api_key_query_parameter.strip():
            raise ValueError("api_key_query_parameter must not be blank")

        headers = dict(public_headers or {})
        for name in headers:
            if name.strip().lower() in _SENSITIVE_HEADER_NAMES:
                raise ValueError(
                    f"sensitive header {name!r} must use runtime API-key injection instead"
                )

        self.provider_name = provider
        self._url_builder = url_builder
        self._api_key = api_key
        self._api_key_header = None if api_key_header is None else api_key_header.strip()
        self._api_key_query_parameter = (
            None if api_key_query_parameter is None else api_key_query_parameter.strip()
        )
        self._timeout_seconds = timeout_seconds
        self._public_headers = headers
        self._opener = opener

    def fetch(self, request: VendorRequest) -> VendorPayload:
        """Fetch exact vendor bytes and return only a sanitized metadata subset."""
        if request.provider != self.provider_name:
            raise PermanentAcquisitionError(
                f"adapter provider {self.provider_name!r} does not match request provider "
                f"{request.provider!r}"
            )
        public_url = self._validate_public_url(self._url_builder(request))
        final_url = self._inject_query_secret(public_url)
        headers = dict(self._public_headers)
        if self._api_key_header is not None:
            assert self._api_key is not None
            headers[self._api_key_header] = self._api_key

        http_request = Request(final_url, headers=headers, method="GET")
        try:
            response = self._opener(http_request, timeout=self._timeout_seconds)
        except HTTPError as exc:
            self._raise_for_status(int(exc.code))
            raise AssertionError("unreachable HTTP error classification") from exc
        except (URLError, TimeoutError, ConnectionError, OSError) as exc:
            raise TransientAcquisitionError(
                f"temporary HTTP transport failure for provider {self.provider_name}"
            ) from exc

        try:
            status = int(getattr(response, "status", getattr(response, "code", 200)))
            self._raise_for_status(status)
            content = response.read()
            response_headers = getattr(response, "headers", {})
            content_type = _header(response_headers, "Content-Type")
            metadata = _sanitized_response_metadata(response_headers, status)
            source_id = metadata.get("request_id") or metadata.get("etag")
            return VendorPayload(
                bytes(content),
                content_type=_base_content_type(content_type),
                source_id=None if source_id is None else str(source_id),
                response_metadata=metadata,
            )
        except (TransientAcquisitionError, PermanentAcquisitionError):
            raise
        except (TimeoutError, ConnectionError, OSError) as exc:
            raise TransientAcquisitionError(
                f"temporary HTTP response failure for provider {self.provider_name}"
            ) from exc
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

    def _validate_public_url(self, url: str) -> str:
        parsed = urlsplit(url)
        if parsed.scheme.lower() != "https" or not parsed.netloc:
            raise PermanentAcquisitionError("vendor URL must be an absolute HTTPS URL")
        if parsed.username is not None or parsed.password is not None:
            raise PermanentAcquisitionError("vendor URL must not contain userinfo credentials")
        if parsed.fragment:
            raise PermanentAcquisitionError("vendor URL must not contain a fragment")

        for name, _value in parse_qsl(parsed.query, keep_blank_values=True):
            if name.strip().lower() in _SENSITIVE_QUERY_NAMES:
                raise PermanentAcquisitionError(
                    "vendor URL contains a credential-like query parameter; "
                    "inject credentials at runtime"
                )
        return urlunsplit(parsed)

    def _inject_query_secret(self, public_url: str) -> str:
        if self._api_key_query_parameter is None:
            return public_url
        assert self._api_key is not None
        parsed = urlsplit(public_url)
        query = parse_qsl(parsed.query, keep_blank_values=True)
        if any(name == self._api_key_query_parameter for name, _value in query):
            raise PermanentAcquisitionError("API-key query parameter already exists in vendor URL")
        query.append((self._api_key_query_parameter, self._api_key))
        return urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment)
        )

    def _raise_for_status(self, status: int) -> None:
        if status < 400:
            return
        message = f"HTTP {status} from provider {self.provider_name}"
        if status in {408, 425, 429} or status >= 500:
            raise TransientAcquisitionError(message)
        raise PermanentAcquisitionError(message)


def _header(headers: Any, name: str) -> str | None:
    getter = getattr(headers, "get", None)
    if callable(getter):
        value = getter(name)
        if value is not None:
            return str(value)
    items = getattr(headers, "items", None)
    if callable(items):
        target = name.lower()
        for key, value in items():
            if str(key).lower() == target:
                return str(value)
    return None


def _base_content_type(value: str | None) -> str:
    if value is None:
        return "application/octet-stream"
    media_type = value.split(";", maxsplit=1)[0].strip()
    return media_type or "application/octet-stream"


def _sanitized_response_metadata(headers: Any, status: int) -> dict[str, str | int]:
    metadata: dict[str, str | int] = {"http_status": status}
    for header_name, field_name in _RESPONSE_METADATA_HEADERS.items():
        value = _header(headers, header_name)
        if value is not None:
            metadata[field_name] = value
    return metadata
