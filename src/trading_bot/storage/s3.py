"""S3-compatible storage backend with retries, timeouts, and atomic publication."""

from __future__ import annotations

import contextlib
import hashlib
import os
import time
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from trading_bot.storage.base import (
    ObjectNotFoundError,
    RetryPolicy,
    StorageObjectMetadata,
    TransferTimeoutPolicy,
    is_temporary_storage_key,
    normalize_storage_key,
    require_checksum,
    retry_call,
    sha256_file,
    temporary_local_path,
    temporary_storage_key,
)

_MIN_S3_PART_SIZE = 5 * 1024 * 1024


class S3StorageBackend:
    """One configurable S3-compatible namespace, reusable for durable or staging storage."""

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "",
        endpoint_url: str | None = None,
        region: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        session_token: str | None = None,
        multipart_threshold_bytes: int = 128 * 1024 * 1024,
        multipart_part_size_bytes: int = 64 * 1024 * 1024,
        retry_policy: RetryPolicy | None = None,
        timeout_policy: TransferTimeoutPolicy | None = None,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not bucket.strip():
            raise ValueError("S3 bucket must not be empty")
        if multipart_threshold_bytes < _MIN_S3_PART_SIZE:
            raise ValueError("S3 multipart threshold must be at least 5 MiB")
        if multipart_part_size_bytes < _MIN_S3_PART_SIZE:
            raise ValueError("S3 multipart part size must be at least 5 MiB")
        self.bucket = bucket
        self.prefix = normalize_storage_key(prefix, allow_empty=True).rstrip("/")
        self.multipart_threshold_bytes = multipart_threshold_bytes
        self.multipart_part_size_bytes = multipart_part_size_bytes
        self.retry_policy = retry_policy or RetryPolicy()
        self.timeout_policy = timeout_policy or TransferTimeoutPolicy()
        self._sleep = sleep
        self.client = client or self._build_client(
            endpoint_url=endpoint_url,
            region=region,
            access_key=access_key,
            secret_key=secret_key,
            session_token=session_token,
        )

    def _build_client(
        self,
        *,
        endpoint_url: str | None,
        region: str | None,
        access_key: str | None,
        secret_key: str | None,
        session_token: str | None,
    ) -> Any:
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:  # pragma: no cover - exercised in core-only environments
            raise RuntimeError("boto3 is required to construct an S3 storage client") from exc
        config = Config(
            connect_timeout=self.timeout_policy.connect_timeout_seconds,
            read_timeout=self.timeout_policy.read_timeout_seconds,
            retries={"max_attempts": 0},
        )
        return boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=session_token,
            config=config,
        )

    def _remote_key(self, key: str) -> str:
        normalized = normalize_storage_key(key)
        return f"{self.prefix}/{normalized}" if self.prefix else normalized

    def _logical_key(self, remote_key: str) -> str:
        if not self.prefix:
            return remote_key
        expected_prefix = f"{self.prefix}/"
        if not remote_key.startswith(expected_prefix):
            raise ValueError(f"remote key is outside configured prefix: {remote_key!r}")
        return remote_key[len(expected_prefix) :]

    def _is_retryable(self, exc: BaseException) -> bool:
        response = getattr(exc, "response", None)
        if isinstance(response, dict):
            metadata = response.get("ResponseMetadata", {})
            status = metadata.get("HTTPStatusCode")
            code = response.get("Error", {}).get("Code")
            if status in {408, 425, 429} or (isinstance(status, int) and status >= 500):
                return True
            if code in {"SlowDown", "RequestTimeout", "Throttling", "InternalError"}:
                return True
        return exc.__class__.__name__ in {
            "ConnectTimeoutError",
            "ConnectionClosedError",
            "EndpointConnectionError",
            "ReadTimeoutError",
        }

    def _call(self, operation: Callable[[], Any]) -> Any:
        return retry_call(
            operation,
            policy=self.retry_policy,
            is_retryable=self._is_retryable,
            sleep=self._sleep,
        )

    @staticmethod
    def _is_not_found(exc: BaseException) -> bool:
        response = getattr(exc, "response", None)
        if not isinstance(response, dict):
            return False
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        code = str(response.get("Error", {}).get("Code", ""))
        return status == 404 or code in {"404", "NoSuchKey", "NotFound"}

    def list(self, prefix: str = "") -> list[StorageObjectMetadata]:
        normalized_prefix = normalize_storage_key(prefix, allow_empty=True)
        logical_prefix = f"{self.prefix}/{normalized_prefix}" if self.prefix else normalized_prefix
        continuation: str | None = None
        objects: list[StorageObjectMetadata] = []
        while True:
            kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": logical_prefix}
            if continuation is not None:
                kwargs["ContinuationToken"] = continuation
            response = self._call(lambda kwargs=kwargs: self.client.list_objects_v2(**kwargs))
            for item in response.get("Contents", []):
                key = self._logical_key(str(item["Key"]))
                if is_temporary_storage_key(key):
                    continue
                last_modified = item.get("LastModified")
                objects.append(
                    StorageObjectMetadata(
                        key=key,
                        size=int(item.get("Size", 0)),
                        etag=_strip_etag(item.get("ETag")),
                        last_modified_epoch_seconds=_timestamp(last_modified),
                    )
                )
            if not response.get("IsTruncated"):
                break
            continuation = response.get("NextContinuationToken")
            if continuation is None:
                break
        return sorted(objects, key=lambda item: item.key)

    def exists(self, key: str) -> bool:
        try:
            self.head(key)
        except ObjectNotFoundError:
            return False
        return True

    def upload(
        self,
        source: str | Path,
        key: str,
        *,
        expected_sha256: str | None = None,
    ) -> StorageObjectMetadata:
        source_path = Path(source)
        if not source_path.is_file():
            raise ObjectNotFoundError(f"local upload source does not exist: {source_path}")
        if source_path.stat().st_size >= self.multipart_threshold_bytes:
            return self.multipart_upload(source_path, key, expected_sha256=expected_sha256)
        return self._singlepart_upload(source_path, key, expected_sha256=expected_sha256)

    def _singlepart_upload(
        self,
        source_path: Path,
        key: str,
        *,
        expected_sha256: str | None = None,
    ) -> StorageObjectMetadata:
        checksum = sha256_file(source_path)
        require_checksum(checksum, expected_sha256, context=str(source_path))
        temporary_key = temporary_storage_key(key)
        remote_temporary = self._remote_key(temporary_key)
        remote_final = self._remote_key(key)
        try:
            with source_path.open("rb") as handle:

                def put_temporary() -> Any:
                    handle.seek(0)
                    return self.client.put_object(
                        Bucket=self.bucket,
                        Key=remote_temporary,
                        Body=handle,
                        Metadata={"sha256": checksum},
                    )

                self._call(put_temporary)
            self._publish_temporary(remote_temporary, remote_final)
        finally:
            self._best_effort_delete_remote(remote_temporary)
        return self.head(key)

    def multipart_upload(
        self,
        source: str | Path,
        key: str,
        *,
        expected_sha256: str | None = None,
    ) -> StorageObjectMetadata:
        source_path = Path(source)
        if not source_path.is_file():
            raise ObjectNotFoundError(f"local upload source does not exist: {source_path}")
        if source_path.stat().st_size == 0:
            return self._singlepart_upload(source_path, key, expected_sha256=expected_sha256)
        checksum = sha256_file(source_path)
        require_checksum(checksum, expected_sha256, context=str(source_path))
        temporary_key = temporary_storage_key(key)
        remote_temporary = self._remote_key(temporary_key)
        remote_final = self._remote_key(key)
        upload_id: str | None = None
        try:
            created = self._call(
                lambda: self.client.create_multipart_upload(
                    Bucket=self.bucket,
                    Key=remote_temporary,
                    Metadata={"sha256": checksum},
                )
            )
            upload_id = str(created["UploadId"])
            parts: list[dict[str, Any]] = []
            with source_path.open("rb") as handle:
                part_number = 1
                while chunk := handle.read(self.multipart_part_size_bytes):
                    response = self._call(
                        lambda chunk=chunk, part_number=part_number: self.client.upload_part(
                            Bucket=self.bucket,
                            Key=remote_temporary,
                            UploadId=upload_id,
                            PartNumber=part_number,
                            Body=chunk,
                        )
                    )
                    parts.append({"ETag": response["ETag"], "PartNumber": part_number})
                    part_number += 1
            self._call(
                lambda: self.client.complete_multipart_upload(
                    Bucket=self.bucket,
                    Key=remote_temporary,
                    UploadId=upload_id,
                    MultipartUpload={"Parts": parts},
                )
            )
            upload_id = None
            self._publish_temporary(remote_temporary, remote_final)
        except BaseException:
            if upload_id is not None:
                with contextlib.suppress(BaseException):
                    self.client.abort_multipart_upload(
                        Bucket=self.bucket,
                        Key=remote_temporary,
                        UploadId=upload_id,
                    )
            raise
        finally:
            self._best_effort_delete_remote(remote_temporary)
        return self.head(key)

    def _publish_temporary(self, remote_temporary: str, remote_final: str) -> None:
        self._call(
            lambda: self.client.copy_object(
                Bucket=self.bucket,
                Key=remote_final,
                CopySource={"Bucket": self.bucket, "Key": remote_temporary},
                MetadataDirective="COPY",
            )
        )

    def download(
        self,
        key: str,
        destination: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> Path:
        response = self._call(
            lambda: self.client.get_object(Bucket=self.bucket, Key=self._remote_key(key))
        )
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = temporary_local_path(destination_path)
        digest = hashlib.sha256()
        try:
            with temporary.open("wb") as handle:
                for chunk in _body_chunks(response["Body"]):
                    digest.update(chunk)
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            actual = digest.hexdigest()
            stored = response.get("Metadata", {}).get("sha256")
            if stored is not None:
                require_checksum(actual, str(stored), context=key)
            require_checksum(actual, expected_sha256, context=key)
            temporary.replace(destination_path)
        finally:
            temporary.unlink(missing_ok=True)
        return destination_path

    def copy(
        self,
        source_key: str,
        destination_key: str,
        *,
        expected_sha256: str | None = None,
    ) -> StorageObjectMetadata:
        source = self.head(source_key)
        if expected_sha256 is not None and not self.verify_checksum(source_key, expected_sha256):
            require_checksum(source.checksum_sha256 or "", expected_sha256, context=source_key)
        temporary_key = temporary_storage_key(destination_key)
        remote_temporary = self._remote_key(temporary_key)
        remote_final = self._remote_key(destination_key)
        try:
            self._call(
                lambda: self.client.copy_object(
                    Bucket=self.bucket,
                    Key=remote_temporary,
                    CopySource={"Bucket": self.bucket, "Key": self._remote_key(source_key)},
                    MetadataDirective="COPY",
                )
            )
            self._publish_temporary(remote_temporary, remote_final)
        finally:
            self._best_effort_delete_remote(remote_temporary)
        return self.head(destination_key)

    def delete(self, key: str) -> None:
        self._call(lambda: self.client.delete_object(Bucket=self.bucket, Key=self._remote_key(key)))

    def head(self, key: str) -> StorageObjectMetadata:
        normalized = normalize_storage_key(key)
        try:
            response = self._call(
                lambda: self.client.head_object(
                    Bucket=self.bucket, Key=self._remote_key(normalized)
                )
            )
        except BaseException as exc:
            if self._is_not_found(exc):
                raise ObjectNotFoundError(f"storage object does not exist: {normalized}") from exc
            raise
        return StorageObjectMetadata(
            key=normalized,
            size=int(response.get("ContentLength", 0)),
            checksum_sha256=response.get("Metadata", {}).get("sha256"),
            etag=_strip_etag(response.get("ETag")),
            last_modified_epoch_seconds=_timestamp(response.get("LastModified")),
        )

    def verify_checksum(self, key: str, expected_sha256: str) -> bool:
        metadata = self.head(key)
        if metadata.checksum_sha256 is not None:
            return metadata.checksum_sha256.lower() == expected_sha256.lower()
        response = self._call(
            lambda: self.client.get_object(Bucket=self.bucket, Key=self._remote_key(key))
        )
        digest = hashlib.sha256()
        for chunk in _body_chunks(response["Body"]):
            digest.update(chunk)
        return digest.hexdigest().lower() == expected_sha256.lower()

    def _best_effort_delete_remote(self, remote_key: str) -> None:
        with contextlib.suppress(BaseException):
            self._call(lambda: self.client.delete_object(Bucket=self.bucket, Key=remote_key))


def _body_chunks(body: Any, *, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
    if hasattr(body, "iter_chunks"):
        yield from body.iter_chunks(chunk_size=chunk_size)
        return
    while chunk := body.read(chunk_size):
        yield chunk


def _strip_etag(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip('"')


def _timestamp(value: Any) -> float | None:
    if isinstance(value, datetime):
        return value.timestamp()
    return None
