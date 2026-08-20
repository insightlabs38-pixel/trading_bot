"""Unit tests for interchangeable local and S3-compatible storage backends."""

from __future__ import annotations

import hashlib
import io
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from trading_bot.storage import (
    ChecksumMismatchError,
    LocalStorageBackend,
    ObjectNotFoundError,
    RetryPolicy,
    S3StorageBackend,
    TransferTimeoutPolicy,
    UnsafeStorageKeyError,
    normalize_storage_key,
    sha256_file,
)


class FakeClientError(Exception):
    def __init__(self, code: str, status: int) -> None:
        super().__init__(code)
        self.response = {
            "Error": {"Code": code},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }


class EndpointConnectionError(Exception):
    pass


class FakeBody(io.BytesIO):
    def iter_chunks(self, chunk_size: int) -> Any:
        while chunk := self.read(chunk_size):
            yield chunk


class InMemoryS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], tuple[bytes, dict[str, str]]] = {}
        self.multipart: dict[str, dict[str, Any]] = {}
        self.next_upload = 1
        self.put_failures_remaining = 0
        self.put_calls = 0

    def put_object(self, *, Bucket: str, Key: str, Body: Any, Metadata: dict[str, str]) -> dict:
        self.put_calls += 1
        if self.put_failures_remaining:
            self.put_failures_remaining -= 1
            raise EndpointConnectionError("transient")
        data = Body.read() if hasattr(Body, "read") else bytes(Body)
        self.objects[(Bucket, Key)] = (data, dict(Metadata))
        return {"ETag": hashlib.md5(data).hexdigest()}

    def head_object(self, *, Bucket: str, Key: str) -> dict:
        try:
            data, metadata = self.objects[(Bucket, Key)]
        except KeyError as exc:
            raise FakeClientError("NoSuchKey", 404) from exc
        return {
            "ContentLength": len(data),
            "Metadata": dict(metadata),
            "ETag": f'"{hashlib.md5(data).hexdigest()}"',
            "LastModified": datetime(2026, 8, 18, tzinfo=UTC),
        }

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        try:
            data, metadata = self.objects[(Bucket, Key)]
        except KeyError as exc:
            raise FakeClientError("NoSuchKey", 404) from exc
        return {"Body": FakeBody(data), "Metadata": dict(metadata)}

    def delete_object(self, *, Bucket: str, Key: str) -> dict:
        self.objects.pop((Bucket, Key), None)
        return {}

    def copy_object(
        self,
        *,
        Bucket: str,
        Key: str,
        CopySource: dict[str, str],
        MetadataDirective: str,
    ) -> dict:
        del MetadataDirective
        source = (CopySource["Bucket"], CopySource["Key"])
        data, metadata = self.objects[source]
        self.objects[(Bucket, Key)] = (data, dict(metadata))
        return {"CopyObjectResult": {"ETag": hashlib.md5(data).hexdigest()}}

    def list_objects_v2(self, **kwargs: Any) -> dict:
        bucket = kwargs["Bucket"]
        prefix = kwargs.get("Prefix", "")
        contents = []
        for (object_bucket, key), (data, _metadata) in sorted(self.objects.items()):
            if object_bucket == bucket and key.startswith(prefix):
                contents.append(
                    {
                        "Key": key,
                        "Size": len(data),
                        "ETag": f'"{hashlib.md5(data).hexdigest()}"',
                        "LastModified": datetime(2026, 8, 18, tzinfo=UTC),
                    }
                )
        return {"Contents": contents, "IsTruncated": False}

    def create_multipart_upload(self, *, Bucket: str, Key: str, Metadata: dict[str, str]) -> dict:
        upload_id = f"u{self.next_upload}"
        self.next_upload += 1
        self.multipart[upload_id] = {
            "bucket": Bucket,
            "key": Key,
            "metadata": dict(Metadata),
            "parts": {},
        }
        return {"UploadId": upload_id}

    def upload_part(
        self,
        *,
        Bucket: str,
        Key: str,
        UploadId: str,
        PartNumber: int,
        Body: bytes,
    ) -> dict:
        record = self.multipart[UploadId]
        assert record["bucket"] == Bucket and record["key"] == Key
        record["parts"][PartNumber] = bytes(Body)
        return {"ETag": f"part-{PartNumber}"}

    def complete_multipart_upload(
        self,
        *,
        Bucket: str,
        Key: str,
        UploadId: str,
        MultipartUpload: dict[str, Any],
    ) -> dict:
        del MultipartUpload
        record = self.multipart.pop(UploadId)
        data = b"".join(record["parts"][part] for part in sorted(record["parts"]))
        self.objects[(Bucket, Key)] = (data, record["metadata"])
        return {"ETag": "multipart"}

    def abort_multipart_upload(self, **kwargs: Any) -> dict:
        self.multipart.pop(kwargs["UploadId"], None)
        return {}


def write_file(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_storage_keys_reject_escape_and_absolute_paths() -> None:
    assert normalize_storage_key("datasets/a.parquet") == "datasets/a.parquet"
    for value in ("../secret", "/absolute", "a/../b", ""):
        with pytest.raises(UnsafeStorageKeyError):
            normalize_storage_key(value)


def test_local_backend_round_trip_and_operations(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path / "objects")
    source = tmp_path / "source.bin"
    checksum = write_file(source, b"alpha" * 1000)

    uploaded = backend.upload(source, "datasets/source.bin", expected_sha256=checksum)
    assert uploaded.size == source.stat().st_size
    assert uploaded.checksum_sha256 == checksum
    assert backend.exists("datasets/source.bin")
    assert [item.key for item in backend.list("datasets")] == ["datasets/source.bin"]
    assert backend.verify_checksum("datasets/source.bin", checksum)

    copied = backend.copy("datasets/source.bin", "copies/source.bin", expected_sha256=checksum)
    assert copied.checksum_sha256 == checksum

    destination = tmp_path / "downloaded.bin"
    assert (
        backend.download("copies/source.bin", destination, expected_sha256=checksum).read_bytes()
        == source.read_bytes()
    )

    backend.delete("datasets/source.bin")
    assert not backend.exists("datasets/source.bin")
    backend.delete("datasets/source.bin")


def test_local_multipart_uses_same_atomic_contract(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path / "objects")
    source = tmp_path / "large.bin"
    checksum = write_file(source, b"0123456789" * 1_000_000)
    result = backend.multipart_upload(source, "large/object.bin", expected_sha256=checksum)
    assert result.checksum_sha256 == checksum
    assert all(".trading-bot-tmp-" not in item.key for item in backend.list())


def test_local_checksum_mismatch_never_publishes(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path / "objects")
    source = tmp_path / "source.bin"
    write_file(source, b"payload")
    with pytest.raises(ChecksumMismatchError):
        backend.upload(source, "bad.bin", expected_sha256="0" * 64)
    assert not backend.exists("bad.bin")


def test_local_download_mismatch_does_not_replace_existing_destination(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path / "objects")
    source = tmp_path / "source.bin"
    write_file(source, b"payload")
    backend.upload(source, "source.bin")
    destination = tmp_path / "destination.bin"
    destination.write_bytes(b"keep-me")
    with pytest.raises(ChecksumMismatchError):
        backend.download("source.bin", destination, expected_sha256="0" * 64)
    assert destination.read_bytes() == b"keep-me"


def test_local_missing_object_raises_clear_error(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path / "objects")
    with pytest.raises(ObjectNotFoundError):
        backend.head("missing.bin")


def test_s3_backend_round_trip_and_atomic_temp_cleanup(tmp_path: Path) -> None:
    client = InMemoryS3Client()
    backend = S3StorageBackend(bucket="research", prefix="project", client=client)
    source = tmp_path / "source.bin"
    checksum = write_file(source, b"s3-data" * 1000)

    metadata = backend.upload(source, "datasets/source.bin", expected_sha256=checksum)
    assert metadata.checksum_sha256 == checksum
    assert backend.exists("datasets/source.bin")
    assert backend.verify_checksum("datasets/source.bin", checksum)
    assert [item.key for item in backend.list("datasets")] == ["datasets/source.bin"]
    assert all(".trading-bot-tmp-" not in key for _bucket, key in client.objects)

    copied = backend.copy("datasets/source.bin", "copies/source.bin", expected_sha256=checksum)
    assert copied.checksum_sha256 == checksum
    destination = tmp_path / "download.bin"
    backend.download("copies/source.bin", destination, expected_sha256=checksum)
    assert destination.read_bytes() == source.read_bytes()
    backend.delete("datasets/source.bin")
    assert not backend.exists("datasets/source.bin")


def test_s3_multipart_upload(tmp_path: Path) -> None:
    client = InMemoryS3Client()
    backend = S3StorageBackend(
        bucket="research",
        client=client,
        multipart_part_size_bytes=5 * 1024 * 1024,
    )
    source = tmp_path / "multipart.bin"
    checksum = write_file(source, b"a" * (5 * 1024 * 1024) + b"tail")
    result = backend.multipart_upload(source, "large.bin", expected_sha256=checksum)
    assert result.size == source.stat().st_size
    assert result.checksum_sha256 == checksum
    assert not client.multipart


def test_s3_retries_transient_failures_without_infinite_loop(tmp_path: Path) -> None:
    client = InMemoryS3Client()
    client.put_failures_remaining = 2
    delays: list[float] = []
    backend = S3StorageBackend(
        bucket="research",
        client=client,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_delay_seconds=0.1,
            multiplier=2,
            max_delay_seconds=1,
        ),
        sleep=delays.append,
    )
    source = tmp_path / "retry.bin"
    write_file(source, b"retry")
    backend.upload(source, "retry.bin")
    assert client.put_calls == 3
    assert delays == [0.1, 0.2]


def test_s3_checksum_mismatch_is_detected(tmp_path: Path) -> None:
    client = InMemoryS3Client()
    backend = S3StorageBackend(bucket="research", client=client)
    source = tmp_path / "source.bin"
    write_file(source, b"payload")
    with pytest.raises(ChecksumMismatchError):
        backend.upload(source, "bad.bin", expected_sha256="f" * 64)
    assert not backend.exists("bad.bin")


def test_s3_real_client_receives_timeout_policy() -> None:
    backend = S3StorageBackend(
        bucket="research",
        endpoint_url="https://example.invalid",
        region="us-east-1",
        access_key="test",
        secret_key="test",
        timeout_policy=TransferTimeoutPolicy(
            connect_timeout_seconds=3.5,
            read_timeout_seconds=17.0,
        ),
    )
    assert backend.client.meta.config.connect_timeout == 3.5
    assert backend.client.meta.config.read_timeout == 17.0
    assert backend.client.meta.config.retries["total_max_attempts"] == 1


def test_same_s3_backend_can_represent_separate_staging_namespace() -> None:
    client = InMemoryS3Client()
    durable = S3StorageBackend(bucket="cold", prefix="production", client=client)
    staging = S3StorageBackend(bucket="staging", prefix="transfer", client=client)
    assert durable.bucket == "cold" and durable.prefix == "production"
    assert staging.bucket == "staging" and staging.prefix == "transfer"


def test_sha256_file_is_streaming_and_stable(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    expected = write_file(path, b"hash-me" * 10000)
    assert sha256_file(path, chunk_size=7) == expected


def test_s3_upload_retry_rewinds_stream_after_partial_read(tmp_path: Path) -> None:
    class PartialReadFailureClient(InMemoryS3Client):
        def put_object(self, *, Bucket: str, Key: str, Body: Any, Metadata: dict[str, str]) -> dict:
            self.put_calls += 1
            if self.put_calls == 1:
                Body.read(3)
                raise EndpointConnectionError("failed after partial body consumption")
            data = Body.read()
            self.objects[(Bucket, Key)] = (data, dict(Metadata))
            return {"ETag": "ok"}

    client = PartialReadFailureClient()
    backend = S3StorageBackend(
        bucket="research",
        client=client,
        retry_policy=RetryPolicy(max_attempts=2, initial_delay_seconds=0),
    )
    source = tmp_path / "source.bin"
    checksum = write_file(source, b"complete-payload")
    result = backend.upload(source, "source.bin", expected_sha256=checksum)
    assert result.size == len(b"complete-payload")
    assert backend.verify_checksum("source.bin", checksum)


def test_s3_upload_automatically_uses_multipart_above_threshold(tmp_path: Path) -> None:
    client = InMemoryS3Client()
    backend = S3StorageBackend(
        bucket="research",
        client=client,
        multipart_threshold_bytes=5 * 1024 * 1024,
        multipart_part_size_bytes=5 * 1024 * 1024,
    )
    source = tmp_path / "large.bin"
    checksum = write_file(source, b"z" * (5 * 1024 * 1024 + 1))
    result = backend.upload(source, "large.bin", expected_sha256=checksum)
    assert result.size == source.stat().st_size
    assert client.put_calls == 0


def test_empty_explicit_multipart_falls_back_to_atomic_put(tmp_path: Path) -> None:
    client = InMemoryS3Client()
    backend = S3StorageBackend(bucket="research", client=client)
    source = tmp_path / "empty.bin"
    checksum = write_file(source, b"")
    result = backend.multipart_upload(source, "empty.bin", expected_sha256=checksum)
    assert result.size == 0
    assert client.put_calls == 1


def test_factory_builds_local_and_s3_without_exposing_secret_wrapper(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from trading_bot.storage import create_storage_backend

    local_config = SimpleNamespace(
        backend="local",
        root_path=str(tmp_path / "local"),
    )
    assert isinstance(create_storage_backend(local_config), LocalStorageBackend)

    class Secret:
        def __init__(self, value: str) -> None:
            self.value = value

        def get_secret_value(self) -> str:
            return self.value

    s3_config = SimpleNamespace(
        backend="s3",
        root_path=None,
        bucket="bucket",
        prefix="prefix",
        endpoint_url="https://example.invalid",
        region="us-east-1",
        access_key=Secret("access"),
        secret_key=Secret("secret"),
        session_token=None,
        multipart_threshold_mb=128,
    )
    fake = InMemoryS3Client()
    backend = create_storage_backend(s3_config, client=fake)
    assert isinstance(backend, S3StorageBackend)
    assert backend.bucket == "bucket"
