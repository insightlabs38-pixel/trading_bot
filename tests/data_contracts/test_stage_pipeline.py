"""Tests for restartable Phase 3 stage manifests and success markers."""

from __future__ import annotations

from pathlib import Path

import pytest

from trading_bot.data.pipeline import (
    DataStage,
    StageArtifactDraft,
    StagePublicationError,
    StageRunner,
    StageRunSpec,
)
from trading_bot.storage import (
    ArtifactVerificationError,
    LocalStorageBackend,
    StorageObjectMetadata,
)


class RecordingBackend:
    """Transparent local backend wrapper that records durable upload order."""

    def __init__(self, root: Path) -> None:
        self.inner = LocalStorageBackend(root)
        self.uploaded_keys: list[str] = []

    def list(self, prefix: str = "") -> list[StorageObjectMetadata]:
        return self.inner.list(prefix)

    def exists(self, key: str) -> bool:
        return self.inner.exists(key)

    def upload(
        self,
        source: str | Path,
        key: str,
        *,
        expected_sha256: str | None = None,
    ) -> StorageObjectMetadata:
        result = self.inner.upload(source, key, expected_sha256=expected_sha256)
        self.uploaded_keys.append(key)
        return result

    def multipart_upload(
        self,
        source: str | Path,
        key: str,
        *,
        expected_sha256: str | None = None,
    ) -> StorageObjectMetadata:
        result = self.inner.multipart_upload(source, key, expected_sha256=expected_sha256)
        self.uploaded_keys.append(key)
        return result

    def download(
        self,
        key: str,
        destination: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> Path:
        return self.inner.download(key, destination, expected_sha256=expected_sha256)

    def copy(
        self,
        source_key: str,
        destination_key: str,
        *,
        expected_sha256: str | None = None,
    ) -> StorageObjectMetadata:
        return self.inner.copy(
            source_key,
            destination_key,
            expected_sha256=expected_sha256,
        )

    def delete(self, key: str) -> None:
        self.inner.delete(key)

    def head(self, key: str) -> StorageObjectMetadata:
        return self.inner.head(key)

    def verify_checksum(self, key: str, expected_sha256: str) -> bool:
        return self.inner.verify_checksum(key, expected_sha256)


def _spec(**overrides: object) -> StageRunSpec:
    values: dict[str, object] = {
        "dataset_version": "dataset-v1",
        "stage": DataStage.FEATURES,
        "stage_version": "features-v1",
        "upstream_ids": ("canonical-sha",),
        "producer_git_sha": "a" * 40,
        "producer_config_sha256": "b" * 64,
    }
    values.update(overrides)
    return StageRunSpec(**values)  # type: ignore[arg-type]


def _producer(call_count: list[int]):
    def produce(workspace: Path) -> tuple[StageArtifactDraft, ...]:
        call_count.append(1)
        first = workspace / "features.json"
        second = workspace / "index.json"
        first.write_text('{"rows":2}', encoding="utf-8")
        second.write_text('{"assets":["a","b"]}', encoding="utf-8")
        return (
            StageArtifactDraft(
                name="features.json",
                path=first,
                artifact_schema="feature_rows",
                artifact_version="1",
                row_count=2,
                metadata={"frequency": "5m"},
            ),
            StageArtifactDraft(
                name="metadata/index.json",
                path=second,
                artifact_schema="feature_index",
                artifact_version="1",
            ),
        )

    return produce


def test_stage_publishes_artifacts_manifests_then_success_and_reuses_completion(
    tmp_path: Path,
) -> None:
    backend = RecordingBackend(tmp_path / "store")
    runner = StageRunner(backend)
    calls: list[int] = []

    first = runner.run(_spec(), _producer(calls))
    assert not first.reused
    assert len(first.marker.artifacts) == 2
    assert backend.uploaded_keys[-1] == first.success_key
    assert first.success_key.endswith("/_SUCCESS.json")
    assert all(backend.exists(item.artifact_key) for item in first.marker.artifacts)
    assert all(backend.exists(item.manifest_key) for item in first.marker.artifacts)

    upload_count = len(backend.uploaded_keys)
    second = runner.run(_spec(), _producer(calls))
    assert second.reused
    assert second.marker == first.marker
    assert calls == [1]
    assert len(backend.uploaded_keys) == upload_count


def test_failed_producer_never_publishes_success_marker(tmp_path: Path) -> None:
    backend = RecordingBackend(tmp_path / "store")
    runner = StageRunner(backend)
    spec = _spec()

    def fail(workspace: Path) -> tuple[StageArtifactDraft, ...]:
        (workspace / "partial").write_bytes(b"partial")
        raise RuntimeError("producer failed")

    with pytest.raises(RuntimeError, match="producer failed"):
        runner.run(spec, fail)
    assert not backend.exists(runner.success_key(spec))
    assert backend.uploaded_keys == []


def test_completed_stage_fails_closed_when_artifact_is_corrupted(tmp_path: Path) -> None:
    root = tmp_path / "store"
    backend = RecordingBackend(root)
    runner = StageRunner(backend)
    result = runner.run(_spec(), _producer([]))
    artifact = result.marker.artifacts[0]
    (root / artifact.artifact_key).write_bytes(b"tampered")

    with pytest.raises(ArtifactVerificationError, match="mismatch"):
        runner.run(_spec(), _producer([]))


def test_completed_stage_is_bound_to_git_and_config_provenance(tmp_path: Path) -> None:
    runner = StageRunner(LocalStorageBackend(tmp_path / "store"))
    calls: list[int] = []
    runner.run(_spec(), _producer(calls))

    changed = _spec(producer_git_sha="c" * 40)
    with pytest.raises(StagePublicationError, match="producer Git SHA"):
        runner.run(changed, _producer(calls))
    assert calls == [1]


def test_stage_rejects_duplicate_names_external_files_and_reserved_metadata(
    tmp_path: Path,
) -> None:
    runner = StageRunner(LocalStorageBackend(tmp_path / "store"))

    def duplicate(workspace: Path) -> tuple[StageArtifactDraft, ...]:
        path = workspace / "file"
        path.write_bytes(b"x")
        return (
            StageArtifactDraft("same", path, "schema", "1"),
            StageArtifactDraft("same", path, "schema", "1"),
        )

    with pytest.raises(StagePublicationError, match="names must be unique"):
        runner.run(_spec(stage_version="duplicate"), duplicate)

    outside = tmp_path / "outside"
    outside.write_bytes(b"x")

    def external(workspace: Path) -> tuple[StageArtifactDraft, ...]:
        return (StageArtifactDraft("outside", outside, "schema", "1"),)

    with pytest.raises(StagePublicationError, match="inside the provided workspace"):
        runner.run(_spec(stage_version="external"), external)

    def reserved(workspace: Path) -> tuple[StageArtifactDraft, ...]:
        path = workspace / "file"
        path.write_bytes(b"x")
        return (
            StageArtifactDraft(
                "file",
                path,
                "schema",
                "1",
                metadata={"dataset_version": "forged"},
            ),
        )

    with pytest.raises(StagePublicationError, match="reserved field"):
        runner.run(_spec(stage_version="reserved"), reserved)


def test_stage_spec_normalizes_safe_identifiers_and_upstream_ids() -> None:
    spec = StageRunSpec(
        dataset_version=" dataset-v1 ",
        stage=DataStage.PACKED,
        stage_version=" pack-v1 ",
        upstream_ids=(" split-v1 ",),
    )
    assert spec.dataset_version == "dataset-v1"
    assert spec.stage_version == "pack-v1"
    assert spec.upstream_ids == ("split-v1",)
