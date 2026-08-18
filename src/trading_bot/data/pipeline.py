"""Restartable Phase 3 stage publication with manifest-last success semantics."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from trading_bot.storage import (
    ArtifactManifest,
    StorageBackend,
    build_artifact_manifest,
    load_artifact_manifest,
    manifest_key_for,
    normalize_storage_key,
    sha256_file,
    verify_artifact_manifest,
    write_artifact_manifest,
)


class StagePublicationError(RuntimeError):
    """Raised when a data stage cannot be safely published or resumed."""


class DataStage(StrEnum):
    RAW = "00_raw"
    VALIDATED = "01_validated"
    SECURITY_MASTER = "02_security_master"
    CANONICAL = "03_adjusted_canonical"
    RESAMPLED = "04_resampled"
    UNIVERSE = "05_point_in_time_universe"
    FEATURES = "06_features"
    LABELS = "07_labels"
    SPLITS = "08_immutable_splits"
    PACKED = "09_packed_training_data"


@dataclass(frozen=True, slots=True)
class StageRunSpec:
    """Versioned identity and provenance inputs for one restartable stage run."""

    dataset_version: str
    stage: DataStage
    stage_version: str
    upstream_ids: tuple[str, ...] = ()
    producer_git_sha: str | None = None
    producer_config_sha256: str | None = None

    def __post_init__(self) -> None:
        _safe_identifier(self.dataset_version, "dataset_version")
        _safe_identifier(self.stage_version, "stage_version")
        if any(not item.strip() for item in self.upstream_ids):
            raise ValueError("upstream_ids must not contain blank identifiers")
        if len(set(self.upstream_ids)) != len(self.upstream_ids):
            raise ValueError("upstream_ids must be unique")


@dataclass(frozen=True, slots=True)
class StageArtifactDraft:
    """One local file produced inside the temporary stage workspace."""

    name: str
    path: Path
    artifact_schema: str
    artifact_version: str
    row_count: int | None = None
    tensor_shape: tuple[int, ...] | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = normalize_storage_key(self.name)
        object.__setattr__(self, "name", normalized)
        if normalized == "_SUCCESS.json" or normalized.endswith("/_SUCCESS.json"):
            raise ValueError("artifact name must not use the reserved success-marker name")
        if not self.artifact_schema.strip() or not self.artifact_version.strip():
            raise ValueError("artifact schema/version must not be blank")
        if self.row_count is not None and self.row_count < 0:
            raise ValueError("row_count must be non-negative")
        if self.tensor_shape is not None and any(value < 0 for value in self.tensor_shape):
            raise ValueError("tensor_shape dimensions must be non-negative")


class StageArtifactRef(BaseModel):
    """Durable reference to one verified stage artifact and its manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    name: str
    artifact_key: str
    manifest_key: str
    manifest_sha256: str
    artifact_sha256: str
    size_bytes: int = Field(ge=0)

    @field_validator("name", "artifact_key", "manifest_key")
    @classmethod
    def normalize_key_fields(cls, value: str) -> str:
        return normalize_storage_key(value)

    @field_validator("manifest_sha256", "artifact_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        normalized = value.lower()
        if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
            raise ValueError("SHA-256 values must be 64 hexadecimal characters")
        return normalized


class StageSuccessMarker(BaseModel):
    """Published last; its existence means every referenced artifact verifies."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    dataset_version: str
    stage: DataStage
    stage_version: str
    upstream_ids: tuple[str, ...] = ()
    artifacts: tuple[StageArtifactRef, ...]

    @field_validator("dataset_version", "stage_version")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        return _safe_identifier(value, "stage marker identifier")

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def marker_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class StageRunResult:
    success_key: str
    marker: StageSuccessMarker
    reused: bool


StageProducer = Callable[[Path], Iterable[StageArtifactDraft]]


class StageRunner:
    """Run one stage in scratch space and publish verified immutable outputs before success."""

    def __init__(self, backend: StorageBackend, *, root_prefix: str = "datasets") -> None:
        self.backend = backend
        self.root_prefix = normalize_storage_key(root_prefix)

    def run(self, spec: StageRunSpec, producer: StageProducer) -> StageRunResult:
        success_key = self.success_key(spec)
        if self.backend.exists(success_key):
            marker = self._load_and_verify_success(success_key, spec)
            return StageRunResult(success_key, marker, reused=True)

        with tempfile.TemporaryDirectory(prefix="trading-bot-stage-") as directory:
            workspace = Path(directory)
            drafts = tuple(producer(workspace))
            if not drafts:
                raise StagePublicationError("stage producer returned no artifacts")
            _validate_drafts(workspace, drafts)
            refs = tuple(self._publish_artifact(spec, draft) for draft in drafts)

            marker = StageSuccessMarker(
                dataset_version=spec.dataset_version,
                stage=spec.stage,
                stage_version=spec.stage_version,
                upstream_ids=spec.upstream_ids,
                artifacts=refs,
            )
            if self.backend.exists(success_key):
                existing = self._load_and_verify_success(success_key, spec)
                if existing != marker:
                    raise StagePublicationError(
                        "concurrent stage publication produced a different success marker"
                    )
                return StageRunResult(success_key, existing, reused=True)
            self._write_success_marker(success_key, marker, workspace)

        verified = self._load_and_verify_success(success_key, spec)
        return StageRunResult(success_key, verified, reused=False)

    def success_key(self, spec: StageRunSpec) -> str:
        return normalize_storage_key(f"{self._base_key(spec)}/_SUCCESS.json")

    def _base_key(self, spec: StageRunSpec) -> str:
        return normalize_storage_key(
            f"{self.root_prefix}/{spec.dataset_version}/{spec.stage.value}/{spec.stage_version}"
        )

    def _publish_artifact(
        self,
        spec: StageRunSpec,
        draft: StageArtifactDraft,
    ) -> StageArtifactRef:
        checksum = sha256_file(draft.path)
        artifact_key = normalize_storage_key(
            f"{self._base_key(spec)}/artifacts/{checksum}/{draft.name}"
        )
        manifest_key = manifest_key_for(artifact_key)
        metadata = dict(draft.metadata)
        metadata.update(
            {
                "artifact_name": draft.name,
                "dataset_version": spec.dataset_version,
                "stage_version": spec.stage_version,
            }
        )

        if self.backend.exists(artifact_key):
            if not self.backend.verify_checksum(artifact_key, checksum):
                raise StagePublicationError(
                    f"content-addressed artifact checksum conflict at {artifact_key}"
                )
        else:
            self.backend.upload(draft.path, artifact_key, expected_sha256=checksum)

        if self.backend.exists(manifest_key):
            manifest = load_artifact_manifest(self.backend, manifest_key)
            verify_artifact_manifest(self.backend, manifest)
            _assert_manifest_matches(manifest, spec, draft, checksum)
        else:
            manifest = build_artifact_manifest(
                self.backend,
                artifact_key,
                artifact_schema=draft.artifact_schema,
                artifact_version=draft.artifact_version,
                producer_git_sha=spec.producer_git_sha,
                producer_config_sha256=spec.producer_config_sha256,
                row_count=draft.row_count,
                tensor_shape=draft.tensor_shape,
                generation_stage=spec.stage.value,
                upstream_ids=spec.upstream_ids,
                metadata=metadata,
            )
            verify_artifact_manifest(self.backend, manifest)
            write_artifact_manifest(self.backend, manifest, manifest_key=manifest_key)

        return StageArtifactRef(
            name=draft.name,
            artifact_key=artifact_key,
            manifest_key=manifest_key,
            manifest_sha256=manifest.manifest_sha256(),
            artifact_sha256=manifest.checksum,
            size_bytes=manifest.size_bytes,
        )

    def _write_success_marker(
        self,
        success_key: str,
        marker: StageSuccessMarker,
        workspace: Path,
    ) -> None:
        payload = marker.canonical_json().encode("utf-8")
        path = workspace / "_SUCCESS.json"
        path.write_bytes(payload)
        self.backend.upload(
            path,
            success_key,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )

    def _load_and_verify_success(
        self,
        success_key: str,
        spec: StageRunSpec,
    ) -> StageSuccessMarker:
        with tempfile.TemporaryDirectory(prefix="trading-bot-stage-marker-") as directory:
            path = Path(directory) / "_SUCCESS.json"
            try:
                self.backend.download(success_key, path)
                payload = json.loads(path.read_text(encoding="utf-8"))
                marker = StageSuccessMarker.model_validate(payload)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                raise StagePublicationError(f"invalid stage success marker {success_key}: {exc}") from exc

        if (
            marker.dataset_version != spec.dataset_version
            or marker.stage != spec.stage
            or marker.stage_version != spec.stage_version
            or marker.upstream_ids != spec.upstream_ids
        ):
            raise StagePublicationError("stage success marker does not match requested stage spec")
        if not marker.artifacts:
            raise StagePublicationError("stage success marker must reference at least one artifact")

        names: set[str] = set()
        for reference in marker.artifacts:
            if reference.name in names:
                raise StagePublicationError("stage success marker contains duplicate artifact names")
            names.add(reference.name)
            manifest = load_artifact_manifest(self.backend, reference.manifest_key)
            verify_artifact_manifest(self.backend, manifest)
            if manifest.manifest_sha256() != reference.manifest_sha256:
                raise StagePublicationError("stage artifact manifest hash does not match success marker")
            if manifest.artifact_key != reference.artifact_key:
                raise StagePublicationError("stage artifact key does not match success marker")
            if manifest.checksum != reference.artifact_sha256:
                raise StagePublicationError("stage artifact checksum does not match success marker")
            if manifest.size_bytes != reference.size_bytes:
                raise StagePublicationError("stage artifact size does not match success marker")
        return marker


def _validate_drafts(workspace: Path, drafts: tuple[StageArtifactDraft, ...]) -> None:
    names = [draft.name for draft in drafts]
    if len(set(names)) != len(names):
        raise StagePublicationError("stage artifact names must be unique")
    root = workspace.resolve()
    for draft in drafts:
        path = draft.path.resolve()
        if root not in path.parents or not path.is_file():
            raise StagePublicationError(
                f"stage artifact {draft.name!r} must be a file inside the provided workspace"
            )


def _assert_manifest_matches(
    manifest: ArtifactManifest,
    spec: StageRunSpec,
    draft: StageArtifactDraft,
    checksum: str,
) -> None:
    if manifest.checksum != checksum:
        raise StagePublicationError("existing stage manifest checksum does not match local output")
    if manifest.artifact_schema != draft.artifact_schema:
        raise StagePublicationError("existing stage manifest schema does not match stage output")
    if manifest.artifact_version != draft.artifact_version:
        raise StagePublicationError("existing stage manifest version does not match stage output")
    if manifest.generation_stage != spec.stage.value:
        raise StagePublicationError("existing stage manifest generation stage does not match")
    if manifest.upstream_ids != spec.upstream_ids:
        raise StagePublicationError("existing stage manifest upstream IDs do not match")
    if manifest.row_count != draft.row_count or manifest.tensor_shape != draft.tensor_shape:
        raise StagePublicationError("existing stage manifest shape/count does not match stage output")


def _safe_identifier(value: str, field_name: str) -> str:
    normalized = value.strip()
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    if not normalized or any(character not in allowed for character in normalized):
        raise ValueError(f"{field_name} must use only letters, digits, '-', '_', or '.'")
    return normalized
