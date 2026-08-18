"""Immutable run-manifest construction from validated configuration and runtime metadata."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import field_validator

from trading_bot.config import AppConfig, config_to_canonical_json
from trading_bot.config.base import FrozenConfigModel
from trading_bot.metadata.hashing import config_sha256, model_config_id
from trading_bot.metadata.identifiers import (
    CampaignId,
    CheckpointId,
    DatasetVersion,
    ModelConfigId,
    PredictionArtifactId,
    SplitVersion,
    TrialId,
)
from trading_bot.metadata.runtime import (
    ContainerMetadata,
    EnvironmentMetadata,
    GitMetadata,
    capture_container_metadata,
    capture_environment_metadata,
    capture_git_metadata,
)


class RunManifest(FrozenConfigModel):
    """Immutable provenance snapshot for one configured execution."""

    schema_version: Literal[1] = 1
    created_at_utc: datetime
    dataset_version: DatasetVersion
    split_version: SplitVersion | None = None
    model_config_id: ModelConfigId
    campaign_id: CampaignId
    trial_id: TrialId | None = None
    parent_trial_id: TrialId | None = None
    checkpoint_id: CheckpointId | None = None
    prediction_artifact_id: PredictionArtifactId | None = None
    config_sha256: str
    config_canonical_json: str
    git: GitMetadata
    container: ContainerMetadata
    environment: EnvironmentMetadata
    seed: int
    precision: str
    compile_mode: str

    @field_validator("created_at_utc")
    @classmethod
    def normalize_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("created_at_utc must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("config_sha256")
    @classmethod
    def validate_config_hash(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("config_sha256 must be a lowercase SHA-256 hex digest")
        return value


def build_run_manifest(
    config: AppConfig,
    *,
    split_version: SplitVersion | str | None = None,
    trial_id: TrialId | str | None = None,
    parent_trial_id: TrialId | str | None = None,
    checkpoint_id: CheckpointId | str | None = None,
    prediction_artifact_id: PredictionArtifactId | str | None = None,
    git: GitMetadata | None = None,
    container: ContainerMetadata | None = None,
    environment: EnvironmentMetadata | None = None,
    created_at_utc: datetime | None = None,
    repo_path: str | Path = ".",
    environ: Mapping[str, str] | None = None,
) -> RunManifest:
    """Build a complete immutable manifest without requiring market data or a GPU."""
    return RunManifest(
        created_at_utc=datetime.now(UTC) if created_at_utc is None else created_at_utc,
        dataset_version=DatasetVersion(config.dataset.version),
        split_version=None if split_version is None else SplitVersion.model_validate(split_version),
        model_config_id=model_config_id(config.model),
        campaign_id=CampaignId(config.campaign.campaign_id),
        trial_id=None if trial_id is None else TrialId.model_validate(trial_id),
        parent_trial_id=(
            None if parent_trial_id is None else TrialId.model_validate(parent_trial_id)
        ),
        checkpoint_id=(
            None if checkpoint_id is None else CheckpointId.model_validate(checkpoint_id)
        ),
        prediction_artifact_id=(
            None
            if prediction_artifact_id is None
            else PredictionArtifactId.model_validate(prediction_artifact_id)
        ),
        config_sha256=config_sha256(config),
        config_canonical_json=config_to_canonical_json(config),
        git=capture_git_metadata(repo_path, environ) if git is None else git,
        container=capture_container_metadata(environ) if container is None else container,
        environment=capture_environment_metadata() if environment is None else environment,
        seed=config.training.seed,
        precision=config.training.precision,
        compile_mode=config.training.compile_mode,
    )
