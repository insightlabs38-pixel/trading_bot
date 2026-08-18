"""Common identifiers, hashing, runtime capture, and run-manifest utilities."""

from trading_bot.metadata.hashing import (
    config_sha256,
    model_config_id,
    model_config_to_canonical_json,
    sha256_text,
)
from trading_bot.metadata.identifiers import (
    CampaignId,
    CheckpointId,
    DatasetVersion,
    ModelConfigId,
    PredictionArtifactId,
    ProjectIdentifier,
    SplitVersion,
    TrialId,
)
from trading_bot.metadata.manifest import RunManifest, build_run_manifest
from trading_bot.metadata.runtime import (
    ContainerMetadata,
    EnvironmentMetadata,
    GitMetadata,
    PackageVersion,
    RuntimeMetadataError,
    capture_container_metadata,
    capture_environment_metadata,
    capture_git_metadata,
)

__all__ = [
    "CampaignId",
    "CheckpointId",
    "ContainerMetadata",
    "DatasetVersion",
    "EnvironmentMetadata",
    "GitMetadata",
    "ModelConfigId",
    "PackageVersion",
    "PredictionArtifactId",
    "ProjectIdentifier",
    "RunManifest",
    "RuntimeMetadataError",
    "SplitVersion",
    "TrialId",
    "build_run_manifest",
    "capture_container_metadata",
    "capture_environment_metadata",
    "capture_git_metadata",
    "config_sha256",
    "model_config_id",
    "model_config_to_canonical_json",
    "sha256_text",
]
