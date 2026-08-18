"""Typed identifiers used to preserve experiment and artifact lineage."""

from __future__ import annotations

import re
from typing import ClassVar

from pydantic import ConfigDict, RootModel, field_validator

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ProjectIdentifier(RootModel[str]):
    """Immutable filesystem/object-store-safe project identifier."""

    model_config = ConfigDict(frozen=True)
    kind: ClassVar[str] = "identifier"

    @field_validator("root")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if not _IDENTIFIER_PATTERN.fullmatch(value):
            raise ValueError(
                f"{cls.kind} must match {_IDENTIFIER_PATTERN.pattern!r}; "
                "use only letters, digits, dot, underscore, or hyphen"
            )
        return value

    def __str__(self) -> str:
        return self.root


class DatasetVersion(ProjectIdentifier):
    """Immutable identifier for a materialized dataset version."""

    kind = "dataset version"


class SplitVersion(ProjectIdentifier):
    """Immutable identifier for a train/validation/holdout split definition."""

    kind = "split version"


class ModelConfigId(ProjectIdentifier):
    """Content-derived identifier for a model configuration."""

    kind = "model configuration ID"


class TrialId(ProjectIdentifier):
    """Immutable identifier for one attempted training/evaluation trial."""

    kind = "trial ID"


class CampaignId(ProjectIdentifier):
    """Immutable identifier for a campaign."""

    kind = "campaign ID"


class CheckpointId(ProjectIdentifier):
    """Immutable identifier for a checkpoint artifact."""

    kind = "checkpoint ID"


class PredictionArtifactId(ProjectIdentifier):
    """Immutable identifier for a prediction artifact."""

    kind = "prediction artifact ID"
