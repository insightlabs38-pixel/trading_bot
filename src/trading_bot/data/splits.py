"""Immutable chronological split manifests with protected final-holdout access."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SplitManifestError(ValueError):
    """Raised when chronological split definitions are invalid."""


class FinalHoldoutAccessError(PermissionError):
    """Raised when routine code attempts to inspect the protected final holdout."""


class FinalHoldoutAccess(StrEnum):
    ROUTINE_RESEARCH = "routine_research"
    FINAL_EVALUATION = "final_evaluation"


class DateRange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start: date
    end: date

    @model_validator(mode="after")
    def validate_range(self) -> DateRange:
        if self.end < self.start:
            raise ValueError("date range end cannot precede start")
        return self

    def contains(self, value: date) -> bool:
        return self.start <= value <= self.end


class WalkForwardFold(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fold_id: str = Field(min_length=1)
    train: DateRange
    validation: DateRange

    @field_validator("fold_id")
    @classmethod
    def normalize_fold_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("fold_id must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_chronology(self) -> WalkForwardFold:
        if self.train.end >= self.validation.start:
            raise ValueError("training period must end before validation begins")
        return self


class SplitManifest(BaseModel):
    """Versioned split contract independent of model/training code."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    split_version: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    folds: tuple[WalkForwardFold, ...]
    final_holdout_id: str = Field(min_length=1)
    final_holdout: DateRange

    @field_validator("split_version", "dataset_version", "final_holdout_id")
    @classmethod
    def normalize_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("split manifest identifiers must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_manifest(self) -> SplitManifest:
        if not self.folds:
            raise ValueError("at least one walk-forward fold is required")
        fold_ids = [fold.fold_id for fold in self.folds]
        if len(set(fold_ids)) != len(fold_ids):
            raise ValueError("fold IDs must be unique")
        if self.final_holdout_id in set(fold_ids):
            raise ValueError("final_holdout_id must be distinct from routine fold IDs")
        ordered = sorted(self.folds, key=lambda fold: fold.validation.start)
        if tuple(ordered) != self.folds:
            raise ValueError("folds must be stored in chronological validation order")
        previous_validation_end: date | None = None
        for fold in self.folds:
            if (
                previous_validation_end is not None
                and fold.validation.start <= previous_validation_end
            ):
                raise ValueError("validation periods must not overlap or move backward")
            previous_validation_end = fold.validation.end
        latest_routine_date = max(
            max(fold.train.end, fold.validation.end) for fold in self.folds
        )
        if self.final_holdout.start <= latest_routine_date:
            raise ValueError("final holdout must begin after all routine fold periods")
        return self

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def split_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def fold(self, fold_id: str) -> WalkForwardFold:
        for fold in self.folds:
            if fold.fold_id == fold_id:
                return fold
        raise KeyError(fold_id)

    def final_holdout_range(
        self,
        *,
        access: FinalHoldoutAccess = FinalHoldoutAccess.ROUTINE_RESEARCH,
    ) -> DateRange:
        if access is not FinalHoldoutAccess.FINAL_EVALUATION:
            raise FinalHoldoutAccessError(
                "final holdout is inaccessible in routine research/search mode"
            )
        return self.final_holdout

    def routine_partition(self, value: date) -> tuple[str, str] | None:
        """Identify only train/validation partitions; final holdout is intentionally invisible."""
        for fold in self.folds:
            if fold.train.contains(value):
                return fold.fold_id, "train"
            if fold.validation.contains(value):
                return fold.fold_id, "validation"
        return None
