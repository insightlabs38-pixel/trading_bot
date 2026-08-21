"""CPU-safe scheduler state and immutable trial contracts for Phase 11."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, JsonValue, model_validator

from trading_bot.campaign.search_space import CampaignStage
from trading_bot.config.base import FrozenConfigModel


class CampaignState(StrEnum):
    BOOTSTRAP = "BOOTSTRAP"
    CALIBRATION = "CALIBRATION"
    SCREENING = "SCREENING"
    PROMOTION = "PROMOTION"
    OBJECTIVE_SEARCH = "OBJECTIVE_SEARCH"
    FINALISTS = "FINALISTS"
    DRAIN = "DRAIN"
    COMPLETE = "COMPLETE"


class TrialState(StrEnum):
    PENDING = "PENDING"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    EVALUATING = "EVALUATING"
    SYNCING = "SYNCING"
    COMPLETE = "COMPLETE"
    PRUNED = "PRUNED"
    RETRYABLE_FAILURE = "RETRYABLE_FAILURE"
    TERMINAL_FAILURE = "TERMINAL_FAILURE"
    INTERRUPTED = "INTERRUPTED"


class LaunchPriority(StrEnum):
    MANDATORY = "MANDATORY"
    OPTIONAL = "OPTIONAL"
    FINALIST = "FINALIST"


_CAMPAIGN_TRANSITIONS: dict[CampaignState, frozenset[CampaignState]] = {
    CampaignState.BOOTSTRAP: frozenset({CampaignState.CALIBRATION, CampaignState.DRAIN}),
    CampaignState.CALIBRATION: frozenset({CampaignState.SCREENING, CampaignState.DRAIN}),
    CampaignState.SCREENING: frozenset({CampaignState.PROMOTION, CampaignState.DRAIN}),
    CampaignState.PROMOTION: frozenset({CampaignState.OBJECTIVE_SEARCH, CampaignState.DRAIN}),
    CampaignState.OBJECTIVE_SEARCH: frozenset({CampaignState.FINALISTS, CampaignState.DRAIN}),
    CampaignState.FINALISTS: frozenset({CampaignState.DRAIN}),
    CampaignState.DRAIN: frozenset({CampaignState.COMPLETE}),
    CampaignState.COMPLETE: frozenset(),
}

_TRIAL_TRANSITIONS: dict[TrialState, frozenset[TrialState]] = {
    TrialState.PENDING: frozenset({TrialState.STARTING, TrialState.PRUNED, TrialState.INTERRUPTED}),
    TrialState.STARTING: frozenset(
        {
            TrialState.RUNNING,
            TrialState.RETRYABLE_FAILURE,
            TrialState.TERMINAL_FAILURE,
            TrialState.INTERRUPTED,
        }
    ),
    TrialState.RUNNING: frozenset(
        {
            TrialState.EVALUATING,
            TrialState.PRUNED,
            TrialState.RETRYABLE_FAILURE,
            TrialState.TERMINAL_FAILURE,
            TrialState.INTERRUPTED,
        }
    ),
    TrialState.EVALUATING: frozenset(
        {
            TrialState.SYNCING,
            TrialState.COMPLETE,
            TrialState.PRUNED,
            TrialState.RETRYABLE_FAILURE,
            TrialState.TERMINAL_FAILURE,
        }
    ),
    TrialState.SYNCING: frozenset(
        {
            TrialState.COMPLETE,
            TrialState.RETRYABLE_FAILURE,
            TrialState.TERMINAL_FAILURE,
        }
    ),
    TrialState.COMPLETE: frozenset(),
    TrialState.PRUNED: frozenset(),
    TrialState.RETRYABLE_FAILURE: frozenset(),
    TrialState.TERMINAL_FAILURE: frozenset(),
    TrialState.INTERRUPTED: frozenset(),
}


def require_campaign_transition(current: CampaignState, target: CampaignState) -> None:
    if target not in _CAMPAIGN_TRANSITIONS[current]:
        raise ValueError(f"invalid campaign transition {current.value} -> {target.value}")


def require_trial_transition(current: TrialState, target: TrialState) -> None:
    if target not in _TRIAL_TRANSITIONS[current]:
        raise ValueError(f"invalid trial transition {current.value} -> {target.value}")


class TrialSpec(FrozenConfigModel):
    """Immutable scheduler-side trial identity; workers receive this as data."""

    trial_id: str = Field(min_length=1)
    family: str = Field(min_length=1)
    scale: str = Field(min_length=1)
    stage: CampaignStage
    budget_fraction: float = Field(gt=0.0, le=1.0)
    priority: LaunchPriority = LaunchPriority.MANDATORY
    config: dict[str, JsonValue]
    parent_trial_id: str | None = None
    root_trial_id: str | None = None
    attempt: int = Field(default=0, ge=0)
    fallback_runtime_seconds: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_lineage(self) -> TrialSpec:
        if self.parent_trial_id is None and self.root_trial_id not in {None, self.trial_id}:
            raise ValueError("root trial without parent must root at itself")
        if self.parent_trial_id is not None and not self.root_trial_id:
            raise ValueError("child trial requires root_trial_id")
        return self

    @property
    def canonical_config_json(self) -> str:
        return json.dumps(self.config, sort_keys=True, separators=(",", ":"))

    @property
    def config_sha256(self) -> str:
        return hashlib.sha256(self.canonical_config_json.encode("utf-8")).hexdigest()

    @property
    def effective_root_trial_id(self) -> str:
        return self.root_trial_id or self.trial_id


class RuntimeObservation(FrozenConfigModel):
    trial_id: str = Field(min_length=1)
    family: str = Field(min_length=1)
    scale: str = Field(min_length=1)
    context_length: int | None = Field(default=None, gt=0)
    precision: str = Field(default="bf16", min_length=1)
    budget_fraction: float = Field(gt=0.0, le=1.0)
    runtime_seconds: float = Field(gt=0.0)
    samples_per_second: float | None = Field(default=None, gt=0.0)
    gpu_utilization_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    peak_vram_bytes: int | None = Field(default=None, ge=0)


class DrainInputs(FrozenConfigModel):
    outstanding_evaluator_seconds: float = Field(default=0.0, ge=0.0)
    unsynced_bytes: int = Field(default=0, ge=0)
    storage_bytes_per_second: float | None = Field(default=None, gt=0.0)


class ResourceSample(FrozenConfigModel):
    cpu_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    gpu_utilization_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    peak_vram_bytes: int | None = Field(default=None, ge=0)


TrialFailureClass = Literal["process_failure", "worker_exit", "scheduler_interruption"]


def canonical_json_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
