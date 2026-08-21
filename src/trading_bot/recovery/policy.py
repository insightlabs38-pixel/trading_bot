"""Versioned deterministic recovery policy and child-trial derivation."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Literal

import yaml  # type: ignore[import-untyped]
from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from trading_bot.config.base import FrozenConfigModel
from trading_bot.recovery.types import (
    FailureClass,
    FailureClassification,
    RecoveryAction,
    RecoveryDecision,
    WorkerPhase,
)
from trading_bot.scheduler.types import TrialSpec


class RecoveryPolicyError(ValueError):
    """Raised when deterministic recovery policy cannot be loaded safely."""


class RecoveryPolicy(FrozenConfigModel):
    schema_version: Literal[1] = 1
    policy_id: str = Field(min_length=1)
    max_process_retries: int = Field(default=2, ge=0, le=10)
    max_non_finite_retries: int = Field(default=1, ge=0, le=5)
    max_evaluator_retries: int = Field(default=3, ge=0, le=10)
    max_storage_retries: int = Field(default=5, ge=0, le=20)
    oom_microbatch_divisor: PositiveInt = 2
    minimum_microbatch_size: PositiveInt = 1
    preserve_effective_batch: bool = True
    heartbeat_timeouts_seconds: dict[WorkerPhase, PositiveFloat]
    circuit_window_seconds: PositiveFloat = 900.0
    circuit_failure_threshold: PositiveInt = 3
    circuit_cooldown_seconds: PositiveFloat = 300.0
    minimum_free_disk_bytes: PositiveInt = 5 * 1024**3
    repair_primary_timeout_seconds: PositiveInt = 45
    repair_reasoning_timeout_seconds: PositiveInt = 120
    repair_max_output_bytes: PositiveInt = 256 * 1024
    debug_bundle_max_bytes: PositiveInt = 512 * 1024

    @model_validator(mode="after")
    def validate_policy(self) -> RecoveryPolicy:
        missing = set(WorkerPhase) - set(self.heartbeat_timeouts_seconds)
        if missing:
            missing_names = sorted(item.value for item in missing)
            raise ValueError(f"heartbeat timeouts missing phases: {missing_names}")
        if self.circuit_failure_threshold < 2:
            raise ValueError("circuit breaker threshold must be at least two failures")
        return self


def load_recovery_policy(path: str | Path) -> RecoveryPolicy:
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RecoveryPolicyError(f"unable to load recovery policy {path}") from exc
    if not isinstance(payload, dict):
        raise RecoveryPolicyError("recovery policy root must be a mapping")
    try:
        return RecoveryPolicy.model_validate(payload)
    except ValueError as exc:
        raise RecoveryPolicyError(str(exc)) from exc


def derive_oom_child(parent: TrialSpec, policy: RecoveryPolicy) -> TrialSpec | None:
    """Derive an immutable lower-microbatch child while preserving effective batch."""
    batch_value = parent.config.get("batch")
    if not isinstance(batch_value, dict):
        return None
    batch = copy.deepcopy(batch_value)
    microbatch = batch.get("microbatch_size")
    effective = batch.get("effective_batch_size")
    if not isinstance(microbatch, int) or not isinstance(effective, int):
        return None
    if microbatch <= policy.minimum_microbatch_size:
        return None

    reduced = max(policy.minimum_microbatch_size, microbatch // policy.oom_microbatch_divisor)
    if reduced >= microbatch:
        return None
    if policy.preserve_effective_batch and effective % reduced != 0:
        return None

    accumulation = math.ceil(effective / reduced)
    if policy.preserve_effective_batch and reduced * accumulation != effective:
        return None

    child_config = copy.deepcopy(parent.config)
    child_config["batch"] = {
        "microbatch_size": reduced,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": (
            effective if policy.preserve_effective_batch else reduced * accumulation
        ),
    }
    return TrialSpec(
        trial_id=f"{parent.trial_id}-oom-r{parent.attempt + 1}",
        family=parent.family,
        scale=parent.scale,
        stage=parent.stage,
        budget_fraction=parent.budget_fraction,
        priority=parent.priority,
        config=child_config,
        parent_trial_id=parent.trial_id,
        root_trial_id=parent.effective_root_trial_id,
        attempt=parent.attempt + 1,
        fallback_runtime_seconds=parent.fallback_runtime_seconds,
    )


def derive_reference_fallback_child(parent: TrialSpec) -> TrialSpec:
    child_config = copy.deepcopy(parent.config)
    child_config["runtime_overrides"] = {"custom_backend": "reference"}
    return TrialSpec(
        trial_id=f"{parent.trial_id}-reference-r{parent.attempt + 1}",
        family=parent.family,
        scale=parent.scale,
        stage=parent.stage,
        budget_fraction=parent.budget_fraction,
        priority=parent.priority,
        config=child_config,
        parent_trial_id=parent.trial_id,
        root_trial_id=parent.effective_root_trial_id,
        attempt=parent.attempt + 1,
        fallback_runtime_seconds=parent.fallback_runtime_seconds,
    )


def decide_recovery(
    classification: FailureClassification,
    parent: TrialSpec,
    policy: RecoveryPolicy,
    *,
    valid_checkpoint_key: str | None = None,
    reference_backend_available: bool = False,
    evaluator_attempts: int = 0,
    storage_attempts: int = 0,
) -> tuple[RecoveryDecision, TrialSpec | None]:
    """Return an explicit deterministic action and optional immutable child trial."""
    failure_class = classification.failure_class

    if failure_class == FailureClass.CUDA_OOM:
        if parent.attempt >= policy.max_process_retries:
            return _quarantine("CUDA OOM retry budget exhausted"), None
        child = derive_oom_child(parent, policy)
        if child is None:
            return _quarantine("CUDA OOM cannot derive a valid smaller microbatch child"), None
        return (
            RecoveryDecision(
                retryable=True,
                actions=(RecoveryAction.REDUCE_MICROBATCH, RecoveryAction.RETRY_PROCESS),
                reason="derive lower-microbatch immutable child after full worker restart",
                child_config=child.config,
            ),
            child,
        )

    if failure_class == FailureClass.TRITON_COMPILE:
        if not reference_backend_available:
            return _quarantine("validated reference backend unavailable"), None
        child = derive_reference_fallback_child(parent)
        return (
            RecoveryDecision(
                retryable=True,
                actions=(RecoveryAction.FALLBACK_REFERENCE, RecoveryAction.RETRY_PROCESS),
                reason="retry as immutable child on validated reference backend",
                child_config=child.config,
            ),
            child,
        )

    if failure_class == FailureClass.NON_FINITE:
        if parent.attempt >= policy.max_non_finite_retries:
            return _quarantine("non-finite recovery budget exhausted"), None
        if valid_checkpoint_key is None:
            return _quarantine("non-finite failure has no last-good checkpoint"), None
        return (
            RecoveryDecision(
                retryable=True,
                actions=(RecoveryAction.RESUME_CHECKPOINT, RecoveryAction.RETRY_PROCESS),
                reason="one bounded retry from last-good checkpoint",
                resume_checkpoint_key=valid_checkpoint_key,
            ),
            None,
        )

    if failure_class == FailureClass.CHECKPOINT_CORRUPTION:
        if valid_checkpoint_key is None:
            return _quarantine("no earlier verified checkpoint remains"), None
        return (
            RecoveryDecision(
                retryable=True,
                actions=(RecoveryAction.RESUME_CHECKPOINT, RecoveryAction.RETRY_PROCESS),
                reason="resume from earlier checksum-verified checkpoint",
                resume_checkpoint_key=valid_checkpoint_key,
            ),
            None,
        )

    if failure_class == FailureClass.EVALUATOR_FAILURE:
        if evaluator_attempts >= policy.max_evaluator_retries:
            return _quarantine("evaluator retry budget exhausted"), None
        return (
            RecoveryDecision(
                retryable=True,
                actions=(RecoveryAction.RETRY_EVALUATOR,),
                reason="retry evaluator independently of trainer",
            ),
            None,
        )

    if failure_class == FailureClass.STORAGE_FAILURE:
        if storage_attempts >= policy.max_storage_retries:
            return _quarantine("storage retry budget exhausted"), None
        return (
            RecoveryDecision(
                retryable=True,
                actions=(RecoveryAction.RETRY_STORAGE,),
                reason="retain local artifact and retry durable sync independently",
            ),
            None,
        )

    if failure_class in {FailureClass.ILLEGAL_MEMORY_ACCESS, FailureClass.STALE_HEARTBEAT}:
        if parent.attempt >= policy.max_process_retries:
            return _quarantine("worker restart budget exhausted"), None
        return (
            RecoveryDecision(
                retryable=True,
                actions=(RecoveryAction.KILL_AND_RETRY,),
                reason="kill process group and retry in a fresh worker",
                resume_checkpoint_key=valid_checkpoint_key,
                requires_gpu_health_gate=(failure_class == FailureClass.ILLEGAL_MEMORY_ACCESS),
            ),
            None,
        )

    if failure_class == FailureClass.PROCESS_CRASH:
        if parent.attempt >= policy.max_process_retries:
            return _quarantine("process retry budget exhausted"), None
        return (
            RecoveryDecision(
                retryable=True,
                actions=(RecoveryAction.RETRY_PROCESS,),
                reason="bounded fresh-process retry",
                resume_checkpoint_key=valid_checkpoint_key,
            ),
            None,
        )

    if failure_class in {FailureClass.CORRUPT_DATA_SHARD, FailureClass.DISK_PRESSURE}:
        return (
            RecoveryDecision(
                retryable=False,
                actions=(RecoveryAction.PAUSE_CAMPAIGN, RecoveryAction.QUARANTINE),
                reason="global safety condition requires intervention before new launches",
            ),
            None,
        )

    if failure_class == FailureClass.CONFIGURATION_ERROR:
        return _quarantine("deterministic configuration errors are terminal"), None

    return (
        RecoveryDecision(
            retryable=False,
            actions=(RecoveryAction.QUARANTINE, RecoveryAction.REQUEST_AI_REPAIR),
            reason="unknown failure is quarantined before optional repair escalation",
        ),
        None,
    )


def _quarantine(reason: str) -> RecoveryDecision:
    return RecoveryDecision(
        retryable=False,
        actions=(RecoveryAction.QUARANTINE,),
        reason=reason,
    )
