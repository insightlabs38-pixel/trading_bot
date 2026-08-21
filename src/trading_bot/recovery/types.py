"""CPU-safe fault classification and recovery contracts."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import Field, JsonValue

from trading_bot.config.base import FrozenConfigModel


class FailureClass(StrEnum):
    CUDA_OOM = "cuda_oom"
    NON_FINITE = "nan_or_inf"
    PROCESS_CRASH = "transient_process_crash"
    TRITON_COMPILE = "triton_compile_failure"
    ILLEGAL_MEMORY_ACCESS = "illegal_memory_access"
    STALE_HEARTBEAT = "stale_heartbeat"
    CORRUPT_DATA_SHARD = "corrupted_data_shard"
    CHECKPOINT_CORRUPTION = "checkpoint_corruption"
    EVALUATOR_FAILURE = "evaluator_failure"
    STORAGE_FAILURE = "storage_failure"
    DISK_PRESSURE = "disk_pressure"
    CONFIGURATION_ERROR = "deterministic_configuration_error"
    UNKNOWN = "unknown"


class WorkerPhase(StrEnum):
    COMPILING = "COMPILING"
    DATALOADING = "DATALOADING"
    TRAINING = "TRAINING"
    CHECKPOINTING = "CHECKPOINTING"
    EVALUATING = "EVALUATING"


class RecoveryAction(StrEnum):
    RETRY_PROCESS = "retry_process"
    REDUCE_MICROBATCH = "reduce_microbatch"
    FALLBACK_REFERENCE = "fallback_reference"
    RESUME_CHECKPOINT = "resume_checkpoint"
    RETRY_EVALUATOR = "retry_evaluator"
    RETRY_STORAGE = "retry_storage"
    KILL_AND_RETRY = "kill_and_retry"
    PAUSE_CAMPAIGN = "pause_campaign"
    QUARANTINE = "quarantine"
    REQUEST_AI_REPAIR = "request_ai_repair"


class FailureEvidence(FrozenConfigModel):
    """Worker-side evidence supplied to deterministic classification."""

    message: str = ""
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    worker_phase: WorkerPhase | None = None
    heartbeat_age_seconds: float | None = Field(default=None, ge=0.0)
    heartbeat_timeout_seconds: float | None = Field(default=None, gt=0.0)
    free_disk_bytes: int | None = Field(default=None, ge=0)
    expected_disk_floor_bytes: int | None = Field(default=None, ge=0)


class FailureClassification(FrozenConfigModel):
    failure_class: FailureClass
    infrastructure_like: bool
    retryable_default: bool
    evidence_summary: str = Field(min_length=1)


class RecoveryDecision(FrozenConfigModel):
    retryable: bool
    actions: tuple[RecoveryAction, ...]
    reason: str = Field(min_length=1)
    child_config: dict[str, JsonValue] | None = None
    resume_checkpoint_key: str | None = None
    requires_gpu_health_gate: bool = False


RepairTier = Literal["primary", "reasoning"]


class ProposedFileChange(FrozenConfigModel):
    path: str = Field(min_length=1)
    expected_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    replacement_text: str


class RepairProposal(FrozenConfigModel):
    schema_version: Literal[1] = 1
    summary: str = Field(min_length=1, max_length=2000)
    diagnosis: str = Field(min_length=1, max_length=10000)
    changes: tuple[ProposedFileChange, ...]
    requested_tests: tuple[str, ...] = ()

    @property
    def canonical_sha256(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class GateResult(FrozenConfigModel):
    name: str = Field(min_length=1)
    passed: bool
    detail: str = ""
    duration_seconds: float = Field(default=0.0, ge=0.0)


class RepairValidationResult(FrozenConfigModel):
    static_gate: GateResult
    unit_gate: GateResult
    regression_gate: GateResult
    gpu_smoke_gate: GateResult

    @property
    def eligible_for_requeue(self) -> bool:
        return all(
            gate.passed
            for gate in (
                self.static_gate,
                self.unit_gate,
                self.regression_gate,
                self.gpu_smoke_gate,
            )
        )
