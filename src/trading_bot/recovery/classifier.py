"""Deterministic failure classification from worker evidence."""

from __future__ import annotations

import re

from trading_bot.recovery.types import FailureClass, FailureClassification, FailureEvidence, WorkerPhase

_INFRASTRUCTURE_LIKE = frozenset(
    {
        FailureClass.PROCESS_CRASH,
        FailureClass.ILLEGAL_MEMORY_ACCESS,
        FailureClass.STALE_HEARTBEAT,
        FailureClass.STORAGE_FAILURE,
        FailureClass.DISK_PRESSURE,
    }
)

_PATTERNS: tuple[tuple[FailureClass, tuple[re.Pattern[str], ...]], ...] = (
    (
        FailureClass.CUDA_OOM,
        (
            re.compile(r"cuda out of memory", re.I),
            re.compile(r"outofmemoryerror", re.I),
            re.compile(r"cublas_status_alloc_failed", re.I),
        ),
    ),
    (
        FailureClass.ILLEGAL_MEMORY_ACCESS,
        (
            re.compile(r"illegal memory access", re.I),
            re.compile(r"misaligned address", re.I),
            re.compile(r"device-side assert", re.I),
        ),
    ),
    (
        FailureClass.TRITON_COMPILE,
        (
            re.compile(r"triton.*(?:compile|compilation)", re.I | re.S),
            re.compile(r"(?:compile|compilation).*triton", re.I | re.S),
        ),
    ),
    (
        FailureClass.CHECKPOINT_CORRUPTION,
        (
            re.compile(r"checkpoint.*(?:corrupt|checksum|sha-?256.*mismatch)", re.I | re.S),
            re.compile(r"(?:corrupt|checksum).*checkpoint", re.I | re.S),
        ),
    ),
    (
        FailureClass.CORRUPT_DATA_SHARD,
        (
            re.compile(r"data shard.*(?:corrupt|checksum|invalid)", re.I | re.S),
            re.compile(r"(?:parquet|arrow|npy).*checksum.*mismatch", re.I | re.S),
        ),
    ),
    (
        FailureClass.DISK_PRESSURE,
        (
            re.compile(r"no space left on device", re.I),
            re.compile(r"disk pressure", re.I),
            re.compile(r"enospc", re.I),
        ),
    ),
    (
        FailureClass.STORAGE_FAILURE,
        (
            re.compile(r"storage.*(?:upload|download|sync).*(?:fail|timeout|unavailable)", re.I | re.S),
            re.compile(r"(?:s3|object store).*(?:timeout|unavailable|connection)", re.I | re.S),
        ),
    ),
    (
        FailureClass.CONFIGURATION_ERROR,
        (
            re.compile(r"deterministic configuration error", re.I),
            re.compile(r"invalid configuration", re.I),
            re.compile(r"validation error.*config", re.I | re.S),
            re.compile(r"unknown config(?:uration)? field", re.I),
        ),
    ),
    (
        FailureClass.NON_FINITE,
        (
            re.compile(r"non[- ]?finite", re.I),
            re.compile(r"loss (?:is|became) nan", re.I),
            re.compile(r"contains nan/inf", re.I),
            re.compile(r"floatingpointerror.*(?:nan|inf)", re.I | re.S),
        ),
    ),
)

_DEFAULT_RETRYABLE = frozenset(
    {
        FailureClass.CUDA_OOM,
        FailureClass.NON_FINITE,
        FailureClass.PROCESS_CRASH,
        FailureClass.TRITON_COMPILE,
        FailureClass.ILLEGAL_MEMORY_ACCESS,
        FailureClass.STALE_HEARTBEAT,
        FailureClass.CHECKPOINT_CORRUPTION,
        FailureClass.EVALUATOR_FAILURE,
        FailureClass.STORAGE_FAILURE,
    }
)


def classify_failure(evidence: FailureEvidence) -> FailureClassification:
    """Classify one failure using stable priority rules and no model inference."""
    if (
        evidence.heartbeat_age_seconds is not None
        and evidence.heartbeat_timeout_seconds is not None
        and evidence.heartbeat_age_seconds > evidence.heartbeat_timeout_seconds
    ):
        return _classification(
            FailureClass.STALE_HEARTBEAT,
            f"heartbeat age {evidence.heartbeat_age_seconds:.3f}s exceeded "
            f"{evidence.heartbeat_timeout_seconds:.3f}s",
        )

    if (
        evidence.free_disk_bytes is not None
        and evidence.expected_disk_floor_bytes is not None
        and evidence.free_disk_bytes < evidence.expected_disk_floor_bytes
    ):
        return _classification(
            FailureClass.DISK_PRESSURE,
            f"free disk {evidence.free_disk_bytes} below floor {evidence.expected_disk_floor_bytes}",
        )

    combined = "\n".join((evidence.message, evidence.stdout, evidence.stderr))
    for failure_class, patterns in _PATTERNS:
        if any(pattern.search(combined) for pattern in patterns):
            return _classification(failure_class, f"matched deterministic {failure_class.value} signature")

    if evidence.worker_phase == WorkerPhase.EVALUATING and evidence.exit_code not in {None, 0}:
        return _classification(
            FailureClass.EVALUATOR_FAILURE,
            f"evaluator exited with code {evidence.exit_code}",
        )

    if evidence.exit_code not in {None, 0}:
        return _classification(
            FailureClass.PROCESS_CRASH,
            f"worker exited with nonzero code {evidence.exit_code}",
        )

    return _classification(FailureClass.UNKNOWN, "no deterministic failure signature matched")


def _classification(failure_class: FailureClass, summary: str) -> FailureClassification:
    return FailureClassification(
        failure_class=failure_class,
        infrastructure_like=failure_class in _INFRASTRUCTURE_LIKE,
        retryable_default=failure_class in _DEFAULT_RETRYABLE,
        evidence_summary=summary,
    )
