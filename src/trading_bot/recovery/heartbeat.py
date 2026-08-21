"""Atomic worker heartbeats and state-specific stale detection."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from pydantic import Field

from trading_bot.config.base import FrozenConfigModel
from trading_bot.recovery.policy import RecoveryPolicy
from trading_bot.recovery.types import FailureEvidence, WorkerPhase


class WorkerHeartbeat(FrozenConfigModel):
    schema_version: int = 1
    trial_id: str = Field(min_length=1)
    phase: WorkerPhase
    observed_at: float = Field(gt=0.0)
    training_step: int | None = Field(default=None, ge=0)
    loss: float | None = None
    primary_metric: float | None = None
    samples_per_second: float | None = Field(default=None, gt=0.0)
    latest_checkpoint_age_seconds: float | None = Field(default=None, ge=0.0)


def write_heartbeat(path: str | Path, heartbeat: WorkerHeartbeat) -> None:
    """Atomically publish one heartbeat so readers never observe partial JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    payload = heartbeat.model_dump_json(indent=None)
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def read_heartbeat(path: str | Path) -> WorkerHeartbeat:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return WorkerHeartbeat.model_validate(payload)


def heartbeat_evidence(
    heartbeat: WorkerHeartbeat,
    policy: RecoveryPolicy,
    *,
    now: float | None = None,
) -> FailureEvidence:
    current = time.time() if now is None else now
    if current < heartbeat.observed_at:
        raise ValueError("heartbeat timestamp is in the future")
    timeout = float(policy.heartbeat_timeouts_seconds[heartbeat.phase])
    return FailureEvidence(
        message=f"heartbeat phase={heartbeat.phase.value}",
        worker_phase=heartbeat.phase,
        heartbeat_age_seconds=current - heartbeat.observed_at,
        heartbeat_timeout_seconds=timeout,
    )
