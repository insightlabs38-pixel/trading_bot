"""Sanitized, provider-neutral AI repair boundary and audit logging."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Protocol

from pydantic import Field, JsonValue

from trading_bot.config.base import FrozenConfigModel
from trading_bot.config.schemas import AIRepairConfig
from trading_bot.recovery.policy import RecoveryPolicy
from trading_bot.recovery.types import (
    GateResult,
    RepairProposal,
    RepairTier,
    RepairValidationResult,
)

_REDACTED = "<redacted>"
_SECRET_KEY_RE = re.compile(
    r"(?i)(api[_-]?key|token|password|secret|authorization|credential|webhook)"
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|password|secret|authorization|credential|webhook)\b"
    r"\s*[:=]\s*[^\s,;]+"
)
_AWS_KEY_RE = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_GITHUB_TOKEN_RE = re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b")
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")

_DEFAULT_ALLOWED_PATTERNS = (
    "src/trading_bot/models/**",
    "src/trading_bot/training/trainer.py",
)

_DEFAULT_PROTECTED_PATTERNS = (
    "PLAN.md",
    "IMPLEMENTATION_PLAN.md",
    "docs/evaluation_contract.md",
    "docs/data_and_storage_plan.md",
    "src/trading_bot/data/**",
    "src/trading_bot/evaluation/**",
    "src/trading_bot/config/**",
    "configs/campaigns/**",
    "src/trading_bot/scheduler/**",
    "src/trading_bot/recovery/**",
    ".github/**",
    "docker/**",
    "compose*.yml",
    "docker-compose*.yml",
    ".env*",
    "**/.env*",
    "**/*credential*",
    "**/*secret*",
)

_FORBIDDEN_DATA_SUFFIXES = frozenset({".parquet", ".arrow", ".feather", ".csv", ".npy", ".npz"})


class DebugBundle(FrozenConfigModel):
    schema_version: int = 1
    trial_id: str = Field(min_length=1)
    failure_class: str = Field(min_length=1)
    stack_trace: str
    recent_logs: str
    environment: dict[str, str]
    tensor_shapes: dict[str, tuple[int, ...]]
    config: dict[str, JsonValue]
    source_files: dict[str, str]

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


class ProtectedFilePolicy(FrozenConfigModel):
    """Default-deny repair surface with explicit narrow code allow-list."""

    allowed_patterns: tuple[str, ...] = _DEFAULT_ALLOWED_PATTERNS
    protected_patterns: tuple[str, ...] = _DEFAULT_PROTECTED_PATTERNS

    def is_protected(self, path: str) -> bool:
        normalized = PurePosixPath(path.replace("\\", "/"))
        if normalized.is_absolute() or ".." in normalized.parts:
            return True
        normalized_text = normalized.as_posix()
        if any(
            fnmatchcase(normalized_text, pattern)
            for pattern in self.protected_patterns
        ):
            return True
        return not any(
            fnmatchcase(normalized_text, pattern) for pattern in self.allowed_patterns
        )

    def require_allowed(self, paths: tuple[str, ...]) -> None:
        blocked = sorted(path for path in paths if self.is_protected(path))
        if blocked:
            raise PermissionError(f"AI repair proposal touches protected paths: {blocked}")


class RepairClient(Protocol):
    """Injected provider client. Scientific/recovery logic does not hardcode an API."""

    def propose(
        self,
        bundle: DebugBundle,
        *,
        tier: RepairTier,
        timeout_seconds: int,
        max_output_bytes: int,
    ) -> str: ...


class RepairAttempt(FrozenConfigModel):
    tier: RepairTier
    proposal: RepairProposal | None = None
    error: str | None = None


class RepairAuditRecord(FrozenConfigModel):
    schema_version: int = 1
    trial_id: str = Field(min_length=1)
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tier: RepairTier
    proposal_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    changed_paths: tuple[str, ...] = ()
    validation: RepairValidationResult | None = None
    outcome: str = Field(min_length=1)


class RepairAuditLog:
    """Append-only JSONL audit with fsync after each repair record."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: RepairAuditRecord) -> None:
        line = json.dumps(record.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def read_all(self) -> tuple[RepairAuditRecord, ...]:
        if not self.path.exists():
            return ()
        return tuple(
            RepairAuditRecord.model_validate(json.loads(line))
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )


def redact_text(value: str) -> str:
    redacted = _BEARER_RE.sub(f"Bearer {_REDACTED}", value)
    redacted = _SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}={_REDACTED}", redacted)
    redacted = _AWS_KEY_RE.sub(_REDACTED, redacted)
    return _GITHUB_TOKEN_RE.sub(_REDACTED, redacted)


def _redact_json(value: JsonValue) -> JsonValue:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return redact_mapping(value)
    if isinstance(value, list):
        return [_redact_json(item) for item in value]
    return value


def redact_mapping(value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, item in value.items():
        if _SECRET_KEY_RE.search(key):
            result[key] = _REDACTED
        else:
            result[key] = _redact_json(item)
    return result


def build_debug_bundle(
    *,
    trial_id: str,
    failure_class: str,
    stack_trace: str,
    recent_logs: str,
    environment: Mapping[str, str],
    tensor_shapes: Mapping[str, tuple[int, ...]],
    config: Mapping[str, JsonValue],
    source_files: Mapping[str, str],
    policy: RecoveryPolicy,
) -> DebugBundle:
    """Build a redacted bundle that contains no raw market-data attachments."""
    for path in source_files:
        suffix = PurePosixPath(path).suffix.lower()
        if suffix in _FORBIDDEN_DATA_SUFFIXES:
            raise ValueError(f"debug bundle refuses raw/licensed data file {path!r}")
    sanitized_environment = {
        key: (_REDACTED if _SECRET_KEY_RE.search(key) else redact_text(value))
        for key, value in environment.items()
    }
    bundle = DebugBundle(
        trial_id=trial_id,
        failure_class=failure_class,
        stack_trace=redact_text(stack_trace),
        recent_logs=redact_text(recent_logs),
        environment=sanitized_environment,
        tensor_shapes=dict(tensor_shapes),
        config=redact_mapping(config),
        source_files={path: redact_text(text) for path, text in source_files.items()},
    )
    if len(bundle.canonical_bytes) > policy.debug_bundle_max_bytes:
        raise ValueError("sanitized debug bundle exceeds configured size limit")
    return bundle


def parse_repair_proposal(raw: str, *, max_output_bytes: int) -> RepairProposal:
    encoded = raw.encode("utf-8")
    if len(encoded) > max_output_bytes:
        raise ValueError("AI repair output exceeds configured byte limit")
    payload = json.loads(raw)
    return RepairProposal.model_validate(payload)


class AIRepairCoordinator:
    """Optional primary-then-reasoning repair escalation that never raises to scheduler."""

    def __init__(
        self,
        *,
        config: AIRepairConfig,
        policy: RecoveryPolicy,
        primary_client: RepairClient | None,
        reasoning_client: RepairClient | None = None,
    ) -> None:
        self.config = config
        self.policy = policy
        self.primary_client = primary_client
        self.reasoning_client = reasoning_client

    def request(self, bundle: DebugBundle, *, high_value: bool) -> tuple[RepairAttempt, ...]:
        if not self.config.enabled or self.primary_client is None:
            return ()
        attempts: list[RepairAttempt] = []
        primary = self._attempt(bundle, self.primary_client, tier="primary")
        attempts.append(primary)
        if primary.proposal is not None:
            return tuple(attempts)
        if (
            high_value
            and self.config.allow_reasoning_escalation
            and self.reasoning_client is not None
            and len(attempts) < self.config.max_repair_attempts
        ):
            attempts.append(self._attempt(bundle, self.reasoning_client, tier="reasoning"))
        return tuple(attempts)

    def _attempt(
        self,
        bundle: DebugBundle,
        client: RepairClient,
        *,
        tier: RepairTier,
    ) -> RepairAttempt:
        timeout = (
            self.policy.repair_primary_timeout_seconds
            if tier == "primary"
            else self.policy.repair_reasoning_timeout_seconds
        )
        try:
            raw = client.propose(
                bundle,
                tier=tier,
                timeout_seconds=timeout,
                max_output_bytes=self.policy.repair_max_output_bytes,
            )
            proposal = parse_repair_proposal(
                raw, max_output_bytes=self.policy.repair_max_output_bytes
            )
            return RepairAttempt(tier=tier, proposal=proposal)
        except Exception as exc:  # provider failures must never stop the campaign
            return RepairAttempt(tier=tier, error=f"{type(exc).__name__}: {exc}")


def unavailable_gpu_gate() -> GateResult:
    return GateResult(
        name="gpu_smoke",
        passed=False,
        detail="GPU smoke validation unavailable in CPU-only environment",
    )
