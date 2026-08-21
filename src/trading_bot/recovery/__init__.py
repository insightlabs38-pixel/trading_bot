"""CPU-safe deterministic fault recovery and repair infrastructure."""

from trading_bot.recovery.canary import GoldenCanaryResult, GoldenCanarySpec, run_cpu_golden_canary
from trading_bot.recovery.circuit import CircuitBreaker, CircuitState
from trading_bot.recovery.classifier import classify_failure
from trading_bot.recovery.health import check_dataset_sample, check_disk, check_storage_object
from trading_bot.recovery.heartbeat import WorkerHeartbeat, heartbeat_evidence, read_heartbeat, write_heartbeat
from trading_bot.recovery.lineage import derive_repaired_child
from trading_bot.recovery.policy import (
    RecoveryPolicy,
    RecoveryPolicyError,
    decide_recovery,
    derive_oom_child,
    derive_reference_fallback_child,
    load_recovery_policy,
)
from trading_bot.recovery.repair import (
    AIRepairCoordinator,
    DebugBundle,
    ProtectedFilePolicy,
    RepairAttempt,
    RepairAuditLog,
    RepairAuditRecord,
    RepairClient,
    build_debug_bundle,
    parse_repair_proposal,
    redact_mapping,
    redact_text,
    unavailable_gpu_gate,
)
from trading_bot.recovery.sandbox import RepairSandbox, run_command_gate, validate_repair
from trading_bot.recovery.types import (
    FailureClass,
    FailureClassification,
    FailureEvidence,
    GateResult,
    ProposedFileChange,
    RecoveryAction,
    RecoveryDecision,
    RepairProposal,
    RepairValidationResult,
    WorkerPhase,
)

__all__ = [
    "AIRepairCoordinator",
    "CircuitBreaker",
    "CircuitState",
    "DebugBundle",
    "FailureClass",
    "FailureClassification",
    "FailureEvidence",
    "GateResult",
    "GoldenCanaryResult",
    "GoldenCanarySpec",
    "ProposedFileChange",
    "ProtectedFilePolicy",
    "RecoveryAction",
    "RecoveryDecision",
    "RecoveryPolicy",
    "RecoveryPolicyError",
    "RepairAttempt",
    "RepairAuditLog",
    "RepairAuditRecord",
    "RepairClient",
    "RepairProposal",
    "RepairSandbox",
    "RepairValidationResult",
    "WorkerHeartbeat",
    "WorkerPhase",
    "build_debug_bundle",
    "check_dataset_sample",
    "check_disk",
    "check_storage_object",
    "classify_failure",
    "decide_recovery",
    "derive_oom_child",
    "derive_reference_fallback_child",
    "derive_repaired_child",
    "heartbeat_evidence",
    "load_recovery_policy",
    "parse_repair_proposal",
    "read_heartbeat",
    "redact_mapping",
    "redact_text",
    "run_command_gate",
    "run_cpu_golden_canary",
    "unavailable_gpu_gate",
    "validate_repair",
    "write_heartbeat",
]
