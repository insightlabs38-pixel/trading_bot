from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from trading_bot.config.schemas import AIRepairConfig
from trading_bot.recovery import (
    AIRepairCoordinator,
    CircuitBreaker,
    DebugBundle,
    FailureClass,
    FailureEvidence,
    GateResult,
    ProtectedFilePolicy,
    RecoveryPolicy,
    RepairAuditLog,
    RepairAuditRecord,
    RepairProposal,
    RepairSandbox,
    WorkerHeartbeat,
    WorkerPhase,
    build_debug_bundle,
    check_dataset_sample,
    check_disk,
    check_storage_object,
    classify_failure,
    decide_recovery,
    derive_repaired_child,
    heartbeat_evidence,
    load_recovery_policy,
    parse_repair_proposal,
    read_heartbeat,
    redact_mapping,
    redact_text,
    run_cpu_golden_canary,
    unavailable_gpu_gate,
    validate_repair,
    write_heartbeat,
)
from trading_bot.recovery.types import ProposedFileChange, RecoveryAction, RepairTier
from trading_bot.scheduler.types import TrialSpec, TrialState, require_trial_transition
from trading_bot.storage.local import LocalStorageBackend

ROOT = Path(__file__).parents[2]
POLICY_PATH = ROOT / "configs/campaigns/recovery_policy_v1.yaml"


def _policy() -> RecoveryPolicy:
    return load_recovery_policy(POLICY_PATH)


def _trial(*, attempt: int = 0, microbatch: int = 64) -> TrialSpec:
    return TrialSpec(
        trial_id="phase12-trial",
        family="market_mixer",
        scale="small",
        stage="screening",
        budget_fraction=0.15,
        config={
            "family": "market_mixer",
            "scale": "small",
            "batch": {
                "microbatch_size": microbatch,
                "gradient_accumulation_steps": 4,
                "effective_batch_size": 256,
            },
        },
        attempt=attempt,
        fallback_runtime_seconds=60.0,
    )


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("RuntimeError: CUDA out of memory", FailureClass.CUDA_OOM),
        ("loss is NaN", FailureClass.NON_FINITE),
        ("Triton compilation failed", FailureClass.TRITON_COMPILE),
        (
            "CUDA error: an illegal memory access was encountered",
            FailureClass.ILLEGAL_MEMORY_ACCESS,
        ),
        ("checkpoint checksum mismatch", FailureClass.CHECKPOINT_CORRUPTION),
        ("data shard checksum mismatch", FailureClass.CORRUPT_DATA_SHARD),
        ("storage upload timeout", FailureClass.STORAGE_FAILURE),
        ("No space left on device", FailureClass.DISK_PRESSURE),
        ("invalid configuration for model", FailureClass.CONFIGURATION_ERROR),
    ],
)
def test_deterministic_failure_signatures(message: str, expected: FailureClass) -> None:
    assert classify_failure(FailureEvidence(stderr=message)).failure_class == expected


def test_stale_heartbeat_has_priority() -> None:
    result = classify_failure(
        FailureEvidence(
            stderr="CUDA out of memory",
            heartbeat_age_seconds=181.0,
            heartbeat_timeout_seconds=180.0,
        )
    )
    assert result.failure_class == FailureClass.STALE_HEARTBEAT


def test_disk_floor_evidence_is_deterministic() -> None:
    result = classify_failure(FailureEvidence(free_disk_bytes=5, expected_disk_floor_bytes=10))
    assert result.failure_class == FailureClass.DISK_PRESSURE


def test_evaluator_and_process_exit_classification() -> None:
    evaluator = classify_failure(FailureEvidence(worker_phase=WorkerPhase.EVALUATING, exit_code=2))
    process = classify_failure(FailureEvidence(worker_phase=WorkerPhase.TRAINING, exit_code=2))
    assert evaluator.failure_class == FailureClass.EVALUATOR_FAILURE
    assert process.failure_class == FailureClass.PROCESS_CRASH


def test_unknown_failure_stays_unknown() -> None:
    result = classify_failure(FailureEvidence(message="mystery"))
    assert result.failure_class == FailureClass.UNKNOWN


def test_policy_loads_and_declares_every_worker_timeout() -> None:
    policy = _policy()
    assert set(policy.heartbeat_timeouts_seconds) == set(WorkerPhase)
    assert policy.preserve_effective_batch is True


def test_oom_child_reduces_microbatch_and_preserves_effective_batch() -> None:
    classification = classify_failure(FailureEvidence(stderr="CUDA out of memory"))
    decision, child = decide_recovery(classification, _trial(), _policy())
    assert decision.actions == (
        RecoveryAction.REDUCE_MICROBATCH,
        RecoveryAction.RETRY_PROCESS,
    )
    assert child is not None
    batch = child.config["batch"]
    assert isinstance(batch, dict)
    assert batch["microbatch_size"] == 32
    assert batch["gradient_accumulation_steps"] == 8
    assert batch["effective_batch_size"] == 256
    assert child.parent_trial_id == "phase12-trial"
    assert child.effective_root_trial_id == "phase12-trial"


def test_oom_fails_closed_when_microbatch_cannot_shrink() -> None:
    classification = classify_failure(FailureEvidence(stderr="CUDA out of memory"))
    decision, child = decide_recovery(classification, _trial(microbatch=1), _policy())
    assert child is None
    assert decision.actions == (RecoveryAction.QUARANTINE,)


def test_triton_fallback_is_an_immutable_reference_child() -> None:
    classification = classify_failure(FailureEvidence(stderr="Triton compilation failed"))
    decision, child = decide_recovery(
        classification,
        _trial(),
        _policy(),
        reference_backend_available=True,
    )
    assert decision.actions[0] == RecoveryAction.FALLBACK_REFERENCE
    assert child is not None
    assert child.parent_trial_id == "phase12-trial"
    assert child.config["runtime_overrides"] == {"custom_backend": "reference"}


def test_non_finite_recovery_requires_last_good_checkpoint() -> None:
    classification = classify_failure(FailureEvidence(stderr="loss is NaN"))
    without_checkpoint, _ = decide_recovery(classification, _trial(), _policy())
    with_checkpoint, _ = decide_recovery(
        classification,
        _trial(),
        _policy(),
        valid_checkpoint_key="checkpoints/good.ready",
    )
    assert without_checkpoint.actions == (RecoveryAction.QUARANTINE,)
    assert with_checkpoint.resume_checkpoint_key == "checkpoints/good.ready"


def test_evaluator_and_storage_retries_are_independent() -> None:
    evaluator = classify_failure(FailureEvidence(worker_phase=WorkerPhase.EVALUATING, exit_code=7))
    storage = classify_failure(FailureEvidence(stderr="storage upload timeout"))
    evaluator_decision, _ = decide_recovery(evaluator, _trial(), _policy(), evaluator_attempts=0)
    storage_decision, _ = decide_recovery(storage, _trial(), _policy(), storage_attempts=0)
    assert evaluator_decision.actions == (RecoveryAction.RETRY_EVALUATOR,)
    assert storage_decision.actions == (RecoveryAction.RETRY_STORAGE,)


def test_data_and_disk_incidents_pause_campaign() -> None:
    for evidence in (
        FailureEvidence(stderr="data shard checksum mismatch"),
        FailureEvidence(stderr="No space left on device"),
    ):
        decision, _ = decide_recovery(classify_failure(evidence), _trial(), _policy())
        assert decision.actions == (
            RecoveryAction.PAUSE_CAMPAIGN,
            RecoveryAction.QUARANTINE,
        )


def test_unknown_failure_quarantines_before_optional_ai() -> None:
    decision, child = decide_recovery(
        classify_failure(FailureEvidence(message="unexpected custom operator crash")),
        _trial(),
        _policy(),
    )
    assert child is None
    assert decision.actions == (
        RecoveryAction.QUARANTINE,
        RecoveryAction.REQUEST_AI_REPAIR,
    )


def test_heartbeat_roundtrip_and_state_specific_timeout(tmp_path: Path) -> None:
    heartbeat = WorkerHeartbeat(
        trial_id="trial",
        phase=WorkerPhase.TRAINING,
        observed_at=1000.0,
        training_step=4,
        samples_per_second=20.0,
    )
    path = tmp_path / "heartbeat.json"
    write_heartbeat(path, heartbeat)
    assert read_heartbeat(path) == heartbeat
    evidence = heartbeat_evidence(heartbeat, _policy(), now=1181.0)
    assert classify_failure(evidence).failure_class == FailureClass.STALE_HEARTBEAT


def test_circuit_breaker_requires_complete_health_gate() -> None:
    breaker = CircuitBreaker(_policy())
    failure = classify_failure(FailureEvidence(exit_code=9))
    assert breaker.record_failure(failure, now=1.0) is False
    assert breaker.record_failure(failure, now=2.0) is False
    assert breaker.record_failure(failure, now=3.0) is True
    assert breaker.can_launch(now=4.0) is False
    assert breaker.can_launch(now=304.0) is False
    cpu_only = (
        GateResult(name="disk", passed=True),
        GateResult(name="dataset", passed=True),
        GateResult(name="storage", passed=True),
    )
    assert breaker.apply_health_gate(cpu_only) is False
    complete = (
        *cpu_only,
        GateResult(name="gpu_smoke", passed=True, detail="synthetic fixture"),
    )
    assert breaker.apply_health_gate(complete) is True
    assert breaker.can_launch(now=305.0) is True


def test_cpu_health_checks_and_golden_canary(tmp_path: Path) -> None:
    dataset = tmp_path / "sample.bin"
    dataset.write_bytes(b"known-sample")
    checksum = hashlib.sha256(dataset.read_bytes()).hexdigest()
    backend = LocalStorageBackend(tmp_path / "storage")
    backend.upload(dataset, "health/sample.bin", expected_sha256=checksum)
    assert check_disk(tmp_path, minimum_free_bytes=1).passed
    assert check_dataset_sample(dataset, expected_sha256=checksum).passed
    assert check_storage_object(
        backend,
        key="health/sample.bin",
        expected_sha256=checksum,
    ).passed
    canary = run_cpu_golden_canary(backend, storage_key="canary/model.json")
    assert canary.passed
    assert canary.mse == 0.0


def test_secret_and_market_data_redaction() -> None:
    text = "api_key=super-secret Bearer abcdefghijklmnop AKIA1234567890123456"
    redacted = redact_text(text)
    assert "super-secret" not in redacted
    assert "abcdefghijklmnop" not in redacted
    assert "AKIA1234567890123456" not in redacted
    mapping = redact_mapping(
        {
            "token": "abc",
            "nested": {"password": "pw"},
            "items": [{"secret": "hidden"}],
        }
    )
    assert mapping["token"] == "<redacted>"
    assert mapping["nested"] == {"password": "<redacted>"}
    assert mapping["items"] == [{"secret": "<redacted>"}]
    with pytest.raises(ValueError, match="raw/licensed data"):
        build_debug_bundle(
            trial_id="trial",
            failure_class="unknown",
            stack_trace="trace",
            recent_logs="logs",
            environment={},
            tensor_shapes={},
            config={},
            source_files={"sample.parquet": "raw rows"},
            policy=_policy(),
        )


def test_debug_bundle_redacts_secrets() -> None:
    bundle = build_debug_bundle(
        trial_id="trial",
        failure_class="unknown",
        stack_trace="Authorization: Bearer abcdefghijklmnop",
        recent_logs="token=my-token",
        environment={"API_KEY": "top-secret", "PYTHON": "3.12"},
        tensor_shapes={"features": (8, 16, 4)},
        config={"credentials": "do-not-send", "lr": 0.001},
        source_files={"src/model.py": "password=hidden"},
        policy=_policy(),
    )
    serialized = bundle.canonical_bytes.decode("utf-8")
    secrets = (
        "abcdefghijklmnop",
        "my-token",
        "top-secret",
        "do-not-send",
        "hidden",
    )
    for secret in secrets:
        assert secret not in serialized


def test_protected_file_policy_blocks_frozen_contracts() -> None:
    policy = ProtectedFilePolicy()
    for path in (
        "PLAN.md",
        "IMPLEMENTATION_PLAN.md",
        "docs/evaluation_contract.md",
        "src/trading_bot/evaluation/leaderboard.py",
        "src/trading_bot/scheduler/db.py",
        "configs/campaigns/h200_tournament_v1.yaml",
        ".github/workflows/cpu-ci.yml",
        ".env",
    ):
        assert policy.is_protected(path)
    assert not policy.is_protected("src/trading_bot/models/custom.py")


def _proposal_for(path: str, current: bytes, replacement: str) -> RepairProposal:
    return RepairProposal(
        summary="repair",
        diagnosis="fixture",
        changes=(
            ProposedFileChange(
                path=path,
                expected_sha256=hashlib.sha256(current).hexdigest(),
                replacement_text=replacement,
            ),
        ),
    )


def test_repair_proposal_output_limit() -> None:
    raw = json.dumps(
        {
            "schema_version": 1,
            "summary": "repair",
            "diagnosis": "fixture",
            "changes": [],
            "requested_tests": [],
        }
    )
    assert parse_repair_proposal(raw, max_output_bytes=4096).summary == "repair"
    with pytest.raises(ValueError, match="byte limit"):
        parse_repair_proposal(raw, max_output_bytes=5)


class _FakeClient:
    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.calls: list[tuple[RepairTier, int, int]] = []

    def propose(
        self,
        bundle: DebugBundle,
        *,
        tier: RepairTier,
        timeout_seconds: int,
        max_output_bytes: int,
    ) -> str:
        del bundle
        self.calls.append((tier, timeout_seconds, max_output_bytes))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _empty_bundle() -> DebugBundle:
    return DebugBundle(
        trial_id="trial",
        failure_class="unknown",
        stack_trace="",
        recent_logs="",
        environment={},
        tensor_shapes={},
        config={},
        source_files={},
    )


def test_ai_repair_primary_then_optional_reasoning_never_raises() -> None:
    good = json.dumps(
        {
            "schema_version": 1,
            "summary": "repair",
            "diagnosis": "fixture",
            "changes": [],
            "requested_tests": [],
        }
    )
    primary = _FakeClient(RuntimeError("provider unavailable"))
    reasoning = _FakeClient(good)
    coordinator = AIRepairCoordinator(
        config=AIRepairConfig(
            enabled=True,
            provider="fixture",
            model="fast",
            api_key="secret",
            max_repair_attempts=2,
        ),
        policy=_policy(),
        primary_client=primary,
        reasoning_client=reasoning,
    )
    attempts = coordinator.request(_empty_bundle(), high_value=True)
    assert len(attempts) == 2
    assert attempts[0].error is not None
    assert attempts[1].proposal is not None
    assert primary.calls[0][0] == "primary"
    assert reasoning.calls[0][0] == "reasoning"


def test_ai_repair_disabled_is_non_blocking() -> None:
    coordinator = AIRepairCoordinator(
        config=AIRepairConfig(enabled=False),
        policy=_policy(),
        primary_client=None,
    )
    assert coordinator.request(_empty_bundle(), high_value=True) == ()


def test_repair_audit_log_and_child_lineage(tmp_path: Path) -> None:
    proposal = RepairProposal(summary="repair", diagnosis="fixture", changes=())
    child = derive_repaired_child(_trial(), proposal_sha256=proposal.canonical_sha256)
    assert child.parent_trial_id == "phase12-trial"
    assert child.config["repair_provenance"] == {"proposal_sha256": proposal.canonical_sha256}
    audit = RepairAuditLog(tmp_path / "repair.jsonl")
    record = RepairAuditRecord(
        trial_id="phase12-trial",
        bundle_sha256="a" * 64,
        tier="primary",
        proposal_sha256=proposal.canonical_sha256,
        outcome="validated_cpu_pending_gpu",
    )
    audit.append(record)
    assert audit.read_all() == (record,)


def _init_git_repo(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "ci@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", str(path), "config", "user.name", "CI"], check=True)
    (path / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (path / "PLAN.md").write_text("frozen\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "base"], check=True)


def test_repair_sandbox_isolated_patch_and_cpu_gates(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    _init_git_repo(repository)
    worktree = tmp_path / "repair-worktree"
    sandbox = RepairSandbox.create(repository_root=repository, worktree_path=worktree)
    try:
        current = (worktree / "model.py").read_bytes()
        proposal = _proposal_for("model.py", current, "VALUE = 2\n")
        assert sandbox.apply(proposal, ProtectedFilePolicy()) == ("model.py",)
        assert "VALUE = 2" in sandbox.diff()
        cpu_only = validate_repair(
            sandbox,
            static_commands=((sys.executable, "-m", "py_compile", "model.py"),),
            unit_commands=((sys.executable, "-c", "import model; assert model.VALUE == 2"),),
            regression_commands=((sys.executable, "-c", "import model; assert model.VALUE < 3"),),
            gpu_smoke_gate=unavailable_gpu_gate(),
        )
        assert cpu_only.static_gate.passed
        assert cpu_only.unit_gate.passed
        assert cpu_only.regression_gate.passed
        assert cpu_only.eligible_for_requeue is False

        synthetic_gpu = GateResult(name="gpu_smoke", passed=True, detail="synthetic policy fixture")
        complete = validate_repair(
            sandbox,
            static_commands=((sys.executable, "-m", "py_compile", "model.py"),),
            unit_commands=((sys.executable, "-c", "import model; assert model.VALUE == 2"),),
            regression_commands=((sys.executable, "-c", "import model; assert model.VALUE < 3"),),
            gpu_smoke_gate=synthetic_gpu,
        )
        assert complete.eligible_for_requeue
    finally:
        sandbox.close()


def test_repair_sandbox_rejects_protected_and_stale_targets(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    _init_git_repo(repository)
    worktree = tmp_path / "repair-worktree"
    sandbox = RepairSandbox.create(repository_root=repository, worktree_path=worktree)
    try:
        frozen = (worktree / "PLAN.md").read_bytes()
        with pytest.raises(PermissionError, match="protected paths"):
            sandbox.apply(
                _proposal_for("PLAN.md", frozen, "changed\n"),
                ProtectedFilePolicy(),
            )
        stale = RepairProposal(
            summary="repair",
            diagnosis="fixture",
            changes=(
                ProposedFileChange(
                    path="model.py",
                    expected_sha256="0" * 64,
                    replacement_text="VALUE = 2\n",
                ),
            ),
        )
        with pytest.raises(ValueError, match="changed before patch"):
            sandbox.apply(stale, ProtectedFilePolicy())
    finally:
        sandbox.close()


def test_phase12_trial_side_states_are_explicit() -> None:
    require_trial_transition(TrialState.RETRYABLE_FAILURE, TrialState.QUARANTINED)
    require_trial_transition(TrialState.QUARANTINED, TrialState.AI_REPAIR_PENDING)
    require_trial_transition(TrialState.AI_REPAIR_PENDING, TrialState.AI_REPAIR_EXHAUSTED)
    with pytest.raises(ValueError):
        require_trial_transition(TrialState.COMPLETE, TrialState.AI_REPAIR_PENDING)


def test_recovery_import_remains_torch_free() -> None:
    command = (
        "import sys; import trading_bot.recovery; "
        "assert 'torch' not in sys.modules, "
        "sorted(k for k in sys.modules if k.startswith('torch'))"
    )
    subprocess.run([sys.executable, "-c", command], check=True)
