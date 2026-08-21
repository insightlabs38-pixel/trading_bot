# Phase 12 Progress — Deterministic Fault Tolerance and AI Repair

Last updated: **2026-08-21**

Status: **IN PROGRESS — CPU deterministic recovery/repair-sandbox gate passed; GPU/provider acceptance pending**

`IMPLEMENTATION_PLAN.md` remains the authoritative project checklist. Phase 12 now implements and CPU-validates the deterministic recovery policy, heartbeat/circuit-breaker mechanics, CPU golden canary, and isolated AI-repair boundary that can be exercised safely without CUDA or an external AI provider. This status does **not** claim that a real CUDA OOM, illegal-memory-access incident, Triton failure, H200 health gate, or external AI repair request has been executed successfully.

## Deterministic failure classification

`trading_bot.recovery.classify_failure` maps stable worker evidence to explicit failure classes without model inference. CPU fixtures cover CUDA-OOM signatures, non-finite loss, transient process exits, Triton compilation signatures, illegal-memory-access signatures, stale heartbeats, corrupt input shards, checkpoint corruption, evaluator failures, storage failures, disk pressure, deterministic configuration failures, and unknown failures.

Infrastructure-like classifications are explicit so repeated process/illegal-memory/hang/storage/disk incidents can feed the circuit breaker. CUDA/Triton strings are treated only as deterministic log signatures in CPU CI; no GPU reproduction is inferred from those fixtures.

## Bounded recovery policy and immutable lineage

`configs/campaigns/recovery_policy_v1.yaml` freezes retry/time-limit behavior separately from scientific model/evaluation configuration. Recovery decisions can:

- bound process, non-finite, evaluator, and storage retries;
- derive a lower-microbatch OOM child while preserving effective batch exactly when mathematically possible;
- derive a separate reference-backend child for custom-kernel/Triton failures when a validated reference is declared available;
- resume a last-good or earlier checksum-verified checkpoint;
- retry evaluation independently of training;
- retry durable storage synchronization independently of training;
- kill/retry a stale or seriously failed worker in a fresh process lineage;
- pause launches for corrupt-data/disk-safety incidents;
- quarantine terminal or unresolved failures instead of silently mutating the original trial.

Repair-created trials also use immutable parent/root lineage and record the validated proposal SHA-256 in child configuration provenance.

## Heartbeats and circuit breaker

Workers publish atomic heartbeat JSON with an explicit phase and timestamp. The typed phase set distinguishes startup, compilation, data loading, training, checkpointing, evaluation, and synchronization so the recovery policy can apply state-specific timeout thresholds.

Phase 11 already provides process-group `TERM → grace → KILL` isolation. Phase 12 supplies the stale-heartbeat evidence and deterministic kill/retry decision that layer on top of that process runner.

The circuit breaker:

- counts infrastructure-like failures in a bounded time window;
- opens after the configured threshold and blocks new launches;
- enforces a cooldown;
- requires disk, dataset, storage, and GPU-smoke health results before closing;
- fails closed when the GPU-smoke result is missing or failed.

CPU CI validates the state machine and the disk/data/storage gates. A synthetic successful `gpu_smoke` fixture is used only to verify closing logic; it is **not** real GPU-health evidence.

## CPU golden canary

The CPU canary uses a tiny known deterministic dataset/model pair and verifies:

- exact finite prediction/evaluation behavior;
- zero expected MSE for the known mapping;
- JSON save/load round-trip;
- SHA-256 artifact identity;
- upload through the existing storage abstraction;
- post-upload checksum verification;
- a minimum CPU prediction-throughput baseline suitable for regression detection.

This provides a hardware-independent bootstrap/reference health check. A later H200 bootstrap must still add real CUDA forward/backward/checkpoint health evidence.

## Sanitized AI-repair boundary

The repair service is provider-neutral: scientific and scheduler code do not hardcode an external AI API. The coordinator accepts injected primary and optional reasoning clients with configured timeout/output caps. Provider errors are converted into repair-attempt records and never raised into campaign control.

Debug bundles include only bounded, relevant source/config/environment/tensor-shape/log material. Sanitization:

- recursively redacts secret-like mapping keys;
- redacts bearer credentials, AWS-style access keys, GitHub-style tokens, and secret assignments in text;
- rejects raw/licensed market-data file suffixes from source attachments;
- enforces a configured canonical bundle byte limit.

The structured repair response is strict JSON validated by a typed schema and a maximum response size.

## Repair sandbox and protected contracts

Repair proposals apply only inside a detached Git worktree and must carry the expected SHA-256 of every target file. A changed/missing target fails before application.

The production repair surface is **default deny**. The narrow allow-list is currently limited to:

- `src/trading_bot/models/**`;
- `src/trading_bot/training/trainer.py`.

Everything outside that surface is protected by default, with explicit protected patterns also covering project/research plans, evaluation/data/configuration contracts, campaign configs, scheduler/recovery internals, GitHub workflows, Docker/Compose infrastructure, credentials, and secret-like paths. This prevents a repair proposal from changing final-holdout/split semantics, transaction-cost/evaluation/promotion logic, the campaign DB, credentials, cloud/infrastructure controls, or unrelated architectures.

## Validation and audit

A repair candidate runs in its detached worktree through ordered gates:

1. static/syntax commands;
2. unit-test commands;
3. regression/numerical commands;
4. an explicit GPU-smoke result supplied by the GPU environment.

CPU gates are executable in standard CI. `unavailable_gpu_gate()` is deliberately red, so CPU-only validation can never mark a repaired trial eligible for requeue. Tests additionally use a clearly synthetic green GPU result only to prove the all-gates-required policy branch.

The append-only JSONL audit log fsyncs each record and captures the sanitized debug-bundle identity, repair tier, proposal identity, changed paths, validation result, and outcome. This preserves an auditable request/proposal/diff-target/test lineage without storing secrets or raw licensed data.

## Trial repair states

The scheduler trial state machine now includes:

- `QUARANTINED`;
- `AI_REPAIR_PENDING`;
- `AI_REPAIR_EXHAUSTED`.

Invalid transitions still fail closed. AI repair remains optional recovery infrastructure and does not become campaign authority.

## CPU verification

Hardened implementation head: `5c9da9f4edae005d0106c9d8dc95aa3d87a58697`.

Permanent read-only CPU CI run `32445570993` / job `96664409164` tested synthetic merge `e58b5e4ca9408b0ed278385f1208fcb183af0e13` against base `fe30e99ddd603f86efb2b8a485840fe1d96e3b00`.

Results:

- Ubuntu 24.04.4;
- Python 3.12.3;
- uv 0.10.12;
- unchanged 73-package locked `baseline-cpu` environment;
- GitHub token permissions: contents read / metadata read;
- Ruff passed;
- Ruff format passed: **141 files already formatted**;
- strict mypy passed: **no issues in 81 source files**;
- `compileall` passed through `scripts/verify_cpu.sh`;
- pytest: **388 passed, 1 skipped in 18.37s**;
- the sole skip is the pre-existing opt-in Phase 2 real-S3 provider gate requiring external endpoint credentials.

The preceding semantic CI cycle intentionally caught and forced correction of two security-boundary defects: partial bearer-token redaction and incomplete dot-directory (`.github/**`) protection. A later hardening commit changed repair permissions from a broad deny-list to the default-deny allow-list above; the run cited here validates that hardened final implementation.

## Remaining acceptance

Phase 12 is not production-complete until the external/hardware portions are exercised. Remaining items are:

- real GPU smoke forward/backward/checkpoint health test on compatible CUDA hardware;
- real CUDA OOM recovery and effective-batch continuation evidence;
- real illegal-memory-access/process-health recovery evidence;
- real Triton compile/runtime failure and validated reference fallback on a compatible GPU stack;
- target-H200 circuit-breaker/recovery behavior under injected infrastructure faults;
- at least one configured external AI provider exercised through the sanitized client boundary, including provider timeout/unavailability behavior;
- campaign-level proof that real GPU work continues with another known-good trial while repair executes, which belongs with the Phase 15 fault-injection dress rehearsal.

## Gate

**CPU DETERMINISTIC RECOVERY/REPAIR-SANDBOX GATE PASSED.** Unknown/custom-code failures can be deterministically classified or quarantined, recovery decisions are bounded and lineage-preserving, circuit-breaker/health-gate logic fails closed, and repair proposals cannot re-enter the queue unless static/unit/regression plus an explicit GPU-smoke gate are green. Phase 12 remains **IN PROGRESS** for the real GPU/H200/Triton and external-provider acceptance listed above.