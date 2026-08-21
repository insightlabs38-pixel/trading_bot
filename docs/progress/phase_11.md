# Phase 11 Progress — H200 Campaign Scheduler

Last updated: **2026-08-20**

Status: **COMPLETE — CPU/simulation scheduler gate passed**

`IMPLEMENTATION_PLAN.md` remains the authoritative checklist. Phase 11 implements and validates the campaign-controller mechanics that must exist before the paid H200 run. It does not claim that real H200 utilization, VRAM, or throughput values have been measured; those fields are accepted from workers and persisted when target-hardware runs occur later.

## Controller boundary

The scheduler lives under `src/trading_bot/scheduler` and deliberately imports no model or PyTorch/CUDA training modules. A fresh-process acceptance test verifies that importing `trading_bot.scheduler` does not import `torch`.

Trial execution is isolated through `SubprocessTrialRunner`:

- every worker launch is a new subprocess and process group;
- stdout and stderr are captured separately per trial;
- termination is process-group `SIGTERM` followed by a configured grace period and `SIGKILL` escalation;
- retries are represented as new immutable child trials rather than mutating an existing trial row.

Detailed CUDA-OOM/NaN/Triton/hang/storage/evaluator failure classification and circuit-breaker/AI-repair policy remain Phase 12. Phase 11 provides the process-isolation and durable retry lineage that those policies will consume.

## Frozen scheduler operating policy

`configs/campaigns/h200_scheduler_v1.yaml` freezes the scheduler mechanics separately from the Phase 10 experiment manifest:

- initial drain reserve: 90 minutes;
- drain safety margin: 20 minutes;
- runtime estimate quantile: p90-like nearest-rank `0.90`;
- runtime safety multiplier: `1.15`;
- performance-pruning grace: 50% of the current rung budget;
- maximum same-config process retries: 2;
- worker TERM→KILL grace: 45 seconds;
- snapshot interval intent: 5 minutes;
- deadline tiers at 24h / 12h / 6h / 3h / 1.5h usable time;
- one exclusive GPU-trial slot by default;
- two independent CPU-evaluator slots;
- tiny-trial GPU concurrency disabled in v1 unless an alternate validated policy explicitly allows it and measured calibration throughput gain clears the configured threshold.

These values operationalize the existing scheduler/recovery design. They do not change data, evaluation, cost, promotion, final-holdout, or live-trading contracts.

## Durable SQLite authority

`CampaignDB` uses SQLite with foreign keys, WAL mode, `synchronous=FULL`, a busy timeout, and schema versioning. The durable schema includes:

- campaign metadata;
- immutable trials;
- metrics;
- checkpoints;
- runtime statistics;
- events;
- failures;
- promotion lineage;
- resource samples.

Campaign restart verifies the exact canonical Phase 10 manifest SHA-256/JSON and refuses a changed fixed deadline. Trial configs are stored as canonical JSON plus SHA-256; new retries/promotions create child trial rows with explicit root/parent lineage.

A consistent SQLite backup is created with SQLite's backup API, SHA-256 hashed, uploaded through the existing `StorageBackend`, and checksum-verified after upload. Campaign completion fails closed if any active trial remains or any critical unsynced byte count is nonzero.

## Campaign and trial state machines

The persisted campaign state machine covers:

`BOOTSTRAP → CALIBRATION → SCREENING → PROMOTION → OBJECTIVE_SEARCH → FINALISTS → DRAIN → COMPLETE`

with early transition to `DRAIN` allowed from pre-complete campaign states.

The persisted trial state machine covers:

`PENDING`, `STARTING`, `RUNNING`, `EVALUATING`, `SYNCING`, `COMPLETE`, `PRUNED`, `RETRYABLE_FAILURE`, `TERMINAL_FAILURE`, and `INTERRUPTED`.

Invalid state transitions fail closed.

## Phase 10 planning and Phase 6 promotion reuse

The screening planner consumes `configs/campaigns/h200_tournament_v1.yaml` directly. It materializes the frozen **66 screening trials** deterministically, stratifies the first round across searchable families, preserves per-family search-axis opt-in, and gives each immutable trial a stable canonical config hash.

Promotion does not create a new score. It consumes canonical Phase 6 `LeaderboardRow` records, filters to `eligible` rows, preserves the existing `rank` hierarchy, and records explicit parent→child promotion lineage. A grace helper prevents performance pruning before the configured fraction of the rung budget has been consumed.

## Deadline adaptation and resources

Observed partial-budget runtimes are normalized to estimated full-budget runtime by family/scale/context, then the configured p90-like statistic and safety multiplier are applied to the target rung fraction. If no observation exists, a conservative fallback estimate is used.

The adaptive drain reserve is the maximum of the frozen initial reserve and evaluator backlog + estimated durable-sync time + safety margin. If critical unsynced bytes exist but storage throughput is unknown, the reserve becomes effectively infinite and new launches fail closed.

Deadline modes progressively:

1. permit broad exploration;
2. continue normal promotion;
3. stop optional expansion;
4. allow only finalist work;
5. avoid expensive non-finalist work;
6. enter drain and refuse new trials.

GPU utilization and peak-VRAM values are typed and persisted when workers provide them. CPU CI uses synthetic values only; no real H200 telemetry claim is made. Resource-slot tests verify exclusive GPU ownership by default and independent CPU evaluator capacity. The optional tiny-GPU concurrency path cannot open until explicit calibration evidence meets its configured throughput-gain threshold.

## Compressed campaign simulation

The CPU acceptance simulation uses the actual Phase 10 manifest and exercises:

- calibration observations;
- all 66 screening registrations;
- screening launch guards;
- one synthetic process failure and immutable retry child;
- controller/SQLite close and reopen mid-campaign;
- screening → promotion → objective-search → finalist promotion lineage using canonical leaderboard rows;
- finalist-only late scheduling;
- adaptive drain refusal;
- interruption of pending work at drain;
- zero-critical-backlog completion;
- checksum-verified durable final SQLite snapshot.

The restored snapshot is independently queried to verify `COMPLETE` state, expected failure/promotion rows, and intact root promotion lineage.

## CPU verification

Implementation-only PR #22 head: `435f075257104cfdce29be2155c0b0690b562f20`.

Permanent read-only CPU CI run `32440888978` / job `96651209298` tested synthetic merge `1f964bed6edcca988504ab5b89e059bb40cb3df7` against base `3b4f86f5d9d21f87f8183bc66f05c652933706a0`.

Results:

- Ubuntu 24.04.4;
- Python 3.12.3;
- uv 0.10.12;
- 73-package locked `baseline-cpu` environment;
- Ruff passed;
- Ruff format passed: **128 files already formatted**;
- strict mypy passed: **no issues in 70 source files**;
- `compileall` passed;
- pytest: **353 passed, 1 skipped in 18.58s**;
- the sole skip is the existing opt-in Phase 2 real-S3 provider gate requiring external endpoint credentials.

An interim read-only reconciliation run also passed at head `cad5139cd7e5d9c2fe5fa7e83517d78461c09dd7`: run `32441137410` / job `96651906364`, with Ruff clean, **129 files already formatted**, strict mypy clean across 70 source files, `compileall` clean, and **353 passed / 1 skipped in 18.11s**. This interim head still contained the unused branch-only reconciliation helper, which is removed before the final acceptance head.

## Gate

**PASSED — CPU/SIMULATION CAMPAIGN SCHEDULER GATE.** The controller completes the compressed campaign with durable state, correct immutable lineage/promotions, bounded retry behavior, restart continuity, conservative deadline adaptation, finalist-only mode, adaptive drain, and verified durable final state. Phase 12 can now layer deterministic failure classification/recovery and AI-repair isolation onto this scheduler without changing frozen research-selection semantics.
