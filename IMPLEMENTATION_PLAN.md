# Detailed Implementation Plan

Status: **ACTIVE IMPLEMENTATION TRACKER**  
Last updated: **2026-08-18**

This document is the detailed execution plan for `trading_bot`. It is intended to be usable by the repository owner, a future human contributor, or an AI coding agent to answer four questions quickly:

1. **What must be implemented?**
2. **In what order should it be implemented?**
3. **What acceptance criteria must be met before moving forward?**
4. **What is complete, in progress, blocked, or still remaining?**

`PLAN.md` defines the project intent and frozen research decisions. `docs/implementation_roadmap.md` is the high-level milestone view. **This file is the implementation source of truth and progress checklist.**

---

# 0. How to use this document

## Status convention

- `[x]` — complete and acceptance criteria met.
- `[ ]` — not complete.
- `IN PROGRESS` — work has started but the acceptance gate is not met.
- `BLOCKED` — cannot proceed until the stated dependency/decision is resolved.
- `OPTIONAL` — useful but not on the critical path.

A task should not be checked merely because code exists. It is complete only when its tests and acceptance conditions are satisfied.

## Change-control rule

If implementation requires changing a frozen research assumption, do **not** silently modify this document to match the code. First update the relevant design contract or add an ADR under `docs/decisions/`, then update this plan.

Changes that require explicit design review include:

- final-holdout definition;
- train/validation split methodology;
- universe construction methodology;
- feature availability assumptions;
- transaction-cost methodology;
- primary evaluation metrics;
- promotion/leaderboard rules;
- paper-to-live acceptance rules;
- AI repair permissions;
- risk limits or live-trading safety behavior.

## Implementation priority

The critical path is:

```text
contracts
  ↓
data pipeline
  ↓
common research framework
  ↓
baseline models
  ↓
advanced/custom models
  ↓
evaluator + metrics
  ↓
scheduler/recovery
  ↓
Docker/Compose
  ↓
full rehearsal
  ↓
production dataset
  ↓
H200 campaign
  ↓
final holdout
  ↓
paper trading
  ↓
tiny live canary
```

Do not optimize Triton kernels, build live broker integration, or add elaborate dashboards before the upstream critical path is reliable.

---

# 1. Master progress

| Phase | Status | Required before H200? |
|---|---|---|
| 0. Repository/design baseline | **COMPLETE** | Yes |
| 1. Project/config foundations | **IN PROGRESS** | Yes |
| 2. Storage + artifact primitives | Not started | Yes |
| 3. CPU data pipeline | Not started | Yes |
| 4. Dataset validation + leakage protection | Not started | Yes |
| 5. Common training framework | Not started | Yes |
| 6. Evaluation/backtesting framework | Not started | Yes |
| 7. Baseline model families | Not started | Yes |
| 8. Advanced model families | Not started | Yes |
| 9. Custom architectures + Triton | Not started | Yes |
| 10. Experiment configuration/search spaces | Not started | Yes |
| 11. H200 campaign scheduler | Not started | Yes |
| 12. Fault tolerance + AI repair | Not started | Yes |
| 13. Observability + Discord | Not started | Yes |
| 14. Docker/Compose environments | Not started | Yes |
| 15. Campaign simulation + dress rehearsal | Not started | Yes |
| 16. Full production data build | Not started | Yes |
| 17. H200 campaign execution | Not started | N/A |
| 18. Protected final holdout | Not started | Post-campaign |
| 19. Live inference + paper stack | Not started | Post-campaign |
| 20. Paper-trading validation | Not started | Post-campaign |
| 21. Tiny-capital live canary | Not started | Post-paper |

---

# 2. Phase 0 — repository and design baseline

## Goal

Create the architectural and research contracts before implementation begins.

## Completed

- [x] Repository created and accessible.
- [x] Top-level project intent documented in `PLAN.md`.
- [x] Repository rules documented in `AGENTS.md`.
- [x] Architecture diagrams documented in `docs/architecture.md`.
- [x] Data/storage plan documented.
- [x] Model experiment plan documented.
- [x] Evaluation contract documented.
- [x] Scheduler/recovery design documented.
- [x] Paper/live trading plan documented.
- [x] Operations/observability plan documented.
- [x] Reproducibility/security plan documented.
- [x] High-level implementation roadmap documented.
- [x] Module/config/test directory boundaries created.
- [x] Detailed implementation tracker created — this file.

## Gate

**PASSED.** Implementation may begin.

---

# 3. Phase 1 — project and configuration foundations

## Goal

Create the common project infrastructure all later modules depend on.

## Implement

### Python project

- [x] Create `pyproject.toml`.
- [x] Select and pin supported Python version(s).
- [x] Configure `uv` dependency groups, at minimum:
  - [x] core;
  - [x] cpu;
  - [x] gpu;
  - [x] dev/test.
- [x] Configure Ruff.
- [x] Configure pytest.
- [x] Add typing policy/tooling if used.
- [x] Add package metadata and `src/trading_bot` package initialization.

### Configuration system

- [x] Define strongly validated config schemas for:
  - [x] storage;
  - [x] dataset;
  - [x] preprocessing;
  - [x] model;
  - [x] training;
  - [x] objective;
  - [x] evaluation;
  - [x] campaign;
  - [x] scheduler;
  - [x] notifications;
  - [x] AI repair;
  - [x] paper/live risk settings.
- [x] Support environment-variable interpolation for secrets/endpoints.
- [x] Ensure configs can be serialized into immutable run manifests.
- [x] Reject unknown/invalid config fields instead of silently ignoring them.

### Common metadata

- [ ] Define identifiers for:
  - [ ] dataset version;
  - [ ] split version;
  - [ ] model configuration;
  - [ ] trial;
  - [ ] campaign;
  - [ ] checkpoint;
  - [ ] prediction artifact.
- [ ] Implement config hashing/canonical serialization.
- [ ] Implement Git SHA/container/environment capture helpers.

## Tests

- [x] Config round-trip tests.
- [x] Invalid config rejection tests.
- [x] Environment-variable substitution tests.
- [ ] Stable config-hash tests.

### Progress note — 2026-08-18

- Completed: Python project section and Configuration system section.
- Verified by: configuration test suite passes in the available sandbox (`20 passed` under Python 3.13.5, Pydantic 2.13.4, PyYAML 6.0.3, pytest 9.0.2); configuration package/tests also pass `compileall`.
- Contract alignment: AI repair is provider-neutral; paper/live numeric risk limits remain deliberately unfrozen until explicitly configured; evaluation cost configuration requires fee, spread, slippage, and impact components.
- **BLOCKED — target-environment confirmation:** the sandbox does not provide the pinned Python 3.12 runtime, Ruff, or mypy, and package-index access is unavailable. The supported Python 3.12/uv lint/type/test confirmation must therefore be run externally before the Phase 1 gate is declared passed.
- Remaining blocker: Common metadata, stable config hashing/tests, and the minimal run-manifest command.

## Gate

A minimal command can load a validated configuration, generate a run manifest, and exit successfully on any supported machine without requiring market data or a GPU.

---

# 4. Phase 2 — storage and artifact primitives

## Goal

Make local scratch and S3-compatible durable storage interchangeable and reliable before large datasets/checkpoints exist.

## Implement

### Storage abstraction

- [ ] Local backend.
- [ ] S3-compatible backend suitable for GMI Cold Storage.
- [ ] Optional external S3-compatible staging backend.
- [ ] Operations:
  - [ ] list;
  - [ ] exists;
  - [ ] upload;
  - [ ] multipart upload;
  - [ ] download;
  - [ ] copy;
  - [ ] delete;
  - [ ] metadata/head;
  - [ ] checksum verification where practical.
- [ ] Retry/backoff policy.
- [ ] Transfer timeout policy.
- [ ] Atomic/temporary object naming conventions.

### Artifact manifests

- [ ] Define manifest schema containing at least:
  - [ ] path/key;
  - [ ] size;
  - [ ] checksum;
  - [ ] schema/version;
  - [ ] creation time;
  - [ ] producer Git SHA/config hash where relevant.
- [ ] Implement BLAKE3 or SHA-256 generation.
- [ ] Implement manifest verification command.

### Bulk transfer

- [ ] Integrate `rclone`, `s5cmd`, or equivalent for large transfer jobs.
- [ ] Support resumable transfer.
- [ ] Record throughput statistics.

## Tests

- [ ] Local backend unit tests.
- [ ] S3 integration tests against a test bucket or S3-compatible local emulator.
- [ ] Interrupted upload recovery test.
- [ ] Checksum mismatch detection test.
- [ ] Manifest verification test.

## Gate

A generated test artifact can be written locally, uploaded, deleted locally, restored, checksum-verified, and identified by manifest without manual steps.

---

# 5. Phase 3 — CPU data pipeline

## Goal

Build a restartable, provider-aware but provider-independent data pipeline that converts vendor data into causal model-ready datasets.

## Pipeline stages

The intended stage boundary is:

```text
00 raw
01 validated
02 security master
03 adjusted/canonical
04 resampled
05 point-in-time universe
06 features
07 labels
08 immutable splits
09 packed training data
```

Each stage must be restartable and should write a manifest plus success marker only after successful completion.

## Vendor acquisition

- [ ] Define vendor adapter interface.
- [ ] Implement chosen broad-equities vendor adapter once subscription is finalized.
- [ ] Implement targeted Databento or equivalent execution-data adapter if selected.
- [ ] Rate-limit and retry downloads safely.
- [ ] Preserve raw vendor data unchanged where licensing permits.
- [ ] Record exact query/request parameters and download dates.

## Raw validation

- [ ] Validate timestamps/time zones.
- [ ] Detect duplicate rows/events.
- [ ] Detect impossible OHLC relationships.
- [ ] Validate volumes/prices for obvious corruption.
- [ ] Detect unexpected missing sessions/intervals.
- [ ] Record rather than silently repair anomalies unless repair behavior is explicitly defined.

## Security master

- [ ] Permanent/security identifier mapping.
- [ ] Ticker changes.
- [ ] Listing/delisting dates.
- [ ] Security type classification.
- [ ] Exchange metadata.
- [ ] Corporate actions.

## Adjustment/canonicalization

- [ ] Preserve raw prices.
- [ ] Generate explicit adjustment factors/adjusted series where needed.
- [ ] Handle splits.
- [ ] Handle dividends according to documented return convention.
- [ ] Handle corporate-action boundaries causally.

## Resampling

- [ ] Canonical one-minute bars.
- [ ] 5-minute derived bars.
- [ ] 15-minute derived bars.
- [ ] 30-minute derived bars.
- [ ] Daily aggregates as needed.
- [ ] Session/calendar-aware behavior.
- [ ] No cross-session leakage in rolling operations.

## Point-in-time universe

- [ ] Define eligibility filters.
- [ ] Restrict primary universe to intended U.S. common equities.
- [ ] Calculate trailing liquidity using information available at that timestamp.
- [ ] Rank/select approximately 750–1,500 liquid names according to frozen methodology.
- [ ] Freeze membership at documented cadence.
- [ ] Include historical delisted securities where eligible.
- [ ] Save universe membership snapshots.

## Feature pipeline

Implement primitives first, then derived features. Required categories:

- [ ] raw/normalized OHLC/VWAP information;
- [ ] returns at multiple horizons;
- [ ] volume/dollar-volume/relative-volume features;
- [ ] realized volatility features;
- [ ] range/ATR-like features;
- [ ] momentum/trend features;
- [ ] market-relative features;
- [ ] sector-relative features;
- [ ] cross-sectional ranks;
- [ ] time-of-day/session features;
- [ ] liquidity features;
- [ ] market regime inputs;
- [ ] stock/sector identity metadata suitable for embeddings.

Feature code must support the same transformations later in live inference.

## Labels

Prepare at minimum:

- [ ] 5-minute future return;
- [ ] 15-minute future return;
- [ ] 30-minute future return;
- [ ] 60-minute future return;
- [ ] future excess return relative to market/reference;
- [ ] direction labels;
- [ ] cross-sectional rank targets;
- [ ] future volatility target;
- [ ] optional distribution/quantile targets.

Primary model research remains centered on 15m/30m medium-frequency behavior.

## Splits

- [ ] Define chronological walk-forward folds.
- [ ] Define training periods.
- [ ] Define validation periods.
- [ ] Define immutable final holdout.
- [ ] Persist split IDs independently of model code.
- [ ] Add guard preventing routine campaign code from loading final holdout.

## Packing

- [ ] Research representation: Parquet + compression.
- [ ] Training representation optimized for sequential/batched reads.
- [ ] Preserve asset IDs and timestamps with every sample/prediction target.
- [ ] Support memory mapping or equivalent if appropriate.
- [ ] Implement loader throughput benchmark.

## Gate

A small multi-year/multi-asset sample can run raw → packed end-to-end twice and produce equivalent manifests/data within the expected deterministic tolerance. Leakage/security/universe tests pass.

---

# 6. Phase 4 — dataset validation and leakage protection

## Goal

Make data leakage difficult to introduce accidentally.

## Implement tests/invariants

- [ ] No feature uses observations after decision timestamp.
- [ ] No label data enters feature pipeline.
- [ ] Universe membership uses only historical information.
- [ ] Corporate-action processing obeys point-in-time semantics.
- [ ] Train/validation chronology is preserved.
- [ ] Final holdout is inaccessible in normal search mode.
- [ ] Session boundaries are respected.
- [ ] Resampling does not use future bar information.
- [ ] Missing-data behavior is documented and tested.
- [ ] Asset/ticker changes do not splice unrelated securities together.

## Audits

- [ ] Generate dataset summary report.
- [ ] Missingness statistics.
- [ ] Asset counts through time.
- [ ] Universe turnover statistics.
- [ ] Return/volume sanity distributions.
- [ ] Split date visualization/report.

## Gate

No architecture work should proceed on the full dataset until the leakage/data-contract suite passes.

---

# 7. Phase 5 — common training framework

## Goal

Every architecture must train through the same interfaces so architecture comparisons are meaningful.

## Model interface

- [ ] Common base/protocol for models.
- [ ] Standard batch structure.
- [ ] Standard model output containing applicable fields such as:
  - [ ] expected return;
  - [ ] rank score;
  - [ ] direction probability;
  - [ ] volatility;
  - [ ] uncertainty/quantiles.
- [ ] Model parameter-count reporting.
- [ ] Inference timing interface.

## Trainer

- [ ] BF16 default path.
- [ ] FP32/debug path.
- [ ] Optional FP8 finalist path where supported.
- [ ] Gradient accumulation.
- [ ] Gradient clipping.
- [ ] LR scheduling.
- [ ] Early stopping hooks controlled by scheduler rather than hidden model logic.
- [ ] NaN/Inf detection.
- [ ] GPU memory telemetry.
- [ ] Step/time progress heartbeat.
- [ ] Deterministic debug mode.
- [ ] Fast campaign mode.

## Checkpointing

Checkpoint must contain enough information for true continuation:

- [ ] model state;
- [ ] optimizer state;
- [ ] LR scheduler state;
- [ ] training cursor/step;
- [ ] RNG state;
- [ ] precision/scaler state where relevant;
- [ ] model/training config hashes;
- [ ] dataset/split IDs.

Also implement:

- [ ] atomic temporary-write → verify → rename protocol;
- [ ] latest/best bookkeeping;
- [ ] resume validation;
- [ ] checkpoint corruption detection.

## Prediction artifacts

- [ ] Save validation predictions for promoted/final models.
- [ ] Include timestamp, asset ID, target, prediction, relevant metadata.
- [ ] Allow evaluator to rerun without retraining.

## Gate

At least three architecturally different toy/baseline models train, checkpoint, resume, write predictions, and evaluate through the exact same framework.

---

# 8. Phase 6 — canonical evaluator and backtester

## Goal

Implement the frozen evaluation contract once and make every model use it.

## Return accounting

- [ ] Gross portfolio return.
- [ ] Position changes/turnover.
- [ ] Fees.
- [ ] Spread cost.
- [ ] Slippage.
- [ ] Market-impact approximation.
- [ ] Canonical net return/NAV series.

## Predictive metrics

- [ ] Mean Rank IC.
- [ ] Median Rank IC.
- [ ] IC standard deviation.
- [ ] ICIR.
- [ ] Fraction of positive IC periods.
- [ ] IC by fold/regime/horizon.

## Economic metrics

- [ ] Net Sharpe.
- [ ] CAGR.
- [ ] Sortino.
- [ ] Maximum drawdown.
- [ ] Drawdown duration.
- [ ] Calmar.
- [ ] ES95.
- [ ] Worst day.

## Trading-friction metrics

- [ ] Turnover.
- [ ] Total modeled cost.
- [ ] Break-even transaction cost.
- [ ] Trade/rebalance count.
- [ ] Cost stress at multiple multipliers.
- [ ] Spread stress.
- [ ] Latency/execution-delay stress for finalists.

## Robustness

- [ ] Fold-level statistics.
- [ ] Seed dispersion.
- [ ] Positive-fold fraction.
- [ ] Deflated Sharpe Ratio.
- [ ] Probability of Backtest Overfitting or equivalent implementation matching frozen contract.
- [ ] Multiple-testing trial count accounting.

## Attribution

- [ ] Market beta.
- [ ] Common factor exposures.
- [ ] Residual/alpha diagnostics.

## Execution-oriented finalist metrics

- [ ] Implementation-shortfall calculation.
- [ ] Participation/liquidity diagnostics.
- [ ] BBO/L1 execution simulator when execution dataset is available.
- [ ] Market/limit-order abstraction sufficient for medium-frequency tests.

## Leaderboard behavior

- [ ] Hard validity/disqualification gates.
- [ ] Primary predictive/economic ranking hierarchy.
- [ ] No opaque single score as sole selection criterion.
- [ ] Export machine-readable and human-readable reports.

## Tests

- [ ] Hand-calculated metric fixtures.
- [ ] Zero-return strategy.
- [ ] Buy-and-hold fixture.
- [ ] Known-cost fixture.
- [ ] Drawdown fixture.
- [ ] No-lookahead execution timing fixture.

## Gate

Given saved predictions, the evaluator can reproduce the complete leaderboard and all core metrics without importing the training code.

---

# 9. Phase 7 — baseline model families

## Goal

Establish strong simple references before advanced custom research.

## CPU baselines

- [ ] Ridge/Elastic Net.
- [ ] Logistic/regression baseline as appropriate.
- [ ] LightGBM.
- [ ] XGBoost.

## Neural baselines

- [ ] MLP.
- [ ] GRU/LSTM.
- [ ] TCN.
- [ ] Simple causal Transformer.

## Requirements for every family

- [ ] Same dataset/split interface.
- [ ] Same objective configuration interface.
- [ ] Parameter count.
- [ ] Throughput benchmark.
- [ ] Checkpoint/resume.
- [ ] Unit forward/backward tests.
- [ ] Small end-to-end training test.

## Gate

Baseline leaderboard is produced successfully on the rehearsal dataset before advanced architectures are treated as trustworthy.

---

# 10. Phase 8 — advanced model families

Implement the selected core tournament families:

- [ ] PatchTST.
- [ ] iTransformer.
- [ ] Mamba/Mamba-2 family.
- [ ] xLSTM and/or VSN recurrent variants if retained in final experiment matrix.
- [ ] Temporal + cross-sectional Transformer.
- [ ] Temporal + graph model.
- [ ] Selected pretrained time-series foundation-model adapters/reference evaluations.

For each:

- [ ] small configuration;
- [ ] medium configuration;
- [ ] larger scaling configuration where justified;
- [ ] memory/throughput profiling;
- [ ] representative shape tests;
- [ ] consistent prediction heads.

## Gate

Every core architecture can complete a screening-budget run using the same trainer/evaluator and produces comparable artifacts.

---

# 11. Phase 9 — custom architectures and Triton

## Multi-Scale Market Mixer

Implement incrementally:

- [ ] shared feature encoder;
- [ ] short-timescale branch;
- [ ] medium/long temporal branch;
- [ ] gated multi-timescale fusion;
- [ ] cross-sectional stock interaction;
- [ ] market/sector context tokens or equivalent;
- [ ] return head;
- [ ] rank head;
- [ ] volatility head;
- [ ] uncertainty/distributional head.

Ablations must allow major components to be disabled independently.

## Heterogeneous MoE

- [ ] Router.
- [ ] TCN/local expert.
- [ ] state-space/long-memory expert.
- [ ] attention or alternate structural expert.
- [ ] optional frequency-domain expert.
- [ ] router diagnostics/expert utilization logging.

## Custom temporal operator

- [ ] Clear mathematical specification.
- [ ] Correct PyTorch reference implementation first.
- [ ] Forward numerical tests.
- [ ] Gradient tests.
- [ ] Profiling proving it matters enough to optimize.
- [ ] Triton implementation only after reference validation.
- [ ] Triton numerical equivalence across representative shapes/dtypes/strides.
- [ ] Triton fallback to reference path.
- [ ] Throughput and memory benchmark.

## Triton policy

Do not rewrite already-optimized generic GEMM/attention merely to use Triton. Prefer project-specific fused preprocessing/temporal operations where profiling shows a real bottleneck.

## Gate

Custom architecture results must be explainable through ablations. Custom kernels must demonstrate correctness independently of speed improvement.

---

# 12. Phase 10 — experiment configuration and search spaces

## Goal

Freeze the campaign manifest before rental day.

- [ ] Define architecture-family registry.
- [ ] Define small/medium/large canonical configs.
- [ ] Define screening search spaces.
- [ ] Define objective variants:
  - [ ] Huber/excess-return regression;
  - [ ] cross-sectional ranking;
  - [ ] multi-task return + rank + volatility + direction;
  - [ ] multi-horizon variants;
  - [ ] distributional variants where selected.
- [ ] Define learning-rate ranges.
- [ ] Define dropout/regularization ranges.
- [ ] Define context-length choices.
- [ ] Define batch/effective-batch constraints.
- [ ] Define seed policy.
- [ ] Define mandatory vs optional experiment pools.
- [ ] Define screening/promotion/full training budgets.

## Gate

A version-controlled campaign YAML/manifest can enumerate the intended experiment space without editing Python.

---

# 13. Phase 11 — H200 campaign scheduler

## Goal

Build a deadline-aware campaign controller that remains alive when trials fail.

## Persistent state

- [ ] SQLite campaign DB.
- [ ] Campaign metadata table.
- [ ] Trials table.
- [ ] Metrics table.
- [ ] Checkpoints table.
- [ ] Runtime statistics table.
- [ ] Events table.
- [ ] Failures table.
- [ ] Promotion lineage.
- [ ] State snapshot to durable storage.

## Campaign state machine

- [ ] BOOTSTRAP.
- [ ] CALIBRATION.
- [ ] SCREENING.
- [ ] PROMOTION.
- [ ] OBJECTIVE_SEARCH.
- [ ] FINALISTS.
- [ ] DRAIN.
- [ ] COMPLETE.

## Trial state machine

- [ ] PENDING.
- [ ] STARTING.
- [ ] RUNNING.
- [ ] EVALUATING.
- [ ] SYNCING.
- [ ] COMPLETE.
- [ ] PRUNED.
- [ ] RETRYABLE_FAILURE.
- [ ] TERMINAL_FAILURE.
- [ ] INTERRUPTED.

## Process isolation

- [ ] Scheduler never imports/runs CUDA training in-process.
- [ ] Each trial is a separate subprocess/process group.
- [ ] Capture stdout/stderr per trial.
- [ ] TERM → grace → KILL behavior.
- [ ] Fresh Python process after serious CUDA errors.

## Deadline adaptation

- [ ] Fixed campaign deadline.
- [ ] Dynamic drain reserve.
- [ ] Runtime estimator from observed trials.
- [ ] Conservative p90-like launch estimate.
- [ ] Never start work that cannot plausibly complete before drain.
- [ ] Stop low-priority exploration as deadline approaches.
- [ ] Finalist-only mode.
- [ ] Drain mode.

## Successive halving

- [ ] Screening rung.
- [ ] Promotion rung.
- [ ] Full/finalist rung.
- [ ] Grace budget before pruning.
- [ ] Promotion based on frozen evaluation hierarchy.

## Runtime/resource scheduling

- [ ] GPU utilization reporting.
- [ ] Peak VRAM reporting.
- [ ] Support exclusive GPU trials.
- [ ] Optional limited concurrent tiny trials only if calibration proves beneficial.
- [ ] CPU evaluator runs concurrently where safe.

## Gate

In simulation mode, the controller finishes a compressed campaign with correct lineage, promotions, retries, deadline adaptation, and drain behavior.

---

# 14. Phase 12 — deterministic fault tolerance and AI repair

## Deterministic failure classification

- [ ] CUDA OOM.
- [ ] NaN/Inf.
- [ ] transient process crash.
- [ ] Triton compile failure.
- [ ] illegal memory access.
- [ ] stale heartbeat/hang.
- [ ] corrupted data shard.
- [ ] checkpoint corruption.
- [ ] evaluator failure.
- [ ] storage failure.
- [ ] disk pressure.
- [ ] infrastructure/GPU failure cluster.
- [ ] deterministic configuration error.

## Recovery policies

- [ ] Bounded retry count.
- [ ] OOM child trial with reduced microbatch and preserved effective batch where possible.
- [ ] Reference PyTorch fallback for custom kernel failures.
- [ ] Last-good-checkpoint recovery.
- [ ] Evaluator retry independent of trainer.
- [ ] Storage retry independent of training.
- [ ] Quarantine irrecoverable trials.

## Heartbeats/hangs

- [ ] Worker heartbeat file/event.
- [ ] Explicit COMPILING/DATALOADING/TRAINING/CHECKPOINTING states.
- [ ] State-specific timeout thresholds.
- [ ] Hang kill/recovery behavior.

## Circuit breaker

- [ ] Detect repeated infrastructure-like failures in short window.
- [ ] Pause new launches.
- [ ] Run GPU smoke test.
- [ ] Verify disk/data/storage basics.
- [ ] Resume only after health gate passes.

## Golden canary

- [ ] Small known dataset.
- [ ] Small known model/config.
- [ ] Expected finite loss range.
- [ ] Save/load test.
- [ ] Evaluation test.
- [ ] Storage upload test.
- [ ] Throughput baseline for regression detection.

## AI repair service

Primary behavior:

```text
deterministic recovery
       ↓
unresolved trial quarantined
       ↓
H200 proceeds with next known-good work
       ↓
AI repair works asynchronously
```

Implement:

- [ ] Sanitized debugging bundle generator.
- [ ] Secret/data redaction.
- [ ] Fast non-thinking model first tier.
- [ ] Strict client-side latency/output limits.
- [ ] Structured JSON repair schema.
- [ ] Isolated Git worktree/sandbox.
- [ ] Protected-file policy.
- [ ] Static validation gate.
- [ ] Unit-test gate.
- [ ] GPU smoke gate.
- [ ] Regression gate.
- [ ] New child trial for successful repair.
- [ ] Complete audit log of AI request/response/diff/tests.
- [ ] Optional slower reasoning escalation only for high-value unresolved trials.
- [ ] AI failure/unavailability never blocks campaign.

Protected from AI modification at minimum:

- [ ] final-holdout definitions;
- [ ] split definitions;
- [ ] evaluation contract;
- [ ] transaction-cost rules;
- [ ] promotion rules;
- [ ] campaign DB;
- [ ] credentials;
- [ ] cloud-instance controls.

## Gate

Injected unknown/custom-code failures can be quarantined without idling the H200. Successful AI repairs only re-enter the queue after deterministic validation.

---

# 15. Phase 13 — observability and Discord

## Structured local observability

- [ ] System metrics logging.
- [ ] GPU utilization/memory/power/temperature.
- [ ] CPU/RAM/disk/network.
- [ ] Training samples/sec and steps/sec.
- [ ] Storage-sync backlog.
- [ ] Current campaign/trial state.
- [ ] AI repair queue.

DCGM may be used for GPU health/job telemetry where available.

## Discord notifier

- [ ] Webhook client.
- [ ] Webhook URL supplied only at runtime.
- [ ] Local notification spool.
- [ ] Retry/backoff.
- [ ] Notification failure cannot block training.
- [ ] Disable unsafe mentions.

Required event types:

- [ ] campaign start;
- [ ] periodic summary;
- [ ] promotion/finalist selection;
- [ ] automatic failure recovery;
- [ ] circuit breaker activation;
- [ ] unrecoverable condition requiring human attention;
- [ ] drain start;
- [ ] campaign completion.

Avoid epoch-level notification spam.

## Lightweight status report

- [ ] Generate static HTML or Markdown status report periodically.
- [ ] Include leaderboard, current trial, remaining time, failures, storage backlog, and system health.
- [ ] Persist report to durable storage.

## Gate

A user can understand campaign health from Discord summaries while full forensic data remains available in structured local/storage artifacts.

---

# 16. Phase 14 — Docker and Compose

## CPU image

- [ ] Lightweight pinned base.
- [ ] Python/uv.
- [ ] Polars/PyArrow/DuckDB/NumPy/etc.
- [ ] Storage tooling.
- [ ] No unnecessary CUDA stack.

## GPU image

- [ ] Pinned NVIDIA NGC PyTorch base after validation.
- [ ] Preserve compatible NGC PyTorch/CUDA/Triton stack.
- [ ] Transformer Engine where required.
- [ ] Project GPU dependencies.
- [ ] Record image digest.

## Compose

- [ ] Common Compose file.
- [ ] CPU overlay/profile.
- [ ] GPU overlay/profile.
- [ ] Campaign service is sole owner of H200.
- [ ] CPU evaluator sidecar.
- [ ] Sync sidecar.
- [ ] Monitoring/notifier sidecar as appropriate.
- [ ] Bind-mounted local NVMe scratch.
- [ ] Secrets injected at runtime.
- [ ] Health checks.
- [ ] Restart policies.
- [ ] Sufficient shared memory for training/data loading.

## Entry points

Provide simple wrappers for:

- [ ] CPU preprocess.
- [ ] Data verify.
- [ ] Storage sync.
- [ ] GPU bootstrap.
- [ ] GPU smoke test.
- [ ] Campaign simulation.
- [ ] Real campaign.
- [ ] Final drain/verify.

## Gate

A clean compatible machine can clone the repository, inject environment/secrets, and launch the relevant workflow without manual package installation.

---

# 17. Phase 15 — campaign simulation and dress rehearsal

## Scheduler simulation

- [ ] Simulated trial runtimes.
- [ ] Simulated scores.
- [ ] Simulated OOM.
- [ ] Simulated hang.
- [ ] Simulated storage slowdown.
- [ ] Simulated scheduler restart.
- [ ] Simulated deadline compression.

## Real tiny campaign

Run full Compose stack with small data and short budgets.

Intentionally inject:

- [ ] CUDA OOM;
- [ ] NaN;
- [ ] trial hang;
- [ ] Triton compile/runtime failure;
- [ ] corrupted checkpoint;
- [ ] evaluator crash;
- [ ] controller restart;
- [ ] storage outage;
- [ ] storage slowdown;
- [ ] disk pressure;
- [ ] Discord failure;
- [ ] AI repair success;
- [ ] AI repair failure;
- [ ] early drain deadline.

## Acceptance gate

Do **not** rent the H200 for the production campaign until all are true:

- [ ] scheduler recovers unattended;
- [ ] final campaign DB is durable;
- [ ] required checkpoints are durable;
- [ ] leaderboard/report is complete;
- [ ] no unsynced critical artifacts remain;
- [ ] storage verification succeeds;
- [ ] Discord notifications are useful and non-blocking;
- [ ] AI repair cannot modify protected contracts;
- [ ] H200-equivalent GPU worker never waits for AI debugging when other valid work exists.

---

# 18. Phase 16 — production data build and staging

## Compute-path decision

Before full preprocessing, compare actual available choices:

- [ ] Obtain GMI L4 host quote/specs.
- [ ] Obtain external CPU instance price/specs.
- [ ] Run standardized preprocessing benchmark if needed.
- [ ] Select based on effective price/performance, RAM, NVMe, and transfer workflow.

## Full build

- [ ] Acquire full raw dataset.
- [ ] Complete all preprocessing stages.
- [ ] Produce dataset manifest.
- [ ] Produce immutable split manifest.
- [ ] Upload canonical/packed data to GMI Cold Storage.
- [ ] Verify every critical object/checksum.
- [ ] Benchmark cold-storage → compute transfer.
- [ ] Benchmark packed dataloader throughput.

## Gate

The H200 campaign dataset is completely prepared and verified **before** the H200 rental clock begins.

---

# 19. Phase 17 — production H200 campaign

## Pre-start checklist

- [ ] Repository tagged/commit frozen for campaign start.
- [ ] Campaign config frozen.
- [ ] Dataset/split IDs frozen.
- [ ] Storage credentials verified.
- [ ] Discord webhook verified.
- [ ] AI repair API configured or explicitly disabled.
- [ ] Cold-storage artifacts verified.

## Bootstrap

- [ ] H200 visible and healthy.
- [ ] Driver/container compatibility check.
- [ ] Local NVMe health/capacity check.
- [ ] Dataset staging/checksum.
- [ ] Golden canary.
- [ ] Dataloader benchmark.
- [ ] Representative model calibration runs.

## Campaign

- [ ] Calibration stage.
- [ ] Architecture screening.
- [ ] Promotion stage.
- [ ] Objective/target experiments.
- [ ] Finalist seeds/folds.
- [ ] Optional experiments only if schedule permits.

During this window, do **not** introduce new:

- feature families;
- dataset definitions;
- target semantics;
- validation splits;
- cost assumptions;
- leaderboard rules.

Unexpected code fixes must be versioned as child trials/commits where behavior changes.

## Drain

- [ ] Stop new long work according to adaptive deadline reserve.
- [ ] Checkpoint current important trial.
- [ ] Complete queued evaluations.
- [ ] Generate final campaign leaderboard.
- [ ] Generate complete campaign report.
- [ ] Upload campaign DB snapshot.
- [ ] Upload critical checkpoints/predictions/profiler artifacts.
- [ ] Verify storage.
- [ ] Confirm zero critical sync backlog.
- [ ] Send final Discord summary.

## Gate

Campaign completion means **durable and auditable results**, not merely that the GPU rental ended.

---

# 20. Phase 18 — protected final holdout

## Preconditions

- [ ] Winning system/finalists frozen.
- [ ] Architecture frozen.
- [ ] Features frozen.
- [ ] Targets frozen.
- [ ] Portfolio construction frozen.
- [ ] Transaction-cost model frozen.
- [ ] Risk rules frozen.

## Execution

- [ ] Unlock final holdout deliberately.
- [ ] Run evaluation once according to contract.
- [ ] Produce full report regardless of outcome.
- [ ] Record exact code/config/dataset hashes.

## Hard rule

Do not tune the strategy based on final-holdout inspection and then continue calling the same period a holdout.

If the result fails, document the failure and begin a new research cycle with a newly designated future holdout.

---

# 21. Phase 19 — production inference and paper-trading stack

This may begin after the research campaign, but interfaces should remain compatible with historical training.

## Live market data

- [ ] Live market-data adapter.
- [ ] Timestamp/freshness validation.
- [ ] Session/calendar handling.
- [ ] Reconnect/recovery behavior.

## Live features

- [ ] Reuse same feature definitions as research.
- [ ] Explicit online state handling for rolling features.
- [ ] Replay parity tests between historical and live pipeline.

## Model inference

- [ ] Load frozen model artifact.
- [ ] Batched universe inference.
- [ ] p50/p95/p99 latency measurement.
- [ ] Inference timeout/failure behavior.

## Portfolio construction

- [ ] Convert predictions to target positions.
- [ ] Cost/edge threshold.
- [ ] Exposure constraints.
- [ ] Position-size constraints.
- [ ] Liquidity/participation constraints.

## Deterministic risk engine

- [ ] Maximum position size.
- [ ] Gross exposure limit.
- [ ] Net exposure limit.
- [ ] Maximum order value.
- [ ] Daily loss limit.
- [ ] Stale-data rejection.
- [ ] Duplicate-order protection.
- [ ] Session checks.
- [ ] Outstanding-order limits.
- [ ] Kill switch.

Risk rules must not depend on an LLM.

## Broker adapter

- [ ] Submit/cancel/replace abstraction.
- [ ] Order acknowledgement handling.
- [ ] Fill/partial-fill handling.
- [ ] Rejection handling.
- [ ] Reconnect behavior.
- [ ] Broker positions treated as authoritative.
- [ ] Position/order/cash reconciliation.

## Paper accounting

Implement three ledgers:

- [ ] ideal signal ledger;
- [ ] conservative internal execution simulator;
- [ ] broker paper-fill ledger.

## Gate

Historical replay and live shadow mode produce equivalent predictions/decisions from equivalent input states within defined numerical tolerance.

---

# 22. Phase 20 — paper-trading validation

## Shadow mode

- [ ] Minimum initial shadow period completed.
- [ ] No orders sent.
- [ ] Raw input/state hashes saved.
- [ ] Live-vs-replay predictions match.
- [ ] Proposed orders/risk decisions logged.

## Broker paper mode

Target:

- [ ] at least 40 trading days;
- [ ] preferably 60 trading days;
- [ ] at least ~100 rebalance observations;
- [ ] at least ~250 order/fill observations where strategy activity permits.

## Operational acceptance

Require:

- [ ] zero unintended duplicate orders;
- [ ] zero stale-data trades;
- [ ] zero unresolved position mismatches;
- [ ] zero unexplained account-balance mismatches;
- [ ] zero risk-limit bypasses;
- [ ] broker disconnect handled safely;
- [ ] market-data disconnect handled safely;
- [ ] restart/recovery tested;
- [ ] kill switch tested;
- [ ] corporate-action handling validated.

## Deliberate fault injection

Test:

- [ ] network disconnect;
- [ ] broker API disconnect;
- [ ] delayed/stale market data;
- [ ] duplicate callback;
- [ ] partial fill;
- [ ] order rejection;
- [ ] process restart;
- [ ] machine reboot/restart scenario;
- [ ] corrupted local state;
- [ ] inference timeout.

System must recover correctly or fail closed.

## Strategy acceptance

- [ ] Paper performance lies within preregistered historical plausibility range.
- [ ] Paper drawdown remains within preregistered stress band.
- [ ] Turnover broadly matches expected distribution.
- [ ] Exposure behavior matches research expectations.
- [ ] Rank IC shows no obvious catastrophic deterioration.
- [ ] Paper fills are not used to weaken conservative execution assumptions merely because simulator fills look favorable.

## Gate

A material model/feature/portfolio/risk/execution change restarts the relevant paper-validation clock.

---

# 23. Phase 21 — tiny live canary

Proceed only after paper acceptance.

## Pre-live

- [ ] Explicit capital limit selected.
- [ ] Live risk limits more conservative than intended final scale.
- [ ] Kill switch verified.
- [ ] Manual emergency procedure documented.
- [ ] Broker/account permissions verified.

## Canary metrics

- [ ] Real implementation shortfall.
- [ ] Spread/slippage distribution.
- [ ] Fill/rejection behavior.
- [ ] Position reconciliation.
- [ ] Latency distribution.
- [ ] Real turnover.
- [ ] Real costs.
- [ ] Strategy-vs-paper deviation.

## Scale policy

Do not increase capital based on a small number of profitable trades. Scaling requires both operational reliability and statistically plausible strategy behavior over an adequate observation window.

---

# 24. Cross-cutting requirements

These apply throughout implementation.

## Reproducibility

Every material run should record:

- [ ] Git SHA;
- [ ] container image digest;
- [ ] Python version;
- [ ] PyTorch/CUDA/Triton versions where relevant;
- [ ] GPU/CPU hardware summary;
- [ ] dataset version/hash;
- [ ] split ID;
- [ ] config hash;
- [ ] seed;
- [ ] precision/compile mode.

## Security

- [ ] No secrets committed.
- [ ] `.env` excluded.
- [ ] Vendor/broker/storage/Discord/AI keys injected at runtime.
- [ ] AI debugging context sanitized.
- [ ] Licensed market data not sent to external AI APIs unnecessarily.
- [ ] Production broker credentials isolated from research containers where practical.

## Versioning

- [ ] Dataset versions immutable.
- [ ] Campaign configs immutable once campaign starts.
- [ ] Trial configs immutable.
- [ ] Behavior-changing bug fix creates a new child trial/version.
- [ ] AI-generated patches are committed/audited before being treated as valid experiment code.

## Performance discipline

- [ ] Profile before writing custom kernels.
- [ ] Keep H200 training input local/hot whenever practical.
- [ ] CPU evaluation/storage operations must not unnecessarily idle the GPU.
- [ ] Store promoted/final predictions to permit CPU-only reevaluation.
- [ ] Record throughput so unexpected regressions are visible.

---

# 25. Items deliberately deferred / non-goals

Do not allow these to distract from the critical path unless the plan is explicitly revised:

- [ ] **Docker Swarm** — not planned for the single-H200 campaign.
- [ ] **True colocated HFT** — not the trading target.
- [ ] **Full-market L3 data for the primary alpha model** — unnecessary for current medium-frequency scope.
- [ ] **Autonomous multi-agent debate for stock selection** — not planned.
- [ ] **Multi-agent RL execution research** — future extension after core bot works.
- [ ] **Large Grafana/Prometheus deployment** — optional; structured telemetry + Discord + static status is sufficient initially.
- [ ] **Large (>~200M) custom models as default search space** — only if scaling evidence/data justify it.
- [ ] **Custom replacement for optimized generic attention/GEMM** — only if profiling establishes a reason.

---

# 26. Immediate next implementation sequence

When coding begins, the recommended first commits are:

1. [ ] `pyproject.toml`, dependency lock, package skeleton, lint/test configuration.
2. [ ] Validated configuration schemas and run-manifest utilities.
3. [ ] Local + S3 storage abstraction and checksum manifests.
4. [ ] Data vendor interface plus small-sample downloader.
5. [ ] Raw validation/security-master/resampling pipeline.
6. [ ] Point-in-time universe + feature/label/split pipeline.
7. [ ] Dataset leakage test suite and packer/loader benchmark.
8. [ ] Common model/trainer/checkpoint/prediction interfaces.
9. [ ] Canonical evaluator and metric unit tests.
10. [ ] Simple baseline models and an end-to-end research smoke test.
11. [ ] Advanced/core model families.
12. [ ] Custom Market Mixer/reference custom operators.
13. [ ] Campaign scheduler/state DB/fault handling.
14. [ ] Discord/telemetry/sync services.
15. [ ] AI repair sandbox.
16. [ ] CPU/GPU Docker images and Compose orchestration.
17. [ ] Scheduler simulation and full fault-injection dress rehearsal.
18. [ ] Full production data build/staging.
19. [ ] H200 campaign.

This ordering intentionally gets **correctness, data integrity, evaluation, and recovery** working before performance experimentation.

---

# 27. Pre-H200 final readiness checklist

The H200 should not be rented for the real campaign until all of the following can be checked:

- [ ] Full dataset is built and immutable.
- [ ] Final holdout is protected.
- [ ] Data leakage tests pass.
- [ ] Packed loader feeds the GPU efficiently on rehearsal hardware.
- [ ] All core architecture families can train with common interfaces.
- [ ] Custom architectures have reference correctness tests.
- [ ] Any Triton kernels match references.
- [ ] Evaluation metrics pass hand-calculated tests.
- [ ] Cost/slippage model is frozen.
- [ ] Campaign YAML/search spaces are frozen.
- [ ] Scheduler simulation passes.
- [ ] Real shortened campaign passes.
- [ ] OOM recovery passes.
- [ ] Hang recovery passes.
- [ ] Checkpoint corruption recovery passes.
- [ ] Controller restart recovery passes.
- [ ] Storage outage/slowdown handling passes.
- [ ] Circuit breaker passes.
- [ ] Discord alerts work and cannot block training.
- [ ] AI repair sandbox cannot modify protected files/contracts.
- [ ] GMI Cold Storage sync/restore verified.
- [ ] Final adaptive drain produces a durable report and zero critical sync backlog.
- [ ] One-command H200 bootstrap/smoke/campaign flow works.

If any critical checkbox above is not satisfied, fixing it before the paid campaign has higher priority than adding another experimental architecture.

---

# 28. Progress-update protocol

Whenever a major implementation task is completed:

1. update the relevant checkbox(es) in this file;
2. ensure acceptance criteria actually pass;
3. add a short note below the phase if implementation materially differs from the original expectation;
4. reference the relevant commit/PR if useful;
5. if the design itself changed, add/update the appropriate design document or ADR first.

Recommended phase note format:

```markdown
### Progress note — YYYY-MM-DD

- Completed: ...
- Verified by: ...
- Relevant commit/PR: ...
- Remaining blocker: ...
```

The objective is for a future reader to open this file and understand the current project state without reconstructing history from Git logs.
