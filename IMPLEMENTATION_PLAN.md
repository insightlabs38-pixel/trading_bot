# Detailed Implementation Plan

Status: **ACTIVE IMPLEMENTATION TRACKER**  
Last updated: **2026-08-20**

This document is the detailed execution plan for `trading_bot`. It is intended to be usable by the repository owner, a future human contributor, or an AI coding agent to answer four questions quickly:

1. **What must be implemented?**
2. **In what order should it be implemented?**
3. **What acceptance criteria must be met before moving forward?**
4. **What is complete, in progress, blocked, or still remaining?**

`PLAN.md` defines the project intent and frozen research decisions. `docs/implementation_roadmap.md` is the high-level milestone view. **This file is the implementation source of truth and progress checklist.**

### Reconciliation note — 2026-08-20

The Phase 0–4 tracker is reconciled below against the current foundation-hardening branch and its supported-runtime CI evidence. Phase 1 is complete; Phase 2 is blocked only on real S3/provider validation; the provider-independent Phase 3/4 CPU implementation is substantially complete, including production exchange-calendar support and the Parquet + Zstd research representation. Remaining Phase 3/4 production gates require external provider choices/data, frozen production methodology/dates, finalized production-dataset validation, or H200 benchmarking.

CPU-only GitHub Actions verification now runs on one standard `ubuntu-latest` hosted runner using Python 3.12 and the committed `uv.lock`. The permanent read-only gate runs Ruff, Ruff format checking, strict mypy, `compileall`, and the full pytest suite. GPU/Triton/H200 checks remain intentionally excluded until compatible GPU infrastructure is available.

---

# 0. How to use this document

## Status convention

- `[x]` — complete and acceptance criteria met for the stated implementation item.
- `[ ]` — not complete, externally unvalidated, or acceptance criteria not yet met.
- `IN PROGRESS` — work has started but the phase acceptance gate is not met.
- `BLOCKED` — cannot close the phase gate until the stated dependency/decision is resolved.
- `OPTIONAL` — useful but not on the critical path.

A task should not be checked merely because code exists. It is complete only when its tests and acceptance conditions are satisfied at the scope implied by the checkbox. A phase may therefore contain many checked implementation items while its production gate remains blocked.

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
| 1. Project/config foundations | **COMPLETE** | Yes |
| 2. Storage + artifact primitives | **BLOCKED — real S3/GMI integration only** | Yes |
| 3. CPU data pipeline | **BLOCKED — provider/frozen-methodology/H200 gates only** | Yes |
| 4. Dataset validation + leakage protection | **BLOCKED — finalized production-data validation only** | Yes |
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
- [x] Add CPU-only GitHub Actions verification on the supported Python 3.12 runtime.
- [x] Commit a Python 3.12-resolved dependency lock for the CPU verification environment.

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

- [x] Define identifiers for:
  - [x] dataset version;
  - [x] split version;
  - [x] model configuration;
  - [x] trial;
  - [x] campaign;
  - [x] checkpoint;
  - [x] prediction artifact.
- [x] Implement config hashing/canonical serialization.
- [x] Implement Git SHA/container/environment capture helpers.

## Tests

- [x] Config round-trip tests.
- [x] Invalid config rejection tests.
- [x] Environment-variable substitution tests.
- [x] Stable config-hash tests.

### Progress note — 2026-08-20

- Completed: Python project/configuration/common-metadata implementation and a reusable `scripts/verify_cpu.sh` verification gate.
- CI: `.github/workflows/cpu-ci.yml` uses one standard `ubuntu-latest` runner, Python 3.12, the locked CPU dependency group, Ruff, Ruff format checking, strict mypy, `compileall`, and the full pytest suite.
- Reproducibility: a Python 3.12-resolved `uv.lock` is committed and CI verifies it with `uv sync --locked --group cpu`.
- Cost control: one job only, 20-minute timeout, concurrency cancellation, no GPU/larger runner, and read-only repository permissions.
- Supported-environment verification is green; later Phase 3 columnar additions increased the authoritative full-suite result to 241 passed with only the opt-in real S3 provider gate skipped.

## Gate

A minimal command can load a validated configuration, generate a run manifest, and exit successfully on any supported machine without requiring market data or a GPU.

**PASSED.** Python 3.12 CPU CI is green, strict lint/type/test verification passes, and the dependency lock is committed.

---

# 4. Phase 2 — storage and artifact primitives

## Goal

Make local scratch and S3-compatible durable storage interchangeable and reliable before large datasets/checkpoints exist.

## Implement

### Storage abstraction

- [x] Local backend.
- [ ] S3-compatible backend suitable for GMI Cold Storage. — **IMPLEMENTED, BLOCKED on real GMI endpoint validation.**
- [ ] Optional external S3-compatible staging backend. — **IMPLEMENTED generically, BLOCKED on selected provider validation.**
- [x] Operations:
  - [x] list;
  - [x] exists;
  - [x] upload;
  - [x] multipart upload;
  - [x] download;
  - [x] copy;
  - [x] delete;
  - [x] metadata/head;
  - [x] checksum verification where practical.
- [x] Retry/backoff policy.
- [x] Transfer timeout policy.
- [x] Atomic/temporary object naming conventions.

### Artifact manifests

- [x] Define manifest schema containing at least:
  - [x] path/key;
  - [x] size;
  - [x] checksum;
  - [x] schema/version;
  - [x] creation time;
  - [x] producer Git SHA/config hash where relevant.
- [x] Implement SHA-256 generation.
- [x] Implement manifest verification command.

### Bulk transfer

- [x] Integrate `rclone`, `s5cmd`, or equivalent for large transfer jobs. — backend-native resumable implementation is the current equivalent path.
- [x] Support resumable transfer.
- [x] Record throughput statistics.

## Tests

- [x] Local backend unit tests.
- [ ] **BLOCKED** — real S3 integration test harness exists but needs a real GMI/test S3 endpoint and credentials.
- [x] Interrupted upload recovery/resume test.
- [x] Checksum mismatch detection test.
- [x] Manifest verification test.

### Progress note — 2026-08-20

- Completed: common storage protocol, atomic local backend, generic S3-compatible backend, required storage operations, retry/backoff, transfer timeouts, temporary publication, multipart upload, checksum verification, artifact manifests, manifest verification CLI, resumable bulk transfer, journals, and throughput statistics.
- Real-provider harness: `tests/integration/test_phase2_s3_provider_gate.py` remains opt-in and fail-safe.
- Provider-specific acceptance is intentionally unchecked until the harness succeeds against GMI Cold Storage and any selected staging provider.

## Gate

A generated test artifact can be written locally, uploaded, deleted locally, restored, checksum-verified, and identified by manifest without manual steps.

**LOCAL FUNCTIONAL GATE PASSED. PRODUCTION PHASE BLOCKED — provider integration only.**

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

- [x] Explicit stage identities exist for `00_raw` through `09_packed_training_data`.
- [x] Stages can publish one or more immutable artifacts through a common restartable runner.
- [x] Success markers publish only after artifacts/manifests/checksums/lineage verify.
- [x] Completed stages fail closed when success metadata or referenced artifacts are corrupt.

## Vendor acquisition

- [x] Define provider-independent vendor adapter interface.
- [ ] Implement chosen broad-equities vendor adapter once subscription is finalized. — **BLOCKED on vendor/API selection.**
- [ ] Implement targeted Databento or equivalent execution-data adapter if selected. — **BLOCKED on provider selection/credentials.**
- [x] Rate-limit and retry downloads safely.
- [x] Preserve raw vendor data unchanged where licensing permits.
- [x] Record exact non-secret query/request parameters and download dates.
- [x] Reject secret-like request/response metadata from durable audit records.
- [x] Provide provider-neutral HTTPS GET transport with runtime-only credential injection.

## Raw validation

- [x] Validate timestamps/time zones.
- [x] Detect duplicate rows/events.
- [x] Detect impossible OHLC relationships.
- [x] Validate volumes/prices/VWAP for obvious corruption.
- [x] Detect unexpected missing sessions/intervals.
- [x] Detect assets with zero rows when expected assets/sessions are supplied.
- [x] Record rather than silently repair anomalies.

## Security master

- [x] Permanent/security identifier mapping.
- [x] Ticker changes.
- [x] Listing/delisting dates.
- [x] Security type classification.
- [x] Exchange metadata.
- [x] Corporate actions.
- [x] Guards against ticker reuse splicing unrelated securities.
- [x] Point-in-time symbol lookup.

## Adjustment/canonicalization

- [x] Preserve raw prices.
- [x] Generate explicit adjustment factors/adjusted series where needed.
- [x] Handle splits causally.
- [x] Handle dividends according to documented return convention.
- [x] Handle corporate-action boundaries causally.
- [x] Independently reject malformed/non-finite source data at canonicalization boundary.

## Resampling

- [x] Canonical one-minute base representation.
- [x] 5-minute derived bars.
- [x] 15-minute derived bars.
- [x] 30-minute derived bars.
- [x] 60-minute derived bars.
- [x] Daily aggregates.
- [x] Session-aware behavior.
- [x] No cross-session leakage.
- [x] Production exchange-calendar session resolver supports real holidays and early closes.
- [x] Non-session bars fail closed when a production exchange-calendar resolver is used.

## Point-in-time universe

- [x] Define eligibility filters in the reference universe policy.
- [x] Restrict the reference universe path to intended common-equity security types.
- [x] Calculate trailing liquidity using only information strictly before rebalance time.
- [ ] Freeze final production ranking/selection cadence and thresholds. — **BLOCKED on frozen production methodology/history.**
- [x] Include historical delisted securities while historically eligible.
- [x] Save versioned universe membership snapshots.

## Feature pipeline

Reference causal implementation covers the required categories:

- [x] raw/normalized OHLC/VWAP information;
- [x] returns at multiple horizons;
- [x] volume/dollar-volume/relative-volume features;
- [x] realized volatility features;
- [x] range/ATR-like features;
- [x] momentum/trend features;
- [x] market-relative features;
- [x] sector-relative features;
- [x] cross-sectional ranks;
- [x] time-of-day/session features;
- [x] liquidity features;
- [x] market regime inputs;
- [x] stock/sector identity metadata suitable for embeddings.
- [x] Prefix-invariance regression tests protect against future observations changing earlier features.
- [x] Derived features fail closed on NaN/Inf/overflow.

Feature code must support the same transformations later in live inference; live parity remains a Phase 19 acceptance item.

## Labels

Prepare at minimum:

- [x] 5-minute future return;
- [x] 15-minute future return;
- [x] 30-minute future return;
- [x] 60-minute future return;
- [x] future excess return relative to market/reference;
- [x] direction labels;
- [x] cross-sectional rank targets;
- [x] future volatility target;
- [x] optional distribution/quantile/rank targets.
- [x] Missing future endpoints are not interpolated.
- [x] Non-finite derived labels fail closed.

Primary model research remains centered on 15m/30m medium-frequency behavior.

## Splits

- [x] Define chronological walk-forward fold schema.
- [x] Define training periods structurally.
- [x] Define validation periods structurally.
- [x] Define immutable protected final-holdout structure.
- [x] Persist split IDs independently of model code.
- [x] Add default-deny guard preventing routine research code from loading final holdout dates.
- [x] Routine split views physically omit protected holdout dates.
- [ ] Freeze final production train/validation/holdout dates. — **BLOCKED on finalized data period.**

## Packing

- [x] Research representation: Parquet + Zstd with deterministic ordering, exact timestamps, float32 feature/target columns, semantic metadata, checksummed manifests, and fail-closed validation.
- [x] Deterministic NumPy `.npy` memory-mapped reference training representation.
- [x] Preserve asset IDs and exact timestamps with every sample/prediction target.
- [x] Support memory mapping.
- [x] Implement loader throughput benchmark.
- [x] Verify array file sizes/checksums before opening a pack.
- [x] Persist and verify a SHA-256 sidecar for semantic pack metadata so dataset/split/feature metadata tampering is detected.
- [ ] Freeze final H200 loader representation after representative target-hardware benchmarking.

### Progress note — 2026-08-20

- The Phase 3 checklist is reconciled with the implemented reference pipeline and `docs/progress/phase_03.md`.
- Production `exchange_calendars` support resolves actual XNYS holidays and early closes and drives per-date resampling session lengths.
- The NumPy memmap pack verifies array and semantic metadata integrity with SHA-256 sidecars.
- The Parquet + Zstd research representation is implemented and CPU-CI validated through PyArrow, Polars, and DuckDB with deterministic ordering, exact timestamps, semantic metadata, and checksum/tamper detection.
- The supported Python 3.12 CPU gate passes 241 tests; the only skip is the opt-in real S3 provider test.
- Remaining blockers are external/frozen/hardware items: concrete vendor adapters/credentials, final universe methodology, final split dates, and H200 loader-format benchmarking.

## Gate

A small multi-year/multi-asset sample can run raw → packed end-to-end twice and produce equivalent manifests/data within the expected deterministic tolerance. Leakage/security/universe tests pass.

**REFERENCE/SYNTHETIC CPU GATE PASSED IN PYTHON 3.12 CI. PRODUCTION PHASE BLOCKED only on the external/finalization/H200 items above.**

---

# 6. Phase 4 — dataset validation and leakage protection

## Goal

Make data leakage difficult to introduce accidentally.

## Implement tests/invariants

- [x] No feature uses observations after decision timestamp.
- [x] No label data enters feature pipeline.
- [x] Universe membership uses only historical information.
- [x] Corporate-action processing obeys point-in-time semantics.
- [x] Train/validation chronology is preserved.
- [x] Final holdout is inaccessible in normal search mode and omitted from routine views.
- [x] Session boundaries are respected.
- [x] Resampling does not use future bar information.
- [x] Missing-data behavior is documented and tested.
- [x] Asset/ticker changes do not splice unrelated securities together.
- [x] Future cross-sectional panel additions cannot alter earlier feature rows.
- [x] Exchange holidays/early closes can be sourced from the production calendar dependency.

## Audits

- [x] Generate dataset summary report.
- [x] Missingness and malformed-observation statistics.
- [x] Asset counts through time.
- [x] Universe turnover statistics.
- [x] Return/volume sanity distributions with overflow/non-finite handling.
- [x] Split timeline/report without exposing final-holdout dates to routine workflows.
- [x] Deterministic canonical JSON and human-readable Markdown report output.

### Progress note — 2026-08-20

- The Phase 4 checklist is reconciled with the implemented leakage/audit suite and `docs/progress/phase_04.md`.
- Exchange-calendar holiday/early-close behavior and the Parquet + Zstd research representation are now exercised by the supported Python 3.12 CPU environment.
- The complete CPU suite passes 241 tests with only the opt-in real S3 provider test skipped.
- The production Phase 4 gate remains blocked because the final provider-derived dataset and frozen production universe/split definitions do not yet exist; the exact leakage/audit suite must be rerun on that frozen dataset and final representation.

## Gate

No architecture work should proceed on the full production dataset until the leakage/data-contract suite passes on that exact frozen dataset and representation.

**REFERENCE/SYNTHETIC LEAKAGE AND AUDIT GATES ARE IMPLEMENTED. PRODUCTION PHASE BLOCKED — finalized production-data validation only.**

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

- [x] Git SHA support in run manifests.
- [ ] container image digest — capture support exists but production image digest gate remains later.
- [x] Python version support in run manifests.
- [ ] PyTorch/CUDA/Triton versions where relevant — capture hooks exist; GPU stack not yet validated.
- [x] GPU/CPU hardware/runtime summary support where available.
- [x] dataset version/hash lineage support.
- [x] split ID lineage support.
- [x] config hash.
- [x] seed.
- [x] precision/compile mode.

## Security

- [x] No secrets committed by project policy/tests.
- [x] `.env` excluded.
- [x] Vendor/storage/Discord/AI keys designed for runtime injection.
- [ ] Broker credential runtime integration — broker layer not implemented yet.
- [ ] AI debugging context sanitizer — Phase 12.
- [x] Vendor acquisition rejects secret-like durable request/response metadata.
- [ ] Licensed market data external-AI redaction gate — Phase 12.
- [ ] Production broker credentials isolated from research containers — Phase 14/19.

## Versioning

- [x] Dataset/stage artifact versioning primitives are immutable/content-addressed.
- [ ] Campaign configs immutable once campaign starts — enforcement belongs to Phase 10/11.
- [ ] Trial configs immutable — enforcement belongs to Phase 5/11.
- [ ] Behavior-changing bug fix creates a new child trial/version — scheduler enforcement not yet implemented.
- [ ] AI-generated patches are committed/audited before being treated as valid experiment code — Phase 12.

## Performance discipline

- [ ] Profile before writing custom kernels.
- [ ] Keep H200 training input local/hot whenever practical.
- [ ] CPU evaluation/storage operations must not unnecessarily idle the GPU.
- [ ] Store promoted/final predictions to permit CPU-only reevaluation.
- [x] Reference packed loader records throughput.

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

Current recommended sequence, reconciled to implemented work:

1. [x] `pyproject.toml`, package skeleton, lint/test/type configuration.
2. [x] Validated configuration schemas and run-manifest utilities.
3. [x] Local + generic S3 storage abstraction, artifact manifests, and resumable transfer.
4. [x] Provider-independent data vendor interface and provider-neutral downloader transport.
5. [x] Raw validation/security-master/canonicalization/resampling reference pipeline.
6. [x] Point-in-time universe + feature/label/split reference pipeline.
7. [x] Dataset leakage/audit suite and deterministic reference packer/loader benchmark.
8. [x] Add supported-Python CPU CI and close high-value foundation integrity gaps.
9. [ ] Commit/freeze a dependency lock strategy for reproducible CPU verification.
10. [ ] Common model/trainer/checkpoint/prediction interfaces.
11. [ ] Canonical evaluator and metric unit tests.
12. [ ] Simple baseline models and an end-to-end research smoke test.
13. [ ] Advanced/core model families.
14. [ ] Custom Market Mixer/reference custom operators.
15. [ ] Campaign scheduler/state DB/fault handling.
16. [ ] Discord/telemetry/sync services.
17. [ ] AI repair sandbox.
18. [ ] CPU/GPU Docker images and Compose orchestration.
19. [ ] Scheduler simulation and full fault-injection dress rehearsal.
20. [ ] Full production data build/staging.
21. [ ] H200 campaign.

External Phase 2/3/4 production blockers should be closed as credentials/data/hardware become available, but they do not prevent CPU/reference implementation of the common training/evaluation framework.

---

# 27. Pre-H200 final readiness checklist

The H200 should not be rented for the real campaign until all of the following can be checked:

- [ ] Full dataset is built and immutable.
- [x] Final-holdout access primitives are protected; final production dates still need freezing.
- [x] Leakage regression suite exists; it must still pass the finalized production dataset.
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
