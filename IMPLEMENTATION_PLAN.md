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

The Phase 0–11 tracker is reconciled below against supported-runtime CI evidence. Phase 1 is complete; Phase 2 is blocked only on real S3/provider validation; the provider-independent Phase 3/4 CPU implementation is substantially complete; the Phase 5 CPU training-framework gate has passed while GPU acceptance remains open; the Phase 6 CPU/reference evaluator gate has passed while real factor/BBO dataset validation remains external; Phase 7 is complete on the CPU/reference baseline gate; the Phase 8 CPU/reference core-architecture gate has passed while selected pretrained-foundation and GPU/H200 acceptance remain open; the Phase 9 CPU/reference custom-architecture/correctness gate has passed while Triton and target-GPU optimization acceptance remain open; Phase 10 is complete on the version-controlled campaign search-manifest gate; and Phase 11 is complete on the CPU/simulation deadline-aware scheduler gate. Real H200 utilization/VRAM/throughput values remain target-hardware observations for the production campaign, while deterministic failure classification, recovery/circuit-breaker policy, and AI repair remain Phase 12. Remaining open items require external provider/data choices, frozen production methodology/dates, finalized production-data validation, selected external model artifacts, GPU/H200/Triton acceptance, or later fault-tolerance/operations integration.

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
| 5. Common training framework | **IN PROGRESS — CPU gate passed; GPU acceptance pending** | Yes |
| 6. Evaluation/backtesting framework | **IN PROGRESS — CPU gate passed; real factor/BBO data acceptance pending** | Yes |
| 7. Baseline model families | **COMPLETE — CPU/reference baseline gate passed** | Yes |
| 8. Advanced model families | **IN PROGRESS — CPU/reference core gate passed; foundation/GPU acceptance pending** | Yes |
| 9. Custom architectures + Triton | **IN PROGRESS — CPU/reference custom gate passed; Triton/GPU acceptance pending** | Yes |
| 10. Experiment configuration/search spaces | **COMPLETE — campaign search-manifest gate passed** | Yes |
| 11. H200 campaign scheduler | **COMPLETE — CPU/simulation scheduler gate passed** | Yes |
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
- Reproducibility: a Python 3.12-resolved `uv.lock` is committed and CI verifies it with `uv sync --locked --group baseline-cpu`.
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

- [x] Common base/protocol for models.
- [x] Standard batch structure.
- [x] Standard model output containing applicable fields such as:
  - [x] expected return;
  - [x] rank score;
  - [x] direction probability;
  - [x] volatility;
  - [x] uncertainty/quantiles.
- [x] Model parameter-count reporting.
- [x] Inference timing interface.

## Trainer

- [x] BF16 default path — CPU autocast is covered by the supported Python 3.12 CI path; GPU BF16 acceptance remains hardware-specific.
- [x] FP32/debug path.
- [ ] Optional FP8 finalist path where supported. — **GPU-DEPENDENT; intentionally deferred until compatible hardware/software is configured.**
- [x] Gradient accumulation.
- [x] Gradient clipping.
- [x] LR scheduling.
- [x] Early stopping hooks controlled by scheduler rather than hidden model logic.
- [x] NaN/Inf detection.
- [ ] GPU memory telemetry. — typed CUDA allocator telemetry is implemented in the heartbeat contract, but real CUDA measurements remain **GPU-DEPENDENT** and unvalidated.
- [x] Step/time progress heartbeat.
- [x] Deterministic debug mode.
- [x] Fast campaign mode.

## Checkpointing

Checkpoint must contain enough information for true continuation:

- [x] model state;
- [x] optimizer state;
- [x] LR scheduler state;
- [x] training cursor/step;
- [x] RNG state;
- [x] precision/scaler state where relevant — generic precision-state plumbing is persisted; CUDA-specific scaler acceptance remains a GPU follow-up;
- [x] model/training config hashes;
- [x] dataset/split IDs.

Also implement:

- [x] atomic temporary-write → verify → rename protocol;
- [x] latest/best bookkeeping;
- [x] resume validation;
- [x] checkpoint corruption detection.

## Prediction artifacts

- [x] Save validation predictions for promoted/final models.
- [x] Include timestamp, asset ID, target, prediction, relevant metadata.
- [x] Allow evaluator to rerun without retraining.

### Progress note — 2026-08-20

- Common contracts: `TrainingBatch`, `ModelOutput`, and `TradingModel` standardize architecture inputs/outputs; shared parameter-count and inference-timing interfaces are implemented.
- Trainer: BF16 is the default, FP32/debug is explicit, and the common loop covers gradient accumulation/clipping, stateful LR schedulers, external early-stop control, non-finite checks, heartbeat reporting, deterministic-debug mode, and fast-campaign mode.
- Telemetry: heartbeat plumbing includes allocated/reserved/peak CUDA allocator counters when running on CUDA and explicitly reports no GPU telemetry on CPU. The GPU-memory checklist remains unchecked until exercised on real CUDA hardware.
- Checkpointing: model/optimizer/scheduler/cursor/RNG/precision state and run identity are checksummed and atomically published; latest/best pointers, resume validation, and corruption detection are covered. A dropout regression reproduces the exact next optimizer-step parameters after restore on CPU.
- Predictions: evaluator-independent immutable Parquet + Zstd artifacts preserve exact asset/timestamp/target/prediction lineage and fail closed on tampering.
- Gate models: affine linear, nonlinear GELU/dropout MLP, and residual gated mixer models all train, checkpoint, reconstruct, resume, predict, and pass a trainer-independent saved-artifact MSE evaluation smoke through the same framework.
- Dependency isolation: the `training-cpu` group uses the official CPU-only PyTorch index for CPU CI while the production `gpu` group remains PyTorch-free so the pinned NGC image retains ownership of the CUDA stack.
- Supported environment: the reconciled read-only Python 3.12 PR head resolved PyTorch 2.13.0+cpu, passed lock freshness, Ruff, format, strict mypy across 41 source files, compileall, and 252 tests with only the opt-in Phase 2 real-S3 provider gate skipped.
- Detailed status and remaining GPU-only acceptance items are recorded in `docs/progress/phase_05.md`.

## Gate

At least three architecturally different toy/baseline models train, checkpoint, resume, write predictions, and evaluate through the exact same framework.

**CPU FRAMEWORK GATE PASSED.** Three distinct toy architectures satisfy the CPU-verifiable gate, including true checkpoint continuation and trainer-independent saved-prediction evaluation. Phase 5 remains **IN PROGRESS** only for real CUDA memory telemetry, optional FP8, CUDA-specific precision/scaler acceptance, and representative GPU/H200 performance validation. Canonical predictive/economic/backtest metrics remain Phase 6.

---

# 8. Phase 6 — canonical evaluator and backtester

## Goal

Implement the frozen evaluation contract once and make every model use it.

## Return accounting

- [x] Gross portfolio return.
- [x] Position changes/turnover.
- [x] Fees.
- [x] Spread cost.
- [x] Slippage.
- [x] Market-impact approximation.
- [x] Canonical net return/NAV series.

## Predictive metrics

- [x] Mean Rank IC.
- [x] Median Rank IC.
- [x] IC standard deviation.
- [x] ICIR.
- [x] Fraction of positive IC periods.
- [x] IC by fold/regime/horizon, with sector breakdown support as well.

## Economic metrics

- [x] Net Sharpe.
- [x] CAGR.
- [x] Sortino.
- [x] Maximum drawdown.
- [x] Drawdown duration.
- [x] Calmar.
- [x] ES95 using the frozen loss-tail convention.
- [x] Worst day.

## Trading-friction metrics

- [x] Turnover.
- [x] Total modeled cost.
- [x] Break-even transaction cost.
- [x] Trade/rebalance count.
- [x] Cost stress at multiple multipliers.
- [x] Spread stress.
- [x] Latency/execution-delay stress for finalists through explicit delayed-return inputs; real execution-data acceptance remains external.

## Robustness

- [x] Fold-level statistics.
- [x] Seed dispersion.
- [x] Positive-fold fraction.
- [x] Deflated Sharpe Ratio.
- [x] Probability of Backtest Overfitting via a tested CSCV-style reference implementation matching the frozen diagnostic intent.
- [x] Multiple-testing trial count accounting.

## Attribution

- [x] Market beta through generic OLS over caller-supplied factor observations.
- [x] Common factor exposures through the same provider-independent attribution interface.
- [x] Residual/alpha diagnostics.

Real factor-provider dataset acceptance remains external; Phase 6 does not silently choose a production factor vendor.

## Execution-oriented finalist metrics

- [x] Implementation-shortfall calculation with adverse-execution sign convention.
- [x] Participation/liquidity diagnostics.
- [x] BBO/L1 market/limit execution simulator reference path with deterministic synthetic/no-lookahead fixtures; real execution-dataset acceptance remains external.
- [x] Market/limit-order abstraction sufficient for medium-frequency tests.

## Leaderboard behavior

- [x] Hard validity/disqualification gates.
- [x] Primary predictive/economic ranking hierarchy.
- [x] No opaque single score as sole selection criterion.
- [x] Export deterministic, checksummed machine-readable JSON and human-readable Markdown reports.

## Tests

- [x] Hand-calculated metric fixtures.
- [x] Zero-return strategy.
- [x] Buy-and-hold fixture.
- [x] Known-cost fixture.
- [x] Drawdown fixture.
- [x] No-lookahead execution timing fixture.
- [x] Fresh-process saved-prediction gate reproduces core metrics, leaderboard, and report without importing `trading_bot.training` or `torch`.

### Progress note — 2026-08-20

- Canonical return accounting, predictive/economic/friction metrics, robustness diagnostics, provider-independent factor attribution, execution diagnostics/simulation, validity gates, leaderboard ordering, and checksummed reports are implemented under `src/trading_bot/evaluation`.
- Portfolio construction remains an explicit frozen evaluator input rather than being invented inside Phase 6.
- The evaluator-side Phase 5 Parquet reader verifies checksums/metadata/Zstd and reproduces the complete CPU leaderboard/report path in a fresh process without importing training code or PyTorch.
- The supported Python 3.12 read-only PR gate passes lock freshness, Ruff, formatting, strict mypy across 50 source files, compileall, and 263 tests with only the opt-in Phase 2 real-S3 provider gate skipped.
- Detailed status and external-data acceptance items are recorded in `docs/progress/phase_06.md`.

## Gate

Given saved predictions, the evaluator can reproduce the complete leaderboard and all core metrics without importing the training code.

**CPU/REFERENCE EVALUATOR GATE PASSED.** The exact fresh-process gate is covered in Python 3.12 CI. Phase 6 remains **IN PROGRESS** only for real common-factor data validation and real BBO/L1 execution-dataset acceptance; no provider-specific production-data result is claimed by the CPU gate.

---

# 9. Phase 7 — baseline model families

## Goal

Establish strong simple references before advanced custom research.

## CPU baselines

- [x] Ridge/Elastic Net — both Ridge and Elastic Net reference estimators are implemented.
- [x] Logistic/regression baseline as appropriate — logistic direction classification and linear return regression are covered.
- [x] LightGBM.
- [x] XGBoost through the official CPU-only Python distribution.

## Neural baselines

- [x] MLP.
- [x] GRU/LSTM — both GRU and LSTM references are implemented.
- [x] TCN with causal left-only temporal convolution.
- [x] Simple causal Transformer.

## Requirements for every family

- [x] Same dataset/split interface through identity-preserving `TrainingBatch` views and `BaselineSplit`.
- [x] Same validated `ObjectiveConfig` configuration boundary.
- [x] Parameter/learned-state count — neural parameter count; linear coefficient/intercept count; tree node count plus serialized bytes.
- [x] Throughput benchmark on the CPU reference path without freezing a hardware-specific threshold.
- [x] Checkpoint/resume — true optimizer-step continuation for neural families and checksummed fitted-state reconstruction for classical estimators.
- [x] Unit forward/backward tests for differentiable neural families; classical algorithms use fit/predict tests because backpropagation is not defined for those estimators.
- [x] Small end-to-end training/fit, prediction-artifact, evaluator, and leaderboard test for every concrete family.

### Progress note — 2026-08-20

- Classical families: Ridge, Elastic Net, logistic direction, LightGBM, and XGBoost all fit from the same rehearsal split, checkpoint/reconstruct, emit common prediction artifacts, and enter the canonical Phase 6 evaluator.
- Neural families: MLP, GRU, LSTM, TCN, and causal Transformer all perform finite forward/backward, train through the Phase 5 `Trainer`, checkpoint at optimizer step 2, reconstruct/restore, continue through step 4, and publish common prediction artifacts.
- Objective interface: the validated shared schema covers excess-return MSE/Huber, direction BCE, same-timestamp pairwise ranking, and composite return/rank/direction multitask behavior where applicable; incompatible family/objective combinations fail closed.
- Rehearsal gate: all ten family entries share one dataset/split identity and one deterministic rank-to-portfolio rule, then produce one cost-aware canonical Phase 6 baseline leaderboard and checksummed report.
- Dependency isolation: `baseline-cpu` layers LightGBM, scikit-learn, and the official `xgboost-cpu` package on the existing CPU training environment while leaving the production GPU dependency group unchanged.
- Supported Python 3.12 read-only CI passes lock freshness, Ruff, formatting across 104 files, strict mypy across 55 source files, compileall, and 268 tests with only the unrelated opt-in Phase 2 real-S3 provider gate skipped.
- Detailed implementation evidence is recorded in `docs/progress/phase_07.md`.

## Gate

Baseline leaderboard is produced successfully on the rehearsal dataset before advanced architectures are treated as trustworthy.

**PASSED — CPU/REFERENCE BASELINE GATE.** Ten concrete baseline entries spanning classical linear/tree estimators and neural sequence models train or fit, restore from durable checkpoints, publish common prediction artifacts, and produce one canonical rehearsal leaderboard in Python 3.12 CPU CI. Phase 8 may proceed without claiming GPU/H200 performance from this CPU gate.

---

# 10. Phase 8 — advanced model families

Implement the selected core tournament families:

- [x] PatchTST — dependency-light channel-independent patch Transformer CPU reference.
- [x] iTransformer — variable-token Transformer CPU reference.
- [x] Mamba/Mamba-2 family — pure-PyTorch selective state-space correctness/screening reference; fused Mamba-2 kernel and H200 performance equivalence remain GPU-dependent.
- [x] xLSTM and/or VSN recurrent variants if retained in final experiment matrix — VSN + LSTM recurrent reference implemented.
- [x] Temporal + cross-sectional Transformer with explicit same-decision-timestamp batch guard.
- [x] Temporal + graph model with same-timestamp learned similarity/top-k message passing.
- [ ] Selected pretrained time-series foundation-model adapters/reference evaluations — typed checksum-identified frozen-backbone adapter is implemented and CPU-tested, but a real selected pretrained checkpoint has not yet been chosen/evaluated.

For each CPU-reference core family:

- [x] small configuration;
- [x] medium configuration;
- [x] larger scaling configuration where justified — deterministic reference specs/shape tests exist; paid-campaign sizing remains subject to GPU/H200 profiling;
- [x] memory/throughput profiling — exact model-state bytes and CPU inference timing are covered; GPU peak-memory/H200 throughput remains external;
- [x] representative shape tests across small/medium/large specs;
- [x] consistent expected-return/rank/direction/volatility/uncertainty prediction heads.

### Progress note — 2026-08-20

- Six dependency-light PyTorch core families are implemented under `src/trading_bot/models/advanced.py` and exported through the common model package.
- Every core family consumes the common Phase 5 `TrainingBatch`, emits `ModelOutput`, performs finite forward/backward, trains through `Trainer`, checkpoints at optimizer step 2, reconstructs/restores, and continues through optimizer step 4.
- Every family publishes the common Parquet + Zstd prediction artifact and enters one canonical Phase 6 cost-aware evaluator/leaderboard/report rehearsal on the same deterministic split and portfolio rule.
- Deterministic small/medium/large specs, exact parameter/model-state byte accounting, CPU inference timing, and representative scale-shape tests are implemented without adding new runtime dependencies.
- Same-timestamp guards prevent cross-sectional/graph models from silently mixing different decision timestamps.
- The foundation-model boundary is offline and fail-closed: it requires a caller-supplied checksum-identified backbone, freezes that backbone, and trains only the adapter/common heads. Real pretrained checkpoint selection/licensing/evaluation remains external.
- Supported Python 3.12 read-only CI run `32433083576` / job `96628678153` passed lock freshness, Ruff, formatting across 108 files, strict mypy across 56 source files, compileall, and 279 tests with only the unrelated opt-in Phase 2 real-S3 provider gate skipped.
- Detailed scope and remaining external acceptance are recorded in `docs/progress/phase_08.md`.

## Gate

Every core architecture can complete a screening-budget run using the same trainer/evaluator and produces comparable artifacts.

**CPU/REFERENCE CORE ARCHITECTURE GATE PASSED.** Six trainable core families complete the common screening rehearsal in Python 3.12 CPU CI and produce comparable durable artifacts through the canonical evaluator. Phase 8 remains **IN PROGRESS** only for a real selected pretrained foundation-model checkpoint/reference evaluation and representative GPU/H200 memory/throughput acceptance; no such external result is claimed by this CPU gate.

---

# 11. Phase 9 — custom architectures and Triton

## Multi-Scale Market Mixer

Implement incrementally:

- [x] shared feature encoder;
- [x] short-timescale branch — causal depthwise/pointwise temporal convolution reference;
- [x] medium/long temporal branch — learnable causal multi-decay temporal reference;
- [x] gated multi-timescale fusion;
- [x] cross-sectional stock interaction with a same-decision-timestamp guard;
- [x] market/sector context tokens or equivalent — same-timestamp market-mean context is implemented without inventing unavailable sector IDs;
- [x] return head;
- [x] rank head;
- [x] volatility head;
- [x] uncertainty/distributional head — uncertainty head is implemented in the common reference output.

Ablations allow major components to be disabled independently through the stable `full`, `no_short`, `no_long`, `no_gated_fusion`, `no_cross_sectional`, and `no_market_context` suite.

## Heterogeneous MoE

- [x] Router — sparse top-k routing informed by sample and same-timestamp market state.
- [x] TCN/local expert.
- [x] state-space/long-memory expert — multi-decay reference operator.
- [x] attention or alternate structural expert — causal temporal-attention expert.
- [x] optional frequency-domain expert.
- [x] router diagnostics/expert utilization logging — assignment counts, sparse mean weights, entropy, and active-parameter upper bound.

## Custom temporal operator

- [x] Clear mathematical specification in `docs/custom_temporal_operator.md`.
- [x] Correct PyTorch reference implementation first.
- [x] Forward numerical tests including a hand-calculated recurrence fixture.
- [x] Gradient tests using `torch.autograd.gradcheck`.
- [ ] Profiling proving it matters enough to optimize — CPU reference timing/state accounting exists, but target-GPU profiling has not established a material H200 bottleneck.
- [ ] Triton implementation only after reference validation — intentionally not implemented until profiling justifies it on compatible GPU hardware.
- [ ] Triton numerical equivalence across representative shapes/dtypes/strides — requires a real Triton implementation/runtime.
- [ ] Triton fallback to reference path — `auto` currently selects the reference and explicit unvalidated Triton fails closed; real Triton compile/runtime fallback remains unvalidated.
- [x] Throughput and memory benchmark — CPU/reference samples-per-second plus exact learned-state bytes; representative GPU/H200 throughput/peak memory remains external.

### Progress note — 2026-08-20

- The Market Mixer and heterogeneous MoE are implemented under `src/trading_bot/models/custom.py` using the common Phase 5 `TrainingBatch`/`ModelOutput`/`Trainer` boundary and common prediction heads.
- Market Mixer components have a stable one-component-off ablation suite; cross-sectional and market-context paths fail closed on mixed decision timestamps.
- The MoE performs sparse top-k dispatch over local-TCN, long-memory multi-decay, temporal-attention, and optional frequency experts and exposes detached router-utilization diagnostics.
- The custom multi-decay recurrence is mathematically frozen before optimization and is covered by hand-calculated forward, gradcheck, causal-prefix, and non-contiguous/strided-input tests.
- Both custom architecture families train, checkpoint at optimizer step 2, reconstruct/restore, continue to step 4, publish the common Parquet + Zstd predictions, and enter one canonical Phase 6 cost-aware evaluator/leaderboard/report rehearsal.
- Read-only Python 3.12 CI run `32435211863` / job `96634923295` passed lock freshness, Ruff, formatting across 111 files, strict mypy across 57 source files, compileall, and 292 tests with only the unrelated opt-in Phase 2 real-S3 provider gate skipped.
- Detailed scope and remaining Triton/GPU acceptance are recorded in `docs/progress/phase_09.md` and `docs/custom_temporal_operator.md`.

## Triton policy

Do not rewrite already-optimized generic GEMM/attention merely to use Triton. Prefer project-specific fused preprocessing/temporal operations where profiling shows a real bottleneck.

## Gate

Custom architecture results must be explainable through ablations. Custom kernels must demonstrate correctness independently of speed improvement.

**CPU/REFERENCE CUSTOM ARCHITECTURE GATE PASSED.** The Market Mixer ablation boundary, heterogeneous MoE, and custom temporal reference math are exercised by the common CPU training/evaluation pipeline. Phase 9 remains **IN PROGRESS** only for target-GPU bottleneck profiling and any justified Triton implementation/equivalence/fallback/H200 performance acceptance; no Triton or GPU performance result is claimed by this CPU gate.

---

# 12. Phase 10 — experiment configuration and search spaces

## Goal

Freeze the campaign manifest before rental day.

- [x] Define architecture-family registry — 19 explicit family entries with mandatory/optional and searchable/reference-only boundaries.
- [x] Define small/medium/large canonical configs — searchable neural/advanced/custom families have version-controlled scale presets; classical references use an explicit `reference` preset.
- [x] Define screening search spaces — bounded axes plus per-family search-axis opt-in are encoded in YAML.
- [x] Define objective variants:
  - [x] Huber/excess-return regression — MSE and Huber 15-minute variants are registered; Huber is the frozen architecture-screening objective.
  - [x] cross-sectional ranking — 15-minute pairwise-ranking objective is registered.
  - [x] multi-task return + rank + volatility + direction — fully specified in the manifest and deliberately `planned_not_selected` until the common loss/head path supports those exact semantics end-to-end.
  - [x] multi-horizon variants — 15/30-minute Huber candidate is fully specified and `planned_not_selected` until multi-horizon output/loss semantics are implemented.
  - [x] distributional variants where selected — a 15-minute quantile candidate is defined and `planned_not_selected`; no unsupported distributional launch is claimed.
- [x] Define learning-rate ranges — `1e-4`, `3e-4`, `1e-3`.
- [x] Define dropout/regularization ranges — dropout candidates `0`, `0.1`, `0.2` and weight decay `0`, `1e-4`, `1e-3`; families only opt into axes their current constructors support.
- [x] Define context-length choices — `32`, `64`, `128`.
- [x] Define batch/effective-batch constraints — effective batch 256 via `64×4`, `128×2`, or `256×1` microbatch/accumulation.
- [x] Define seed policy — screening seed 17; finalist seeds 17/29/43.
- [x] Define mandatory vs optional experiment pools — 17 mandatory and 2 optional/reference-only entries in v1.
- [x] Define screening/promotion/full training budgets — 9 calibration, 66 screening at 15%, 18 promotion at 50%, 18 objective-search at 50%, and 4 full-budget finalists across 3 seeds, totaling 123 planned fits before runtime adaptation.

### Progress note — 2026-08-20

- `configs/campaigns/h200_tournament_v1.yaml` is the frozen v1 architecture/objective/search/budget contract; the strict schema and manifest-only enumerator live under `src/trading_bot/campaign`.
- Canonical JSON plus SHA-256 give the validated YAML a formatting-independent identity suitable for later scheduler/audit lineage.
- Campaign loading/enumeration does not import PyTorch/model code, preserving the Phase 11 controller boundary.
- CI independently verifies every Phase 8 advanced and Phase 9 custom YAML small/medium/large preset against the corresponding model-spec builder and constructs/forwards every neural-baseline scale.
- Unsupported full-volatility multitask, multi-horizon, and quantile objectives are defined but cannot be referenced by launchable architectures until their common training semantics exist; the manifest fails closed instead of approximating them.
- The rung breadth/fractions follow the already-documented campaign design: 66 screening → 18 promotion → 4 finalists, with 123 planned fits inside the intended approximately 100–130 range.
- Read-only Python 3.12 CI run `32438445520` / job `96644199981` passed lock freshness, Ruff, formatting across 115 files, strict mypy across 59 source files, compileall, and 339 tests with only the unrelated opt-in Phase 2 real-S3 provider gate skipped.
- Detailed evidence and scope boundaries are recorded in `docs/progress/phase_10.md`.

## Gate

A version-controlled campaign YAML/manifest can enumerate the intended experiment space without editing Python.

**PASSED — VERSION-CONTROLLED CAMPAIGN SEARCH MANIFEST.** The v1 architecture registry, canonical size presets, objective candidates, bounded search axes, seed policy, mandatory/optional pools, and relative screening/promotion/finalist budgets load, validate, hash, and enumerate from YAML without Python edits. Phase 11 may consume this immutable search contract; target-GPU runtime calibration and deadline scheduling remain Phase 11/17 work rather than Phase 10 claims.

---

# 13. Phase 11 — H200 campaign scheduler

## Goal

Build a deadline-aware campaign controller that remains alive when trials fail.

## Persistent state

- [x] SQLite campaign DB — schema-versioned, WAL-backed, `synchronous=FULL` local authority.
- [x] Campaign metadata table — stores exact Phase 10 manifest hash/JSON, state, fixed deadline, and drain reserve.
- [x] Trials table — immutable canonical config JSON/SHA-256 plus parent/root lineage and state.
- [x] Metrics table.
- [x] Checkpoints table.
- [x] Runtime statistics table.
- [x] Events table.
- [x] Failures table — generic scheduler failure lineage; detailed deterministic classes remain Phase 12.
- [x] Promotion lineage.
- [x] State snapshot to durable storage — SQLite backup → SHA-256 → `StorageBackend` upload → checksum verification.

## Campaign state machine

- [x] BOOTSTRAP.
- [x] CALIBRATION.
- [x] SCREENING.
- [x] PROMOTION.
- [x] OBJECTIVE_SEARCH.
- [x] FINALISTS.
- [x] DRAIN.
- [x] COMPLETE.

## Trial state machine

- [x] PENDING.
- [x] STARTING.
- [x] RUNNING.
- [x] EVALUATING.
- [x] SYNCING.
- [x] COMPLETE.
- [x] PRUNED.
- [x] RETRYABLE_FAILURE.
- [x] TERMINAL_FAILURE.
- [x] INTERRUPTED.

## Process isolation

- [x] Scheduler never imports/runs CUDA training in-process — fresh-process CI asserts importing `trading_bot.scheduler` does not import `torch`.
- [x] Each trial is a separate subprocess/process group through `SubprocessTrialRunner`.
- [x] Capture stdout/stderr per trial.
- [x] TERM → grace → KILL behavior — Linux CI covers a SIGTERM-ignoring process group and SIGKILL escalation.
- [x] Fresh Python process after serious CUDA errors — every launch/retry is a new subprocess; CUDA-error classification that decides the recovery action remains Phase 12.

## Deadline adaptation

- [x] Fixed campaign deadline — restart rejects silent deadline changes.
- [x] Dynamic drain reserve — evaluator backlog + durable-sync estimate + safety margin, bounded below by the frozen initial reserve.
- [x] Runtime estimator from observed trials — family/scale/context observations normalize partial-rung runtimes to full-budget equivalents.
- [x] Conservative p90-like launch estimate — nearest-rank configured `0.90` quantile plus safety multiplier.
- [x] Never start work that cannot plausibly complete before drain.
- [x] Stop low-priority exploration as deadline approaches.
- [x] Finalist-only mode.
- [x] Drain mode.

## Successive halving

- [x] Screening rung — the Phase 10 manifest deterministically materializes all 66 frozen screening trials.
- [x] Promotion rung.
- [x] Full/finalist rung.
- [x] Grace budget before pruning — pruning helper enforces the configured fraction of the current rung budget.
- [x] Promotion based on frozen evaluation hierarchy — consumes Phase 6 `LeaderboardRow.eligible` and `rank` without creating a scheduler score.

## Runtime/resource scheduling

- [x] GPU utilization reporting — typed/persisted worker telemetry fields are implemented; CPU CI uses synthetic values and makes no real H200 measurement claim.
- [x] Peak VRAM reporting — typed/persisted worker telemetry fields are implemented; real H200 values remain Phase 17 observations.
- [x] Support exclusive GPU trials — frozen v1 policy owns one GPU slot by default.
- [x] Optional limited concurrent tiny trials only if calibration proves beneficial — support exists but cannot open until an explicitly enabled policy receives calibration throughput gain above its threshold; v1 remains exclusive.
- [x] CPU evaluator runs concurrently where safe — independent CPU evaluator slots do not consume the GPU-trial slot.

### Progress note — 2026-08-20

- `configs/campaigns/h200_scheduler_v1.yaml` freezes the operational scheduler policy separately from the Phase 10 experiment/search manifest: 90-minute initial drain reserve, p90-like runtime estimate, deadline tiers at 24/12/6/3/1.5 usable hours, 50% pruning grace, two same-config retries, and exclusive GPU ownership by default.
- `CampaignDB` persists campaign/trial/metric/checkpoint/runtime/event/failure/promotion/resource state and produces transactionally consistent checksum-verified durable SQLite snapshots.
- Restart validates the exact canonical Phase 10 manifest identity and fixed campaign deadline; immutable retry/promotion children preserve parent/root lineage instead of mutating scientific trial identity.
- The scheduler package remains model/CUDA-free; workers use fresh process groups with separate logs and TERM→grace→KILL handling.
- Runtime estimates use observed partial-budget durations; adaptive drain fails closed when critical unsynced bytes exist without a known storage-throughput estimate.
- Promotion reuses the canonical Phase 6 leaderboard eligibility/rank hierarchy. No opaque scheduler score was introduced.
- The compressed CPU simulation exercises calibration, all 66 screening registrations, one retry child, controller/SQLite restart, promotion/objective/finalist lineage, finalist-only late scheduling, adaptive drain, pending-work interruption, zero-critical-backlog completion, and durable snapshot restore verification.
- Implementation-only read-only Python 3.12 CI run `32440888978` / job `96651209298` tested head `435f075257104cfdce29be2155c0b0690b562f20` via synthetic merge `1f964bed6edcca988504ab5b89e059bb40cb3df7`; Ruff/format passed across 128 files, strict mypy passed across 70 source files, compileall passed, and pytest reported 353 passed / 1 skipped.
- The only skip remains the unrelated opt-in Phase 2 real-S3 provider gate. Detailed Phase 11 evidence is recorded in `docs/progress/phase_11.md`.
- No real H200 utilization, peak-VRAM, or throughput measurement is claimed by the CPU/simulation gate. Deterministic CUDA/storage/evaluator failure classification, recovery policies, circuit breaker, golden canary, and AI repair remain Phase 12.

## Gate

In simulation mode, the controller finishes a compressed campaign with correct lineage, promotions, retries, deadline adaptation, and drain behavior.

**PASSED — CPU/SIMULATION CAMPAIGN SCHEDULER GATE.** The model-free controller completes the compressed campaign with durable SQLite state, immutable retry/promotion lineage, restart continuity, frozen Phase 6 promotion ordering, conservative launch guards, finalist-only mode, adaptive drain, zero critical sync backlog, and checksum-verified final state. Phase 12 may proceed without claiming target-H200 performance from this CPU gate.

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
- [x] Campaign configs immutable once campaign starts — Phase 11 restart validates the exact frozen Phase 10 manifest hash/JSON and refuses silent deadline changes.
- [x] Trial configs immutable — Phase 11 stores canonical config JSON/SHA-256 and creates new child rows for retries/promotions instead of mutating trial identity.
- [ ] Behavior-changing bug fix creates a new child trial/version — scheduler enforcement not yet implemented.
- [ ] AI-generated patches are committed/audited before being treated as valid experiment code — Phase 12.

## Performance discipline

- [x] Profile before writing custom kernels — CPU/reference operator timing and state accounting are in place before any Triton implementation; target-GPU profiling still gates whether optimization is justified.
- [ ] Keep H200 training input local/hot whenever practical.
- [ ] CPU evaluation/storage operations must not unnecessarily idle the GPU.
- [x] Store promoted/final predictions to permit CPU-only reevaluation.
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
9. [x] Commit/freeze a dependency lock strategy for reproducible CPU verification.
10. [x] Common model/trainer/checkpoint/prediction interfaces.
11. [x] Canonical evaluator and metric unit tests.
12. [x] Simple baseline models and an end-to-end research smoke test.
13. [x] Advanced/core model families — CPU/reference core gate; external foundation/GPU acceptance remains.
14. [x] Custom Market Mixer/reference custom operators — CPU/reference correctness gate; Triton/GPU optimization acceptance remains.
15. [x] Version-controlled campaign registry/search-space manifest.
16. [x] Campaign scheduler/state DB/deadline/successive-halving simulation — deterministic fault classification/recovery remains Phase 12.
17. [ ] Discord/telemetry/sync services.
18. [ ] AI repair sandbox.
19. [ ] CPU/GPU Docker images and Compose orchestration.
20. [ ] Scheduler simulation and full fault-injection dress rehearsal.
21. [ ] Full production data build/staging.
22. [ ] H200 campaign.

External production/hardware/model-artifact blockers should be closed as credentials, frozen production data/methodology, selected pretrained checkpoints, and GPU/Triton infrastructure become available; they do not invalidate the completed CPU/reference training, evaluation, baseline-model, advanced-core, custom-reference, campaign-manifest, and scheduler-simulation gates.

---

# 27. Pre-H200 final readiness checklist

The H200 should not be rented for the real campaign until all of the following can be checked:

- [ ] Full dataset is built and immutable.
- [x] Final-holdout access primitives are protected; final production dates still need freezing.
- [x] Leakage regression suite exists; it must still pass the finalized production dataset.
- [ ] Packed loader feeds the GPU efficiently on rehearsal hardware.
- [ ] All core architecture families can train with common interfaces.
- [x] Custom architectures have reference correctness tests.
- [ ] Any Triton kernels match references.
- [x] Evaluation metrics pass hand-calculated tests.
- [ ] Cost/slippage model is frozen.
- [x] Campaign YAML/search spaces are frozen.
- [x] Scheduler simulation passes — compressed Phase 11 restart/retry/promotion/deadline/drain gate is green in CPU CI.
- [ ] Real shortened campaign passes.
- [ ] OOM recovery passes.
- [ ] Hang recovery passes.
- [ ] Checkpoint corruption recovery passes.
- [x] Controller restart recovery passes — compressed Phase 11 simulation closes/reopens SQLite and resumes from the exact frozen manifest/deadline.
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