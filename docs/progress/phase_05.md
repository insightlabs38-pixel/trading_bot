# Phase 5 Progress — Common Training Framework

Last updated: **2026-08-20**

Status: **IN PROGRESS — CPU framework gate passed; GPU-specific acceptance pending**

This file records validation detail for Phase 5. `IMPLEMENTATION_PLAN.md` remains the authoritative
checklist. The work in this phase deliberately separates CPU-verifiable framework correctness from
CUDA/FP8/H200 acceptance so GPU behavior is not inferred from CPU tests.

## Dependency and CI strategy

- [x] Added an isolated `training-cpu` uv dependency group for CPU development and CI.
- [x] CPU verification resolves PyTorch from the official CPU-only PyTorch wheel index.
- [x] The production `gpu` dependency group remains PyTorch-free; the future pinned NVIDIA NGC
  image still owns PyTorch, Triton, CUDA libraries, and Transformer Engine.
- [x] `uv.lock` is committed with the Python 3.12 CPU-training resolution.
- [x] The permanent free-tier GitHub Actions job uses `uv lock --check` and
  `uv sync --locked --group training-cpu`, so CI fails on lock drift instead of rewriting it.
- [x] No paid/larger runner or GPU runner is required for the Phase 5 CPU gate.

## Common model interface

- [x] `TradingModel` provides the common PyTorch model base.
- [x] `TrainingBatch` standardizes model-ready features, named targets, asset IDs, and exact int64
  timestamps while validating batch identity and dimensions.
- [x] `ModelOutput` standardizes optional expected-return, rank-score, direction-probability,
  volatility, uncertainty, and quantile heads.
- [x] Shared output validation rejects empty, shape-inconsistent, NaN, and Inf predictions.
- [x] Parameter-count reporting is architecture independent.
- [x] Eager inference timing reports elapsed time, samples/sec, and mean batch latency without
  requiring CUDA.

## Common trainer

- [x] BF16 is the common default precision and is exercised with CPU autocast in CI.
- [x] Explicit FP32/debug training is supported.
- [x] Gradient accumulation is shared across architectures.
- [x] Gradient clipping uses `error_if_nonfinite=True`.
- [x] Optimizer LR schedulers use a minimal stateful protocol and step only after optimizer steps.
- [x] Early stopping is an external callback rather than hidden inside model code.
- [x] Non-finite model outputs, losses, and gradients fail closed.
- [x] Step/time heartbeat callbacks report training cursor, throughput, and learning rate.
- [x] Deterministic-debug mode enables PyTorch deterministic algorithms for the training scope and
  restores the caller's prior global setting afterward.
- [x] Fast-campaign mode disables the deterministic-algorithm requirement without changing model
  semantics.
- [x] CUDA-memory telemetry has a typed heartbeat contract and explicitly reports `None` on CPU.
- [ ] **GPU ACCEPTANCE** — exercise and validate allocated/reserved/peak CUDA memory telemetry on a
  real GPU runner.
- [ ] **GPU ACCEPTANCE** — implement/validate the optional FP8 finalist path only on supported GPU
  hardware and software.

## Checkpointing and true continuation

- [x] Model state is persisted and restored strictly.
- [x] Optimizer state is persisted and restored.
- [x] LR scheduler state is persisted and validated when a scheduler is present.
- [x] Optimizer step, micro-step, samples-seen cursor, latest loss, and stop state are persisted.
- [x] PyTorch CPU RNG, available CUDA RNG states, Python RNG, and NumPy RNG are persisted.
- [x] Generic precision/scaler state has a durable checkpoint field for precision-specific callers.
- [x] Model-config hash, training-config hash, dataset ID, and split ID are mandatory resume
  identities.
- [x] Checkpoints publish through temporary directory -> fsync -> reload/verify -> checksum ->
  atomic rename -> parent-directory fsync.
- [x] `latest.json` and optional `best.json` bookkeeping publish atomically.
- [x] Identity mismatch is rejected before caller-owned model/optimizer state is mutated.
- [x] State and manifest corruption are detected by SHA-256 before restore.
- [x] A dropout MLP checkpoint/resume regression restores optimizer, scheduler, and RNG state and
  reproduces the next optimizer-step parameters exactly with `torch.equal` on CPU.
- [ ] **GPU ACCEPTANCE** — validate CUDA precision/scaler state with the finalized GPU precision
  stack where such state is relevant.

## Prediction artifacts

- [x] Validation/test predictions can be materialized from the common model/batch contracts.
- [x] Prediction records preserve exact timestamp, asset ID, target, prediction heads, and lineage.
- [x] Prediction artifacts publish as immutable Parquet 2.6 + Zstd with a canonical checksummed
  manifest and SHA-256-protected data file.
- [x] Dataset ID, split ID, model-config hash, checkpoint ID, and target name are embedded in both
  durable manifest and Parquet metadata.
- [x] The `PredictionArtifact` reader does not import `Trainer`; Phase 6 can evaluate saved
  predictions without retraining.
- [x] Prediction corruption/tampering fails closed before evaluator use.

## Three-architecture CPU gate

The same training/checkpoint/prediction path is exercised with three intentionally different small
models:

1. affine `LinearReturnModel`;
2. nonlinear GELU/dropout `MLPReturnModel`;
3. residual gated mixer `ResidualGatedReturnModel`.

For every model, the gate performs train -> checkpoint -> fresh-process-style model/optimizer
construction -> resume -> continued train -> saved predictions -> trainer-independent artifact read
and MSE evaluation smoke. The MSE consumer is only a Phase 5 artifact/interface acceptance smoke;
the canonical predictive/economic/backtest evaluator remains Phase 6.

## Authoritative CPU verification

The first complete supported-environment Phase 5 CPU run used Ubuntu 24.04, Python **3.12.3**, uv
**0.10.12**, and PyTorch **2.13.0+cpu** from the official CPU-only index.

```text
Ruff: all checks passed
Formatting: all files formatted
mypy: success, no issues in 41 source files
pytest: 252 passed, 1 skipped in 11.54s
```

The single skipped test is still the opt-in real S3 provider gate from Phase 2. The Phase 5 tests,
including BF16 CPU autocast, exact checkpoint continuation, the three-model gate, and Parquet
prediction artifacts, all executed in that run.

A final CI run is required after this progress/tracker reconciliation and the telemetry-contract
addition; it must remain green before this branch is considered ready for merge.

## Remaining Phase 5 blockers

- [ ] Real CUDA validation of GPU-memory telemetry.
- [ ] Optional FP8 finalist implementation/validation on supported hardware.
- [ ] CUDA-specific precision/scaler resume acceptance with the final NGC stack where relevant.
- [ ] Representative GPU/H200 inference timing and throughput validation; CPU timing is only the
  architecture-independent interface gate.

## Phase 5 status

**CPU FRAMEWORK GATE PASSED.** All Phase 5 items that can be meaningfully implemented and verified
on CPU are present and covered by the common Python 3.12 CI path. Phase 5 remains **IN PROGRESS**
only for the explicitly GPU-dependent acceptance items above. No GPU, FP8, CUDA-memory, or H200
performance claim is made by the CPU gate.
