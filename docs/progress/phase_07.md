# Phase 7 Progress — Baseline Model Families

Last updated: **2026-08-20**

Status: **COMPLETE — CPU/reference baseline gate passed**

`IMPLEMENTATION_PLAN.md` remains the authoritative checklist. This record documents the CPU-verifiable Phase 7 implementation and the exact distinction between classical-estimator and neural-network acceptance semantics.

## Baseline families

### Classical CPU baselines

- [x] Ridge regression.
- [x] Elastic Net regression.
- [x] Logistic direction classification.
- [x] LightGBM regression.
- [x] XGBoost regression through the official CPU-only package distribution.

### Neural baselines

- [x] MLP.
- [x] GRU.
- [x] LSTM.
- [x] Causal TCN.
- [x] Simple causal Transformer.

## Common research interfaces

- [x] Every family consumes the same identity-preserving `TrainingBatch`-based train/validation split contract.
- [x] Every family consumes the validated shared `ObjectiveConfig` schema rather than a family-specific experiment configuration.
- [x] All families emit durable Phase 5 prediction records/artifacts consumable by the canonical Phase 6 evaluator.
- [x] Dataset ID, split ID, model configuration identity, exact asset ID, and exact timestamp lineage are preserved through the rehearsal gate.

Classical estimators flatten non-batch feature axes into deterministic tabular arrays at their adapter boundary. Neural families preserve the temporal `[batch, time, features]` representation.

## Complexity and inference timing

- [x] Neural families use the common trainable-parameter count and inference benchmark contracts from Phase 5.
- [x] Linear classical families report coefficient/intercept learned-scalar counts.
- [x] Tree families report learned tree-node counts plus exact serialized fitted-state byte size.
- [x] Classical estimators have a common CPU inference timing contract.
- [x] The Phase 7 gate asserts positive throughput for every family without freezing a hardware-specific performance threshold.

These measurements are reference CPU measurements only. H200/GPU throughput acceptance belongs to later hardware phases and is not inferred from this gate.

## Objective behavior

- [x] Excess-return MSE/Huber adapter for applicable neural and regression baselines.
- [x] Direction/BCE adapter and logistic-direction baseline.
- [x] Same-timestamp pairwise ranking loss for neural baselines.
- [x] Composite multitask neural loss over return, rank, and direction heads.
- [x] Unsupported objective/family combinations fail closed rather than silently changing the objective.

## Checkpoint and continuation semantics

### Classical estimators

- [x] Fitted estimator state is stored in an immutable, checksummed checkpoint directory.
- [x] Manifest identity includes family, validated objective configuration, model config hash, dataset ID, and split ID.
- [x] Manifest/data hashes and sizes are verified before fitted state is deserialized.
- [x] Reconstructed estimators restore identical validation scores within numerical tolerance.
- [x] Corrupted fitted-state bytes fail closed.

Classical algorithms do not have optimizer-step backpropagation state. Their Phase 7 continuation contract is durable fitted-state restore/reuse, not a fabricated neural optimizer cursor.

### Neural estimators

- [x] Every family performs explicit forward/loss/backward with finite gradients through all shared output heads.
- [x] Every family trains through the common Phase 5 `Trainer`.
- [x] Every family saves model/optimizer/RNG/cursor identity through the Phase 5 checkpoint manager.
- [x] Every family is reconstructed, restored, and continues from optimizer step 2 through optimizer step 4 before prediction publication.

## Rehearsal baseline leaderboard gate

A deterministic rehearsal split contains 10 cross-sectional batches, 8 assets per timestamp, 4 temporal observations per sample, and 3 input features. The first 6 batches train/fit the baselines and the last 4 form the validation view.

For all ten concrete family entries:

1. train/fit from the same split identity;
2. measure complexity and inference throughput;
3. checkpoint and reconstruct/restore;
4. write the common Parquet + Zstd prediction artifact;
5. independently read the artifact through the Phase 6 evaluator;
6. construct the same deterministic market-neutral rank portfolio rule;
7. run the canonical cost-aware evaluator;
8. build one hierarchical baseline leaderboard; and
9. write and checksum-verify the canonical evaluation report.

The gate includes Ridge, Elastic Net, logistic direction, LightGBM, XGBoost, MLP, GRU, LSTM, TCN, and causal Transformer.

## Dependency isolation

The `baseline-cpu` dependency group layers classical estimators on the existing `training-cpu` group while leaving the production `gpu` group unchanged.

The supported Python 3.12 CI environment resolves:

- PyTorch `2.13.0+cpu` from the official CPU-only PyTorch index;
- LightGBM `4.7.0`;
- scikit-learn `1.9.0`;
- `xgboost-cpu` `3.4.1`.

The official CPU-only XGBoost distribution is used so standard CPU CI does not install the GPU-capable wheel or its NCCL dependency.

## CPU verification evidence

A permanent read-only GitHub Actions run on Python **3.12.3** verified the complete implementation before tracker reconciliation:

```text
uv lock --check: pass
baseline-cpu locked sync: pass
Ruff: pass
formatting: 104 files formatted
mypy: success, no issues in 55 source files
compileall: pass
pytest: 268 passed, 1 skipped
```

The single skip remains the existing opt-in Phase 2 real-S3 provider gate and is unrelated to Phase 7.

## Phase 7 status

**COMPLETE — CPU/REFERENCE BASELINE GATE PASSED.** All Phase 7 baseline families and acceptance conditions that are meaningful for their algorithm class are implemented and exercised in standard Python 3.12 CPU CI. Classical models use fit/predict/fitted-state restoration semantics; neural models additionally use forward/backward and true optimizer-step checkpoint continuation. The canonical rehearsal leaderboard is produced through the Phase 6 evaluator before Phase 8 advanced architectures proceed.
