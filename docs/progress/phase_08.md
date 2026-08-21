# Phase 8 Progress — Advanced Model Families

Last updated: **2026-08-20**

Status: **IN PROGRESS — CPU/reference core gate passed; selected foundation checkpoint and GPU/H200 acceptance pending**

`IMPLEMENTATION_PLAN.md` remains the authoritative checklist. This record distinguishes the CPU-reference architecture gate from external pretrained-checkpoint selection and GPU/H200 performance acceptance.

## CPU-reference core families

Implemented as dependency-light PyTorch models that consume the common Phase 5 `TrainingBatch`, emit the common `ModelOutput`, train through the common `Trainer`, restore through the common checkpoint manager, publish the common prediction artifact, and enter the canonical Phase 6 evaluator:

- PatchTST-style channel-independent patch Transformer;
- iTransformer-style variable-token Transformer;
- pure-PyTorch Mamba-family selective state-space reference;
- variable-selection-network + LSTM recurrent reference;
- temporal + same-timestamp cross-sectional Transformer;
- temporal + same-timestamp graph-attention reference.

The Mamba-family implementation is intentionally a readable selective state-space correctness/screening reference. It does not claim fused Mamba-2 kernel equivalence or H200 throughput.

## Configuration scaling

Each core family has deterministic `small`, `medium`, and `large` reference specifications. CPU CI instantiates representative shapes for all scales and checks that learned-state size increases monotonically. The screening-budget rehearsal trains the small configuration for each family; larger paid-campaign sizing remains subject to later GPU/H200 profiling.

## Common prediction heads

Every trainable Phase 8 reference emits:

- expected return;
- rank score;
- direction probability;
- volatility;
- uncertainty.

The shared Phase 7 objective adapter remains the training-objective boundary for return, ranking, direction, and composite multitask CPU tests.

## CPU profiling

Phase 8 adds deterministic model-state accounting:

- total parameter count;
- trainable parameter count;
- exact parameter tensor bytes;
- exact registered-buffer bytes;
- total model-state bytes.

The existing common inference benchmark supplies CPU samples/second and mean batch latency. This is a reference systems signal only; GPU peak-memory and H200 throughput acceptance remain external.

## Cross-sectional/graph safety boundary

The temporal+cross-sectional and temporal+graph families fail closed unless every sample in a batch shares one decision timestamp. Their same-timestamp interactions therefore cannot silently mix different decision times. The graph reference derives its learned similarity graph only from temporal encodings of already-observed context at that decision timestamp.

## Foundation-model adapter boundary

A typed `FoundationBackbone` plus immutable `FoundationModelIdentity` and `FrozenFoundationAdapter` are implemented for offline use with an already-loaded, checksum-identified pretrained time-series backbone. The adapter:

- performs no network access;
- freezes the supplied backbone;
- trains only the projection/common prediction heads;
- validates embedding shapes and checkpoint identity metadata;
- remains compatible with common trainer/checkpoint/prediction contracts.

A real selected pretrained checkpoint has **not** been chosen, downloaded, licensed, or evaluated by the CPU reference gate. That acceptance item remains open until a concrete model artifact is selected.

## Rehearsal gate

The Phase 8 CPU test fixture uses one deterministic cross-sectional split. For each of the six core families it:

1. verifies finite forward/backward gradients through all common heads;
2. records learned-state bytes and CPU inference throughput;
3. trains through the common Phase 5 trainer;
4. checkpoints at optimizer step 2;
5. reconstructs/restores and continues through optimizer step 4;
6. writes the common Parquet + Zstd prediction artifact;
7. reads it independently through the Phase 6 evaluator;
8. applies one deterministic market-neutral rank portfolio rule;
9. produces one canonical cost-aware leaderboard and checksummed report.

## CPU verification evidence

Permanent read-only GitHub Actions run **32433593949** / job **96630186230** verified the reconciled implementation/tracker tree at exact PR head `a56de4508cd8b5927179df45d3ae43b847c613e7` on Python **3.12.3**. The workflow tested synthetic merge ref `b8bb4e8a78e3730f3e32fdc117cf722df3fbc5ad` into base `4faaaf4f7da8a891fbc45229dca165ebe96aa16e`.

```text
uv lock --check: pass
baseline-cpu locked sync: pass (73 locked packages)
Ruff: pass
formatting: 108 files already formatted
mypy: success, no issues in 56 source files
compileall: pass
pytest: 279 passed, 1 skipped in 14.85s
```

The single skip remains the existing opt-in Phase 2 real-S3 provider gate and is unrelated to Phase 8. The permanent workflow uses read-only repository contents permission.

## Remaining Phase 8 acceptance

- selected real pretrained time-series foundation-model checkpoint/reference evaluation;
- representative GPU/H200 memory and throughput validation for scale decisions.

No external-provider, final-holdout, promotion-rule, transaction-cost, or live-trading assumption is changed by Phase 8.
