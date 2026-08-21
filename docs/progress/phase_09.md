# Phase 9 Progress — Custom Architectures and Triton Boundary

Last updated: **2026-08-20**

Status: **IN PROGRESS — CPU/reference custom-architecture gate passed; Triton/GPU acceptance pending**

`IMPLEMENTATION_PLAN.md` remains the authoritative checklist. This record separates the CPU-verifiable custom architecture/reference-math work from later Triton and target-GPU optimization acceptance.

## Multi-Scale Market Mixer

Implemented in plain PyTorch with the common Phase 5 model boundary:

- shared temporal feature encoder;
- causal short-timescale depthwise/pointwise convolution branch;
- medium/long learnable multi-decay temporal branch;
- learned gated fusion when both branches are enabled;
- same-decision-timestamp cross-sectional Transformer interaction;
- same-timestamp market-context equivalent using the cross-sectional mean state;
- expected-return/rank, direction, volatility, and uncertainty heads.

The implementation does not invent sector identifiers that are absent from the common training batch. Market context is therefore used as the documented `market/sector context tokens or equivalent` reference path; any future sector-token design must be supplied by a frozen dataset/model contract rather than inferred inside Phase 9.

### Ablations

A stable one-component-off suite is exposed and CPU-tested:

- `full`;
- `no_short`;
- `no_long`;
- `no_gated_fusion`;
- `no_cross_sectional`;
- `no_market_context`.

The cross-sectional and market-context paths fail closed when a batch mixes decision timestamps. A temporal-only ablation can still operate on mixed-timestamp batches because it performs no cross-sectional interaction.

## Heterogeneous MoE

The CPU/reference MoE uses sparse top-k routing over genuinely different temporal operators:

- causal local TCN expert;
- long-memory multi-decay expert;
- causal temporal-attention expert;
- optional Fourier/frequency-domain expert.

The router uses the sample representation together with the same-timestamp market mean representation. It dispatches only samples selected for each expert rather than evaluating every expert densely and masking afterward.

Detached router diagnostics expose:

- expert names;
- per-expert assignment counts;
- mean sparse routing weights;
- mean routing entropy;
- configured top-k.

Profiling records total parameters/state bytes and a conservative per-sample active-parameter upper bound. The sparse MoE reference demonstrates an active-parameter bound below total learned parameters without making a GPU throughput claim.

## Custom temporal operator

`docs/custom_temporal_operator.md` freezes the mathematical definition of the learnable multi-decay recurrence before any Triton optimization is attempted.

For decay channel `k` and feature `d`:

```text
a[k,d] = sigmoid(decay_logits[k,d])
s[k,t,d] = a[k,d] * s[k,t-1,d] + (1-a[k,d]) * x[t,d]
q[k,d] = softmax(mix_logits[:,d])[k]
y[t,d] = sum_k q[k,d] * s[k,t,d]
```

State starts from zero, so every output is causal. The module wrapper adds learned input/output projections, a gate, residual connection, and layer normalization.

The current dispatch contract is intentionally conservative:

- `auto` -> PyTorch reference;
- `reference` -> PyTorch reference;
- explicit `triton` -> `TritonUnavailableError`.

There is no hidden or unvalidated Triton code path.

## Reference correctness and profiling

CPU tests cover:

- hand-calculated forward numerics;
- `torch.autograd.gradcheck` for the recurrence;
- causal prefix invariance;
- non-contiguous/strided input handling;
- representative custom-model small/medium/large shapes;
- CPU samples/second and exact learned-state bytes;
- fail-closed explicit Triton dispatch;
- stable Market Mixer ablations;
- sparse MoE router utilization diagnostics.

The CPU timing result is a reference systems signal only. It does **not** prove that the operator is a material H200 bottleneck. Target-GPU profiling is still required before a Triton implementation is justified under the repository's Triton policy.

## Common training/evaluation rehearsal

For both `market_mixer` and `heterogeneous_moe`, the Phase 9 CPU fixture:

1. performs finite forward/backward through the common prediction heads;
2. records parameter/state size and common CPU inference timing;
3. trains through the common Phase 5 `Trainer`;
4. checkpoints at optimizer step 2;
5. reconstructs/restores and continues to optimizer step 4;
6. writes the common Parquet + Zstd prediction artifact;
7. reads predictions through the independent Phase 6 evaluator;
8. applies one deterministic market-neutral rank portfolio rule;
9. produces a canonical cost-aware leaderboard and checksummed report.

No transaction-cost, portfolio, promotion, final-holdout, or live-risk rule was changed by this phase.

## CPU verification evidence

Read-only GitHub Actions run **32435211863** / job **96634923295** verified the Phase 9 CPU/reference implementation at PR head `aaaa8ad32b887a9d75148372523a21336140705d` on Python **3.12.3**.

```text
uv lock --check: pass
baseline-cpu locked sync: pass (73 locked packages)
Ruff: pass
formatting: 111 files already formatted
mypy: success, no issues in 57 source files
compileall: pass
pytest: 292 passed, 1 skipped in 16.67s
```

The single skip remains the existing opt-in Phase 2 real-S3 provider gate and is unrelated to Phase 9. The permanent CPU workflow uses read-only repository contents permission.

## Remaining Phase 9 acceptance

- target-GPU profiling demonstrating that the custom temporal operator is material enough to justify optimization;
- Triton implementation, if that profiling justifies one;
- Triton/reference numerical and gradient equivalence across representative shapes/dtypes/strides;
- real compile/runtime fallback validation against the PyTorch reference;
- representative GPU/H200 throughput and peak-memory benchmark.

Until those items are satisfied, Phase 9 remains **IN PROGRESS** even though the CPU/reference custom-architecture and correctness gate has passed.
