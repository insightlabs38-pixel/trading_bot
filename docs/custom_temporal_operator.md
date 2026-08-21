# Multi-Decay Temporal Operator Reference

Status: **CPU/PYTORCH REFERENCE IMPLEMENTED — TRITON NOT YET JUSTIFIED OR VALIDATED**

This document freezes the Phase 9 reference math for the project-specific temporal operator before any GPU/Triton optimization is attempted. A future optimized kernel must reproduce this definition; it may not silently change the model semantics.

## Input and learned parameters

For a batch of causal temporal representations

```text
x[b, t, d]
```

with batch index `b`, time index `t`, and feature index `d`, the operator maintains `K` learned decay channels.

For each decay channel `k` and feature `d`:

```text
a[k, d] = sigmoid(decay_logits[k, d])
q[k, d] = softmax(mix_logits[:, d])[k]
```

Thus every decay factor is in `(0, 1)` and every feature's decay-channel mixing weights sum to one.

## Causal recurrence

State starts at zero before the first observed time step:

```text
s[k, -1, d] = 0
```

and evolves as

```text
s[k, t, d] = a[k, d] * s[k, t-1, d]
             + (1 - a[k, d]) * x[t, d]
```

The mixed output is

```text
y[t, d] = sum_k q[k, d] * s[k, t, d]
```

The implementation therefore uses only the current and earlier inputs for output `t`. Future samples cannot affect an earlier output.

## Module wrapper

`MultiDecayTemporalOperator` applies a learned input projection, the recurrence above, a gated learned output projection, a residual connection, and layer normalization. The recurrence itself is exposed independently as `multi_decay_reference` so optimized kernels can be compared against the exact mathematical core without architecture-specific heads.

The default `auto` dispatch currently resolves to the PyTorch reference implementation. Explicit `backend="triton"` fails closed with `TritonUnavailableError`; there is no hidden or unvalidated Triton path.

## Required correctness evidence before Triton

The CPU/reference gate must cover:

- hand-calculated forward numerics;
- differentiability and finite gradients;
- causal prefix invariance;
- representative shapes;
- non-contiguous/strided tensor inputs;
- reference timing and learned-state byte reporting;
- integration inside the custom Market Mixer and heterogeneous MoE families.

Only after representative GPU profiling shows this operator is material enough to optimize should a Triton implementation be added.

## Future Triton acceptance

A future Triton backend must retain this PyTorch reference and add, at minimum:

- forward numerical equivalence across representative shapes, dtypes, and strides;
- gradient equivalence for all differentiable learned inputs/parameters;
- edge-case coverage;
- automatic fallback to the reference implementation on compile/runtime incompatibility where safe;
- eager/reference versus Triton throughput measurements;
- peak-memory measurements on representative target hardware.

A faster kernel affects the systems leaderboard only. It does not receive trading-performance credit unless the architecture or learned function itself changes under an explicitly versioned experiment.
