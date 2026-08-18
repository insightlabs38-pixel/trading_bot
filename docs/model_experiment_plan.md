# Model Experiment Plan

Status: **BASELINE**

## Goal

Use the H200 rental as a controlled architecture tournament rather than a single giant training run. Most experiments should be small-to-medium models with enough breadth, seeds, and walk-forward validation to distinguish architecture effects from luck.

## Core architecture families

### Classical / lightweight baselines

- Ridge / Elastic Net (CPU)
- logistic regression (CPU)
- LightGBM / XGBoost (CPU)
- small MLP
- GRU/LSTM
- xLSTM / VSN + recurrent variants
- TCN

### Modern time-series sequence models

- PatchTST
- iTransformer
- causal Transformer
- Mamba-family temporal model
- selected pretrained time-series foundation models as references/fine-tuning experiments

### Cross-sectional / relational models

- temporal encoder + cross-sectional attention
- temporal encoder + learned/static/dynamic graph layer
- market/sector tokens and relational embeddings

### Custom architectures

#### Multi-Scale Market Mixer

Primary original architecture direction:

1. feature projection;
2. short temporal branch (e.g. TCN);
3. medium/long temporal branches (e.g. state-space models);
4. learned gated fusion;
5. cross-sectional attention and/or graph interaction;
6. market/sector context tokens;
7. shared representation feeding return, ranking, volatility, uncertainty, and direction heads.

Target sizes: roughly 20–100M parameters across variants.

#### Heterogeneous MoE

Router informed by market state/regime, with genuinely different expert operators rather than identical MLP experts, e.g.:

- TCN/local-pattern expert;
- state-space/long-memory expert;
- attention expert;
- Fourier/frequency-domain expert.

Keep active parameter count substantially below total parameter count.

#### Multi-decay temporal operator

Experiment with a finance-oriented learnable multi-timescale decay/mixer. First implement a clear PyTorch reference; only implement Triton after profiling/validation justifies it.

## Suggested size ranges

| Family | Approximate range |
|---|---:|
| MLP | 0.1–2M |
| GRU/LSTM/xLSTM | 1–20M |
| TCN | 1–15M |
| PatchTST | 5–50M |
| iTransformer | 5–50M |
| Mamba-family | 5–50M |
| causal Transformer | 10–100M |
| cross-sectional model | 15–120M |
| custom market mixer | 20–100M |
| heterogeneous MoE | 30–200M total, lower active |

One larger scaling sanity check may be useful, but the project should not assume that a larger model is better.

## Model interface

All models should eventually expose a common output contract conceptually equivalent to:

```text
return_prediction
rank_score
volatility_prediction
uncertainty/distribution
optional_direction_probability
```

Not every architecture must train every head in every experiment, but the evaluator should not require architecture-specific logic.

## Objectives to compare

Architecture search and objective search are separate dimensions.

Candidate objectives:

1. Huber/excess-return regression;
2. cross-sectional ranking;
3. direction/classification;
4. return + ranking multitask;
5. return + ranking + volatility + direction multitask;
6. distributional/quantile prediction;
7. multi-horizon supervision;
8. cost-aware opportunity targets where scientifically justified.

Do not compare architectures while simultaneously changing unrelated objectives/features unless the experiment is explicitly a joint-system comparison.

## Campaign structure

### Phase 0 — calibration/profiling

Run representative small/medium/large models to measure:

- steps/s and samples/s;
- peak VRAM;
- compilation time;
- dataloader wait;
- CPU/RAM/network utilization;
- BF16 baseline behavior;
- `torch.compile` variants where relevant.

Use these measurements to estimate trial durations and adjust the remaining campaign.

### Phase 1 — architecture screening

Target approximately 60–70 configurations at a small fraction of full training budget. Use fixed data/split/objective conditions for clean architecture comparison.

### Phase 2 — promotion

Promote approximately 16–20 configurations using predefined validity gates and ordered evaluation metrics. Increase budget and evaluate additional walk-forward periods.

### Phase 3 — objective/target tournament

Take the strongest architecture families and compare frozen target/loss designs.

### Phase 4 — finalists

Select roughly 3–4 systems for full-budget training, multiple seeds, multiple walk-forward folds, stronger cost/latency sensitivity analysis, and detailed prediction saving.

### Phase 5 — campaign drain/report

Stop launching expensive new trials, finish evaluations, verify storage sync, produce final report, and preserve all campaign metadata.

## Expected fit count

Rough target: ~100–130 total fits including calibration, short screening fits, promoted fits, objective ablations, and repeated finalist seeds/folds. Most are partial-budget fits; actual count is adaptive to measured runtime.

## H200 utilization

- Prefer one well-shaped training process when it saturates the H200.
- Permit limited concurrent small trials only if profiling demonstrates clear throughput gains.
- CPU evaluation/storage work runs concurrently with GPU training.
- BF16 is the default research precision.
- FP8 is a finalist/system-throughput experiment, not an uncontrolled variable in the core architecture screen.

## Triton policy

Do not rewrite optimized primitives just for novelty. Prioritize custom kernels for project-specific bottlenecks found by profiling, particularly fused preprocessing/mixing or a custom temporal operator.

For every custom kernel:

- maintain a PyTorch reference;
- compare forward numerics across shapes/dtypes/strides;
- validate gradients if differentiable;
- test edge cases;
- benchmark eager vs compiled/reference vs Triton;
- record throughput and memory separately from trading metrics.

## Two leaderboards

### Trading/research leaderboard

Rank IC/stability, net Sharpe, drawdown/Calmar/Sortino, cost robustness, fold/seed robustness, DSR/PBO diagnostics, etc.

### Systems leaderboard

Training throughput, time to convergence, VRAM, compilation overhead, parameter count, inference latency, and final deployment footprint.

A faster kernel should improve the systems leaderboard; it should not be credited with better trading performance unless it changes the architecture itself.
