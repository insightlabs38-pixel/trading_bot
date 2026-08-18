# Core Project Plan

Status: **planning baseline**. Changes to frozen research contracts should be explicit and versioned.

## Objective

Build a technically ambitious medium-frequency U.S. equities trading bot as both a serious ML/systems project and a candidate trading system. The research objective is not to maximize historical profit at any cost; it is to identify robust, cost-aware, reproducible signal/execution systems that survive out-of-sample evaluation and paper trading.

## Frozen intent

| Area | Decision |
|---|---|
| Trading style | Medium-frequency intraday; not colocated HFT |
| Decision cadence | ~1 minute |
| Primary horizons | 15m and 30m |
| Auxiliary horizons | 5m and 60m; EOD/next-day may be auxiliary tasks |
| Universe | Point-in-time liquid U.S. common equities; ~750–1,500 target |
| Broad data | 1-minute historical data with corporate actions/reference data |
| Execution validation | BBO/L1/trade data for finalists; deeper LOB only for later execution research |
| GPU campaign | One H200-class GPU, ~48 hours |
| Campaign strategy | Successive-halving-style screening → promotion → finalist validation |
| Main system form | Specialist model ensemble / modular system, not autonomous debating agents |
| Multi-agent RL | Deferred to execution/microstructure research |
| Storage | GMI Cold Storage as active durable store; local NVMe as hot training scratch |
| Preprocessing | CPU-first; external CPU vs GMI-hosted option selected by real price/performance |
| Containers | Docker Compose; separate CPU and GPU images; no Docker Swarm |
| Observability | Structured logs/metrics + Discord alerts; lightweight generated status report |
| AI repair | Sandboxed fast-model repair tier; deterministic scheduler retains control |
| Validation | Walk-forward, immutable final holdout, transaction-cost aware |
| Deployment | Shadow → paper → tiny live canary → limited live → normal live |

## Research architecture families

Core families to prepare:

- linear / MLP baselines;
- LSTM / xLSTM and VSN-recurrent variants;
- TCN;
- PatchTST;
- iTransformer;
- Mamba-family temporal models;
- causal Transformer;
- temporal + cross-sectional attention;
- temporal + graph model;
- custom multi-scale market mixer;
- heterogeneous mixture-of-experts;
- selected pretrained time-series foundation-model references.

Custom work should emphasize **market-relevant inductive bias** rather than model size alone: multi-timescale temporal mixing, cross-sectional interactions, ranking/distributional heads, and architecture/kernel co-design where profiling justifies Triton work.

## Campaign principle

The campaign optimizes for:

> **best defensible result before the fixed deadline**, not execution of every configuration.

Target shape:

1. calibration/profiling;
2. ~60–70 cheap architecture screens;
3. ~16–20 promoted configurations;
4. objective/target ablations on the strongest families;
5. ~3–4 finalists with multiple seeds and walk-forward folds;
6. final drain, artifact verification, and report generation.

Actual counts are adaptive to measured H200 throughput.

## Hard research invariants

- Never randomly shuffle chronological financial data.
- The final holdout is not used for architecture, hyperparameter, feature, target, transaction-cost, or risk-rule selection.
- Feature availability must be causal at the decision timestamp.
- Survivorship bias must be addressed with point-in-time universe construction.
- Every promoted strategy is evaluated after modeled fees, spread, slippage, and impact assumptions.
- All attempted configurations are logged for multiple-testing/overfitting analysis.
- A bug fix creates a new version/child trial if behavior can change.
- AI repair cannot modify the evaluation contract, splits, cost assumptions, or protected campaign logic.

## Major stages

### Stage A — repository + contracts

Complete documentation, interfaces, container plan, and reproducibility rules.

### Stage B — data engineering

Acquire, validate, canonicalize, adjust, resample, build point-in-time universe, generate causal features/labels, freeze splits, and pack training data.

### Stage C — campaign implementation

Implement common model interface, trainer, evaluator, scheduler, checkpointing, recovery, storage sync, Discord notifications, and simulation/dress-rehearsal mode.

### Stage D — H200 campaign

Run adaptive architecture tournament and produce a fully auditable campaign report.

### Stage E — untouched final holdout

Freeze the complete winning system and evaluate once on the reserved holdout.

### Stage F — paper trading

Shadow mode followed by real broker paper execution and deliberate operational fault testing.

### Stage G — limited live test

Only after paper acceptance gates pass; begin with tiny capital and deterministic risk limits.

## Files that define the contract

- `docs/data_and_storage_plan.md`
- `docs/model_experiment_plan.md`
- `docs/evaluation_contract.md`
- `docs/scheduler_and_recovery.md`
- `docs/paper_and_live_trading.md`
- `docs/operations_and_observability.md`
- `docs/reproducibility_and_security.md`

If implementation and these files disagree, treat the mismatch as a design issue rather than silently changing the plan.
