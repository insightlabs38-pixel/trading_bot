# Implementation Roadmap

Status: **BASELINE**

The goal is to finish almost all code, tests, and orchestration before the paid H200 window. During the rental, normal work should be limited to configuration/hyperparameter decisions and genuinely unexpected bug fixes.

## Milestone 1 — repository contracts

- finalize planning docs;
- freeze directory/module boundaries;
- define config schemas;
- define dataset/model/evaluator interfaces;
- define protected invariants in tests/agent instructions.

Acceptance: documentation agrees on trading intent, data lineage, metrics, scheduler behavior, and paper/live promotion.

## Milestone 2 — CPU data pipeline

Implement:

- vendor download adapters;
- local/S3 storage abstraction;
- raw validation;
- security master/corporate actions;
- point-in-time universe builder;
- resampling;
- causal feature pipeline;
- labels;
- immutable splits;
- dataset manifests/checksums;
- training packer;
- loader benchmark.

Acceptance:

- every stage restartable/idempotent;
- a small dataset can be rebuilt from raw sources deterministically enough for research;
- leakage/data-contract tests pass;
- final-holdout split is protected;
- output uploads and verifies successfully.

## Milestone 3 — common research framework

Implement:

- common model/output interface;
- trainer;
- checkpoint save/resume;
- prediction writer;
- canonical evaluator/backtester;
- metric calculations from `evaluation_contract.md`;
- experiment/run manifests;
- baseline architectures.

Acceptance: several architecture families train/evaluate through exactly the same interfaces.

## Milestone 4 — advanced/custom architectures

Implement and test:

- PatchTST/iTransformer/Mamba/Transformer families;
- xLSTM/VSN variants if selected;
- cross-sectional/graph variants;
- Multi-Scale Market Mixer;
- heterogeneous MoE experiments;
- PyTorch reference for any custom temporal operator;
- Triton only after reference correctness + profiling.

Acceptance: unit/smoke tests cover representative shapes/dtypes, and custom kernels match references within defined tolerance.

## Milestone 5 — scheduler + services

Implement:

- campaign controller/state DB;
- trial subprocess management;
- deadline/runtime estimator;
- successive halving/promotion;
- deterministic failure handling;
- atomic checkpoints;
- evaluator/sync sidecars;
- Discord notifier;
- telemetry ingestion;
- circuit breaker;
- golden canary;
- AI repair sandbox/escalation;
- campaign simulation mode;
- adaptive DRAIN.

Acceptance: a simulated compressed campaign survives injected failures and completes unattended.

## Milestone 6 — Docker/Compose

Implement:

- pinned CPU image;
- pinned validated NGC-based GPU image;
- common Compose file plus CPU/GPU overrides/profiles;
- bind-mounted scratch/data paths;
- runtime secret injection;
- health checks/restart policies;
- one-command bootstrap/smoke/campaign entry points.

Acceptance: clean machine can clone repo, inject secrets/config, stage test data, and run the rehearsal without manual dependency installation.

## Milestone 7 — full dress rehearsal

Run the real stack with tiny data/short budgets. Inject:

- OOM;
- NaN;
- hang;
- CUDA/Triton failure;
- checkpoint corruption;
- evaluator failure;
- controller restart;
- storage outage/slowdown;
- disk pressure;
- Discord failure;
- AI repair pass/fail;
- early deadline/drain.

Acceptance: valid final artifacts/leaderboard/state are durable without manual rescue.

## Milestone 8 — data production + storage staging

Use the chosen CPU path to build the full dataset, upload to GMI Cold Storage, verify manifests, and benchmark actual transfer/loader behavior.

Acceptance: H200 campaign dataset is fully ready before GPU rental begins.

## Milestone 9 — paid H200 campaign

- bootstrap + hardware/data checks;
- calibration;
- adaptive architecture tournament;
- finalists;
- verified drain/report.

Do not develop new feature families/targets/evaluation rules during this window.

## Milestone 10 — final holdout

Freeze winner/system and evaluate once on protected holdout.

Acceptance: report generated regardless of outcome. No tuning follows from holdout inspection.

## Milestone 11 — production/paper stack

Implement live data ingestion, same feature/model artifact interfaces, portfolio/risk layer, broker adapter, reconciliation, shadow mode, three-ledger paper accounting, and operational fault injection.

Acceptance: shadow consistency + 40–60 trading-day paper acceptance gates.

## Milestone 12 — tiny live canary

Deploy with deliberately small capital and strict deterministic risk limits. Measure real implementation shortfall and operational behavior before considering any scale-up.
