# Documentation Index

The repository documentation is organized as a set of explicit project contracts rather than an informal notebook.

## Core documents

- [`../PLAN.md`](../PLAN.md) — concise overall plan and frozen intent.
- [`architecture.md`](architecture.md) — system diagrams and component boundaries.
- [`data_and_storage_plan.md`](data_and_storage_plan.md) — market data, point-in-time universe, preprocessing, storage, and transfer design.
- [`model_experiment_plan.md`](model_experiment_plan.md) — architecture families, custom models, objectives, and H200 campaign shape.
- [`evaluation_contract.md`](evaluation_contract.md) — prediction, economic, robustness, cost, and systems metrics.
- [`scheduler_and_recovery.md`](scheduler_and_recovery.md) — adaptive campaign controller, failure policy, checkpointing, AI repair, and drain mode.
- [`paper_and_live_trading.md`](paper_and_live_trading.md) — shadow/paper/live-canary progression and operational gates.
- [`operations_and_observability.md`](operations_and_observability.md) — Compose services, logs, Discord alerts, health monitoring, and campaign reports.
- [`reproducibility_and_security.md`](reproducibility_and_security.md) — dataset/model lineage, secrets, change control, and deterministic debug mode.
- [`implementation_roadmap.md`](implementation_roadmap.md) — recommended build order and acceptance tests.
- [`decisions/README.md`](decisions/README.md) — Architecture Decision Record (ADR) conventions for future changes.

## Document statuses

Use these labels when modifying plans:

- **FROZEN** — part of a running campaign/release contract; change only via a new version.
- **BASELINE** — current intended design; can still be reviewed before implementation.
- **PROVISIONAL** — depends on pricing, vendor access, benchmarks, or later empirical results.

## Current status summary

- Trading intent: **BASELINE / effectively frozen**.
- Evaluation methodology: **BASELINE**, to be encoded into tests before campaign execution.
- H200 campaign architecture: **BASELINE**.
- Exact data-vendor subscription tier: **PROVISIONAL** until purchase.
- CPU preprocessing provider: **PROVISIONAL** pending GMI L4 host quote/specs vs external CPU options.
- Exact production broker: **OPEN**.
- Exact live-capital risk limits: **OPEN until paper-trading design implementation**.
