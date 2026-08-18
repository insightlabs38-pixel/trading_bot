# System Architecture

Status: **BASELINE**

## Research-to-live system

```mermaid
flowchart TD
    A[Market data vendors] --> B[CPU preprocessing]
    B --> C[GMI Cold Storage]
    C --> D[H200 local NVMe staging]
    D --> E[Model training campaign]
    E --> F[Promoted models]
    F --> G[Walk-forward evaluation]
    G --> H[Finalists]
    H --> I[Untouched final holdout]
    I --> J[Shadow mode]
    J --> K[Broker paper trading]
    K --> L[Tiny live canary]
    L --> M[Limited live deployment]

    E --> N[Campaign DB / artifacts]
    N --> C
    E --> O[Discord notifications]
```

## Medium-frequency live decision path

```mermaid
flowchart LR
    A[Live market data] --> B[1-minute causal feature snapshot]
    B --> C[Alpha model / ensemble]
    C --> D[5m / 15m / 30m / 60m forecasts]
    D --> E[Uncertainty + volatility estimates]
    E --> F[Portfolio construction]
    F --> G[Deterministic risk engine]
    G --> H{Expected edge exceeds costs + threshold?}
    H -- No --> I[Hold]
    H -- Yes --> J[Execution engine]
    J --> K[BBO / liquidity state]
    K --> L[Broker]
    L --> M[Order/fill reconciliation]
    M --> G
```

The model never has authority to bypass the deterministic risk engine. Broker acknowledgements and reconciled positions are authoritative in live operation.

## H200 campaign topology

```mermaid
flowchart TD
    A[Campaign controller\nCPU / no CUDA ownership] --> B[Trial queue]
    B --> C[Trainer subprocess\nowns H200]
    C --> D[Local checkpoint / predictions]
    D --> E[Evaluator sidecar\nCPU]
    D --> F[Sync sidecar\nCPU]
    F --> G[GMI Cold Storage]
    E --> A
    C --> A
    H[System/DCGM telemetry] --> A
    A --> I[Discord notifier]

    C -- unknown/repeated failure --> J[Quarantine]
    J --> K[AI repair worker\nisolated sandbox]
    K --> L[Static + unit + GPU smoke + regression gates]
    L -- pass --> B
    L -- fail --> J
```

## Storage tiers

```mermaid
flowchart LR
    A[Vendor/raw downloads] --> B[CPU scratch NVMe]
    B --> C[Canonical Parquet + manifests]
    C --> D[GMI Cold Storage\ndurable working truth]
    D --> E[H200 local NVMe\nhot training data]
    E --> F[Training]
    F --> G[Checkpoints / predictions / reports]
    G --> D
    D --> H[Optional external backup\ncritical artifacts only]
```

## Component boundaries

### Data layer

Responsible for vendor ingestion, security master, corporate actions, point-in-time universe, causal features, labels, split definitions, validation, and dataset manifests.

### Research/model layer

Responsible for common model interfaces, architecture implementations, custom kernels, objectives, checkpointing, and prediction generation. It does not decide what constitutes a valid backtest.

### Evaluation layer

Owns the canonical return/cost accounting and all frozen model-selection metrics. Models provide predictions; the evaluator determines economic results.

### Campaign layer

Owns trial scheduling, deadline management, retries, promotion, pruning, health checks, AI-repair escalation, and durable campaign state. It does not change scientific contracts mid-campaign.

### Execution/risk layer

Owns target-to-order conversion, cost/latency/liquidity logic, broker reconciliation, position limits, stale-data rejection, loss limits, and kill switches. This layer remains independent from the learned alpha model.

## Non-goal: true HFT

The project is not designed around co-location, direct exchange feed parsing, microsecond order-to-ack latency, FPGA/kernel-bypass networking, or thousands of orders per second. Deeper order-book and multi-agent work may later be used to improve **execution** while preserving the medium-frequency alpha horizon.
