# Operations and Observability Plan

Status: **BASELINE**

## Container strategy

Use Docker Compose, not Docker Swarm.

Maintain two primary immutable runtime images built from the same repository/lockfile:

- `trading-cpu:<git-sha>` — preprocessing, evaluation, storage sync, reports;
- `trading-gpu:<git-sha>` — NVIDIA/NGC PyTorch-based training runtime with Triton/Transformer Engine support as validated.

Multiple services may use the same CPU image with different commands. Only the trainer/campaign worker receives GPU access.

## Planned Compose services

### CPU preprocessing host

- `preprocess` — resumable data DAG;
- `sync` — bulk S3-compatible transfer/checksum verification.

### H200 host

- `controller` — campaign state/deadline/queue logic;
- `trainer` — GPU-owning trial subprocess launcher or training service;
- `evaluator` — CPU backtests/statistics;
- `sync` — asynchronous artifact upload/verification;
- `notifier` — Discord webhook delivery/retry;
- optional lightweight status-report generator;
- optional profiling tools activated by profile/override, not always-on.

Services share host-mounted local NVMe scratch. Datasets/checkpoints are never baked into images.

## Source of truth vs notification

Discord is a **notification layer**, not the source of truth.

Authoritative observability remains:

- SQLite campaign DB;
- structured JSONL/Parquet experiment/system metrics;
- worker logs;
- GPU/system telemetry;
- artifact/checkpoint manifests;
- durable copies in GMI Cold Storage.

## Discord policy

Send concise, actionable events rather than per-epoch spam.

### Immediate events

- campaign start/complete;
- circuit breaker activation;
- critical storage/disk/GPU incident;
- final drain start;
- unresolved safety failure;
- AI repair escalation/exhaustion where useful.

### Aggregated progress

Periodic summary (e.g. every few hours or meaningful phase change) containing:

- phase and elapsed/remaining time;
- trials completed/running/promoted/pruned/failed/recovered;
- current best models/metrics;
- average/recent GPU utilization and throughput;
- unsynced storage backlog and drain estimate;
- active repair queue.

### Delivery behavior

- webhook URL is a secret and never committed;
- notifier uses a local durable spool/queue;
- network/webhook failure never blocks GPU work;
- use exponential backoff and bounded retention;
- disable arbitrary mentions for externally generated log text;
- record notification delivery status in structured logs.

## Lightweight status report

Instead of requiring Grafana, periodically generate a small static HTML or Markdown status artifact containing:

- campaign phase/progress;
- current trial;
- leaderboard snapshot;
- GPU utilization/VRAM/throughput;
- storage backlog;
- recent failures;
- AI repair queue;
- remaining time / current drain reserve.

Store it in the campaign artifact directory. Grafana/Prometheus can be added later if operational needs justify it.

## Health telemetry

Collect at a moderate interval (e.g. 5–15s):

- GPU utilization;
- VRAM;
- power/temperature;
- CPU utilization;
- RAM;
- disk free/read/write;
- network RX/TX;
- dataloader wait where measurable;
- training samples/s and steps/s.

Prefer DCGM/NVML/PyTorch profiler/Nsight for targeted diagnostics; do not run heavy profilers continuously.

## H200 bootstrap sequence

Automate approximately:

1. verify container/runtime/GPU driver compatibility;
2. authenticate/mount/configure storage access;
3. verify dataset manifest and checksums;
4. stage active training data to local NVMe;
5. benchmark loader/storage/GPU;
6. run golden smoke test;
7. start controller/evaluator/sync/notifier services;
8. begin campaign calibration.

## Shutdown/drain checklist

Before the instance is intentionally terminated, require:

- all finalist/best checkpoints uploaded and verified;
- validation predictions uploaded;
- campaign DB snapshot uploaded;
- configs/manifests uploaded;
- final leaderboard/report uploaded;
- AI repair/event history preserved;
- no critical sync backlog;
- checksums verified for required artifacts.

The scheduler should enter DRAIN early enough to satisfy this automatically.

## Dress rehearsal

Before the paid H200 run, execute the exact Compose/service topology against a tiny dataset and cheap/available GPU where possible. Intentionally kill containers and inject representative failures. Acceptance means the system completes unattended with valid durable artifacts and useful Discord notifications.
