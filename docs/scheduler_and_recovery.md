# Scheduler and Recovery Plan

Status: **BASELINE**

## Purpose

The H200 scheduler is a deadline-aware **campaign controller**, not merely an HPO loop. Its responsibility is to maximize useful, defensible research output before a fixed rental deadline while continuing unattended through common failures.

The CPU preprocessing path does not need this scheduler; it should be a resumable/idempotent DAG run to completion.

## Top-level architecture

The controller itself imports no CUDA/model code. Each GPU trial runs in a fresh subprocess so a CUDA/Triton crash cannot take down campaign state.

Persistent campaign state lives in SQLite on local scratch and is periodically snapshotted to durable storage.

## Campaign states

```text
BOOTSTRAP
  -> CALIBRATION
  -> SCREENING
  -> PROMOTION
  -> OBJECTIVE_SEARCH
  -> FINALISTS
  -> DRAIN
  -> COMPLETE
```

## Trial states

```text
PENDING
  -> STARTING
  -> RUNNING
  -> EVALUATING
  -> SYNCING
  -> COMPLETE
```

Side states include:

- PRUNED;
- INTERRUPTED;
- RETRYABLE_FAILURE;
- TERMINAL_FAILURE;
- QUARANTINED;
- AI_REPAIR_PENDING;
- AI_REPAIR_EXHAUSTED.

## Durable state

Suggested SQLite tables:

- `campaign` — start/deadline/current phase;
- `trials` — immutable config, parent, status, retry count;
- `metrics` — intermediate/final metrics;
- `checkpoints` — path/checksum/step/status;
- `runtime_stats` — throughput/runtime observations;
- `events` — append-only high-level scheduler events;
- `failures` — classified failure records;
- `promotions` — promotion/pruning reasons;
- `resources` — summarized GPU/CPU/disk/network health;
- `repairs` — deterministic and AI repair attempts/results.

Snapshot the DB periodically and immediately after major state transitions.

## Trial immutability

Never silently mutate a failed trial.

Example:

```text
trial_0041
batch=1024
status=FAILED_OOM

trial_0041_r1
parent=trial_0041
batch=512
grad_accum=2
status=PENDING
```

The experiment lineage remains inspectable and scientifically meaningful.

## Failure policy

### Deterministic first

| Failure | Default action |
|---|---|
| transient worker/process failure | resume latest valid checkpoint; bounded retry |
| CUDA OOM | full worker restart; derive smaller microbatch while preserving effective batch if possible |
| NaN/Inf | one recovery from last good checkpoint; then stability child/fail |
| Triton compile failure | fail custom backend; attempt validated PyTorch reference where defined |
| illegal memory access | kill worker process group; GPU health check; bounded retry |
| worker hang | TERM → grace period → KILL → checkpoint recovery |
| evaluator crash | retry evaluator without stopping GPU training |
| storage upload failure | retain local artifact and async retry; training continues while disk permits |
| corrupted input shard | stop dependent trials and raise dataset incident |
| critical disk pressure | stop launching trials; prioritize sync/cleanup |
| deterministic config error | terminal failure, no repeated retry |

Retry budgets are bounded globally and per trial.

## Heartbeats and hang detection

Workers periodically publish:

- trial/phase;
- training step;
- last progress timestamp;
- loss/primary metric;
- samples/s;
- latest checkpoint age;
- explicit state such as COMPILING, DATALOADING, TRAINING, CHECKPOINTING, EVALUATING.

Different states receive different timeout thresholds so legitimate compilation is not confused with a deadlocked kernel.

## Wall-clock guards

Every trial has both a training-budget limit and hard wall-clock limit. Screening trials may not consume hours simply because one architecture is unexpectedly slow.

The scheduler continuously computes:

\[
T_{usable}=T_{deadline}-T_{now}-T_{drain}
\]

and refuses to launch work whose conservative runtime estimate no longer fits.

Runtime estimates are updated from measured campaign observations by architecture family, size, context, precision, and budget. Near the deadline use conservative upper-percentile estimates rather than mean runtime.

## Adaptive campaign behavior

Example policy tiers (exact values live in frozen config):

- >24h usable: broad exploration;
- 12–24h: normal promotion and controlled experimentation;
- 6–12h: stop low-value architecture expansion;
- 3–6h: finalists/high-value ablations only;
- ~1.5–3h: avoid expensive new work;
- final adaptive reserve: DRAIN only.

The drain reserve is itself dynamic based on outstanding evaluation work, unsynced bytes, observed storage throughput, and safety margin.

## Successive halving

Use explicit campaign rungs rather than training every trial to completion:

```text
Rung 0: ~60–70 configurations, ~10–15% budget
  -> promote ~16–20
Rung 1: ~35–50% budget
  -> promote strongest subset
Rung 2: full budget / finalists
```

Financial metrics are noisy, so enforce a minimum grace budget before any performance-based pruning.

## Promotion

Promotion follows `docs/evaluation_contract.md`. Do not optimize only raw return or a single noisy Sharpe observation. Validity gates and stability rules apply before performance ranking.

## Checkpointing

Use atomic checkpoint creation:

```text
write .tmp
  -> close/flush
  -> checksum
  -> atomic rename to .ready
  -> async sync
  -> destination verify
  -> eligible for local cleanup
```

A resumable checkpoint includes model, optimizer, LR scheduler, training cursor/step, RNG states, precision state where relevant, config IDs, and dataset/split IDs.

Use time-based checkpointing in addition to best-score and graceful-stop checkpoints because epoch length varies by architecture.

## Circuit breaker

Repeated infrastructure-like failures in a short window trigger a global circuit breaker rather than burning through the queue.

Diagnostics should include:

- GPU health/smoke test;
- disk space/I/O;
- dataset sample verification;
- storage connectivity;
- minimal CUDA forward/backward/checkpoint test.

Only resume launches when the environment passes the required checks.

## Golden canary

Maintain a tiny known-good model/config/dataset smoke test. Run it at bootstrap and after serious infrastructure incidents. Optionally rerun later as a performance canary to detect unexplained throughput degradation.

## AI repair tier

AI assistance is optional recovery infrastructure, never campaign authority.

Escalation:

1. deterministic recovery;
2. fast non-reasoning/low-latency model diagnosis/repair;
3. stronger reasoning mode only for unresolved higher-value trials;
4. expensive/slower reasoning model only when scientific value and remaining time justify it;
5. quarantine for human review.

The specific API/model provider is configured, not hardcoded into scientific logic.

### AI sandbox rules

AI receives a sanitized debug bundle containing only relevant source, config, stack trace, environment versions, tensor shapes, recent logs, and preferably synthetic/minimal reproduction data.

It may propose patches to model/kernel/test code in an isolated worktree/container.

It may not alter:

- train/validation/final-holdout definitions;
- scoring/promotion rules;
- transaction-cost assumptions;
- data/universe contracts;
- secrets;
- campaign DB directly;
- cloud infrastructure or instance lifecycle;
- unrelated architectures.

### Repair gates

Any proposed patch must pass:

1. static/syntax/lint/import gate;
2. unit tests;
3. GPU smoke test on tiny data;
4. regression/numerical gate against known/reference behavior.

Custom Triton repairs additionally compare outputs/gradients to the PyTorch reference across representative shapes/dtypes/strides.

The H200 continues with another known-good trial while repair occurs asynchronously.

## Campaign simulation mode

Before paid compute, support a fast simulated campaign that substitutes fake trial durations/scores/failures and compresses the full 48-hour state machine into minutes.

Inject at least:

- OOM;
- hang;
- Triton crash;
- slow/outage storage;
- evaluator crash;
- scheduler restart;
- unexpected 3x runtime;
- corrupted checkpoint;
- disk pressure;
- AI repair success and failure.

Acceptance criterion: the simulated campaign reaches DRAIN/COMPLETE with valid finalists, durable state, and no unsynced critical artifacts without manual intervention.
