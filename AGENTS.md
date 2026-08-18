# AGENTS.md

Instructions for human and AI contributors working in this repository.

## Project intent

This is a research-first medium-frequency equities trading system. Preserve scientific validity, reproducibility, operational safety, and auditability over convenience.

## Protected invariants

The following must **not** be changed casually or by automated repair agents:

1. Final-holdout boundaries and access rules.
2. Walk-forward split definitions once a campaign begins.
3. Transaction-cost and execution assumptions during a campaign.
4. Leaderboard metric definitions or promotion rules during a campaign.
5. Point-in-time universe methodology during a campaign.
6. Risk limits in paper/live trading without explicit versioned review.
7. Dataset lineage/manifests for completed experiments.

If any protected invariant must change, create a new dataset/campaign version and document the reason.

## Data rules

- No future information may appear in features, universe membership, normalization statistics, labels, or execution assumptions.
- Do not construct historical universes from today's surviving tickers.
- Raw vendor data is immutable; transformations create versioned derived datasets.
- Keep raw and adjusted price concepts explicit rather than overwriting one with the other.
- All model-ready datasets require manifests with checksums, schema/version, feature definitions, label definitions, and split IDs.

## Experiment rules

- Every trial has an immutable ID and config.
- If a config changes after failure, create a child trial; do not mutate the original trial's meaning.
- Record all attempted trials, including failed/pruned trials.
- Save validation predictions for promoted/final models, not only weights.
- Use the same economic accounting code across architectures.
- Do not optimize against the final holdout.

## Scheduler / AI repair rules

- The campaign controller owns the deadline and remains deterministic.
- Model workers run in subprocesses so CUDA/Triton failures cannot kill the controller.
- Known failures use deterministic recovery first.
- AI repair runs in an isolated worktree/container and receives sanitized context only.
- AI repair may patch model/kernel implementation and tests, but may not alter protected invariants, secrets, cloud infrastructure, or campaign state directly.
- Every AI patch must pass static, unit, GPU smoke, and regression gates before a repaired child trial is queued.
- Repeated AI repair failure results in quarantine, not infinite retries.

## Security

- Never commit `.env`, API keys, broker credentials, Discord webhook URLs, S3 credentials, private keys, or licensed raw datasets.
- Secrets are runtime-injected only.
- Logs and AI debugging bundles must be sanitized for credentials.
- The broker/risk layer must fail closed on stale state or reconciliation uncertainty.

## Implementation style

- Prefer explicit interfaces and small testable modules.
- Training code should use plain PyTorch unless a stronger reason emerges.
- CPU and GPU runtimes are separate container images.
- Object storage is durable state; local NVMe is disposable hot scratch.
- Do not put datasets or checkpoints inside Docker images.
- Configuration belongs in versioned config files, not scattered constants.

## Required tests before paid GPU campaign

A compressed dress rehearsal must exercise:

- clean campaign completion;
- OOM fallback;
- NaN/Inf handling;
- worker hang and forced termination;
- Triton compile/runtime failure;
- checkpoint corruption/recovery;
- controller restart;
- evaluator crash;
- storage outage/retry;
- disk-pressure handling;
- Discord outage;
- AI repair pass/fail paths;
- adaptive deadline/drain behavior.

The H200 rental should begin only after the rehearsal can finish unattended with valid artifacts.
