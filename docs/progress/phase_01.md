# Phase 1 Progress — Project and Configuration Foundations

Last updated: **2026-08-20**

Status: **COMPLETE**

This file records validation detail for Phase 1. The authoritative task list remains
`IMPLEMENTATION_PLAN.md`.

## Status convention

- `[x]` — complete and acceptance criteria met.
- `[ ]` — not complete.
- `IN PROGRESS` — work has started but the acceptance gate is not met.
- `BLOCKED` — cannot proceed until the stated dependency/decision is resolved.
- `OPTIONAL` — useful but not on the critical path.

## Python project

- [x] `pyproject.toml` and package metadata are present.
- [x] Python 3.12 is pinned as the supported runtime.
- [x] `uv` dependency groups are defined for core, CPU, GPU, development, and test usage.
- [x] A Python 3.12-resolved `uv.lock` is committed.
- [x] Ruff, pytest, and strict mypy policy/configuration are defined.
- [x] `src/trading_bot` package initialization is present.
- [x] CPU-only GitHub Actions verification is installed on `main`.
- [x] CI uses only one standard `ubuntu-latest` hosted runner with no GPU/larger-runner dependency.
- [x] CI uses an ephemeral read-only `GITHUB_TOKEN`, system Python 3.12, and a temporary pinned `uv`
  bootstrap environment.
- [x] `scripts/verify_cpu.sh` provides the same Ruff/format/mypy/compileall/pytest gate locally.
- [x] A supported-environment CI run has completed green.

## Supported-environment verification — 2026-08-20

After the repository became public, GitHub assigned the standard hosted runner normally. The
permanent read-only workflow then completed successfully on Ubuntu 24.04 with Python **3.12.3** and
`uv` **0.10.12**.

The authoritative Phase 1 run executed:

```text
uv lock
uv sync --locked --group cpu
ruff check .
ruff format --check .
mypy
python -m compileall -q src tests
pytest -q
```

Result:

```text
Ruff: all checks passed
Formatting: all files formatted
mypy: success, no issues in 34 source files
pytest: 241 passed, 1 skipped
```

The single skipped test is the opt-in real S3 provider gate in
`tests/integration/test_phase2_s3_provider_gate.py`; it requires a real S3-compatible test endpoint
and credentials and therefore does not block the Phase 1 project/configuration gate. The later
Phase 3 Parquet/Zstd additions are included in the 241-test result above.

The earlier pre-step GitHub Actions failures are now confirmed to have been hosted-runner minute
availability rather than repository code or workflow-command failures.

## Configuration system

- [x] Strong Pydantic schemas implemented for storage.
- [x] Strong Pydantic schemas implemented for dataset.
- [x] Strong Pydantic schemas implemented for preprocessing.
- [x] Strong Pydantic schemas implemented for model configuration.
- [x] Strong Pydantic schemas implemented for training.
- [x] Strong Pydantic schemas implemented for objectives.
- [x] Strong Pydantic schemas implemented for evaluation.
- [x] Strong Pydantic schemas implemented for campaigns.
- [x] Strong Pydantic schemas implemented for scheduler settings.
- [x] Strong Pydantic schemas implemented for notifications.
- [x] Strong Pydantic schemas implemented for AI repair.
- [x] Strong Pydantic schemas implemented for paper/live risk settings.
- [x] Environment interpolation supports `${VAR}` and `${VAR:-default}` for secrets/endpoints.
- [x] Configs serialize deterministically into manifest-safe canonical JSON with secret redaction.
- [x] Unknown fields are rejected at every typed schema level with `extra="forbid"`.
- [x] Configuration models are frozen after validation.
- [x] Cross-section validation covers campaign/drain compatibility and objective/dataset horizons.

### Contract-alignment fixes completed

- AI repair does not hardcode a provider, model, or API endpoint; enabling it requires explicit
  runtime provider/model/key configuration, matching `docs/scheduler_and_recovery.md`.
- Paper/live risk configuration does not invent numeric risk defaults. Numeric limits remain unset
  while disabled and must be supplied explicitly before the risk configuration is enabled, matching
  `docs/paper_and_live_trading.md`.
- Evaluation configuration requires explicit fee, spread, slippage, and impact components, matching
  the canonical cost equation in `docs/evaluation_contract.md`.
- Model-specific parameter values are constrained to JSON-compatible values so manifest
  serialization cannot silently fail on arbitrary Python objects.

## Common metadata

- [x] Typed immutable identifiers are defined for dataset version, split version, model
  configuration, trial, campaign, checkpoint, and prediction artifacts.
- [x] Stable SHA-256 config hashing is implemented over canonical secret-redacted config JSON.
- [x] Model configuration IDs are content-derived from deterministic canonical serialization.
- [x] Git SHA/branch/dirty-state capture is implemented with explicit environment overrides for
  packaged/containerized execution.
- [x] Container identity capture is allowlisted and never serializes arbitrary environment data.
- [x] Python/platform/package-version environment capture is dependency-light and does not import
  GPU frameworks merely to inspect their installed versions.
- [x] Immutable run-manifest generation records config identity, typed IDs, Git/container/runtime
  provenance, seed, precision, and compile mode.
- [x] `python -m trading_bot.metadata` provides the minimal run-manifest command required by the
  Phase 1 gate and does not require market data or a GPU.

## Prior sandbox evidence

Before GitHub-hosted verification was available, the focused configuration/common-metadata suite
passed under Python 3.13.5 with 36 tests. That result remains useful regression evidence, but the
Python 3.12 GitHub Actions run above is now the authoritative supported-runtime gate.

## Gate

A minimal command can load a validated configuration, generate a run manifest, and exit successfully
on a supported machine without requiring market data or a GPU.

**PASSED — Python 3.12 CPU CI is green and the dependency lock is committed.**
