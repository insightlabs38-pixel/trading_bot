# Phase 1 Progress — Project and Configuration Foundations

Last updated: **2026-08-18**

Status: **BLOCKED — target-environment confirmation**

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
- [x] Ruff, pytest, and mypy policy/configuration are defined.
- [x] `src/trading_bot` package initialization is present.

**BLOCKED — target-environment confirmation:** the current sandbox provides Python 3.13 rather
than the pinned Python 3.12 runtime and has no package-index access, so a clean `uv sync` plus
Ruff/mypy/full-test run in the supported environment cannot be performed here. This is a
validation-environment limitation, not an implementation gap in the Python-project section.

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

- AI repair no longer hardcodes a provider, model, or API endpoint; enabling it requires explicit
  runtime provider/model/key configuration, matching `docs/scheduler_and_recovery.md`.
- Paper/live risk configuration no longer invents numeric risk defaults. Numeric limits remain
  unset while disabled and must be supplied explicitly before the risk configuration is enabled,
  matching `docs/paper_and_live_trading.md`.
- Evaluation configuration now requires explicit fee, spread, slippage, and impact components,
  matching the canonical cost equation in `docs/evaluation_contract.md`.
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

### Sandbox validation performed

The combined configuration/common-metadata suite was executed under Python **3.13.5**, Pydantic
**2.13.4**, PyYAML **6.0.3**, and pytest **9.0.2**.

Result:

```text
36 passed in 0.84s
```

Validated metadata behaviors include:

- all seven identifier families and unsafe-ID rejection;
- a hard-coded golden SHA-256 for the example configuration;
- order-independent config hashing and material-change hash sensitivity;
- stable content-derived model configuration IDs;
- Git environment overrides and real clean/dirty Git repository capture;
- allowlisted container metadata capture without secret/environment leakage;
- installed/missing package-version capture without importing heavy frameworks;
- immutable, UTC-normalized run manifests using redacted canonical config content;
- a CLI smoke test that loads the example config and writes a manifest without market data/GPU.

`python -m compileall` also passes for the configuration and metadata packages/tests, and the
changed Python files contain no lines longer than the repository's configured 100-character
Ruff limit.

**BLOCKED — target-environment confirmation:** the sandbox still lacks the pinned Python 3.12
runtime, Ruff, and mypy, and cannot install them because package-index access is unavailable.
Before the Phase 1 gate is declared passed, run the full suite plus Ruff/mypy in the supported
Python 3.12/uv environment.
