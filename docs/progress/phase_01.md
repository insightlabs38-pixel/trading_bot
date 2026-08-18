# Phase 1 Progress — Project and Configuration Foundations

Last updated: **2026-08-18**

Status: **IN PROGRESS**

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

### Sandbox validation performed

The configuration suite was executed under Python **3.13.5**, Pydantic **2.13.4**, PyYAML
**6.0.3**, and pytest **9.0.2**.

Result:

```text
20 passed in 0.11s
```

Validated behaviors include:

- example YAML loading and configuration round-trip;
- top-level and nested unknown-field rejection;
- environment substitution, defaults, endpoint values, and secret values;
- missing environment-variable failure;
- frozen configuration objects;
- secret redaction from manifest serialization;
- deterministic/order-independent canonical JSON serialization;
- JSON-compatible model-parameter enforcement;
- objective/dataset horizon compatibility;
- notification webhook requirements;
- provider-neutral AI repair settings and required credentials when enabled;
- deliberately unfrozen paper/live numeric limits and mandatory safety gates when enabled;
- explicit spread-cost presence in evaluation configuration.

`python -m compileall` also passes for the configuration package and tests, and the changed Python
files contain no lines longer than the repository's configured 100-character Ruff limit.

**BLOCKED — target-environment confirmation:** Ruff and mypy are not installed in this sandbox,
and the sandbox cannot install them because outbound package-index access is unavailable. The
same configuration suite should be rerun under the pinned Python 3.12/uv environment before the
Phase 1 gate is declared passed.

## Common metadata

- [ ] Identifiers.
- [ ] Config hashing.
- [ ] Git/container/environment capture.
- [ ] Run-manifest generation.

The Phase 1 gate remains **IN PROGRESS** because common metadata and the minimal run-manifest
command are still outstanding. Stable config-hash tests remain part of that next section.
