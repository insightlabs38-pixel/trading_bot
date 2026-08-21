# Phase 10 Progress — Experiment Configuration and Search Spaces

Last updated: **2026-08-20**

Status: **COMPLETE — version-controlled campaign search-manifest gate passed**

`IMPLEMENTATION_PLAN.md` remains the authoritative checklist. Phase 10 freezes the experiment-registry/search-space contract that the Phase 11 scheduler will consume; it does not execute the H200 campaign and makes no GPU/H200 performance claim.

## Frozen v1 campaign manifest

The version-controlled manifest is:

`configs/campaigns/h200_tournament_v1.yaml`

It is validated by `src/trading_bot/campaign/search_space.py` and carries a deterministic canonical-JSON SHA-256 identity. The loader/enumerator deliberately imports no PyTorch or model modules so the future scheduler can inspect campaign state without importing CUDA/model code.

The v1 registry contains:

- 19 architecture entries;
- 17 mandatory families;
- 2 optional/reference-only families (`logistic_direction` and `foundation_adapter`);
- 13 searchable neural/advanced/custom families;
- 45 canonical model-size presets across reference/small/medium/large entries;
- one fixed architecture-screening objective: 15-minute Huber excess-return regression;
- five currently executable objective variants;
- three additional objective variants defined but intentionally `planned_not_selected` until the common trainer/loss/head stack can execute them faithfully.

The optional `foundation_adapter` remains external-artifact-gated and is not silently promoted into the mandatory campaign before a real pretrained checkpoint is selected and accepted.

## Search space

The manifest preregisters bounded axes for:

- learning rate: `1e-4`, `3e-4`, `1e-3`;
- weight decay: `0`, `1e-4`, `1e-3`;
- dropout candidates: `0`, `0.1`, `0.2`;
- context lengths: `32`, `64`, `128`;
- effective batch size `256` through microbatch/accumulation combinations `64×4`, `128×2`, and `256×1`;
- screening seed `17` and finalist seeds `17`, `29`, `43`.

Families opt into only axes supported by their current constructor/training contract. In particular, the dropout candidates are defined for future/eligible model configurations but are not forced into architectures whose current constructors do not expose a dropout parameter. This avoids silently changing the implemented architecture merely to populate a search dimension.

## Objective registry

Enabled objective variants are:

1. `excess_mse_15m`;
2. `excess_huber_15m`;
3. `ranking_15m`;
4. `direction_bce_15m`;
5. `multitask_return_rank_direction_15m`.

The following Phase 10 candidates are fully represented in the manifest but deliberately not launchable yet:

- `multitask_return_rank_vol_direction_15m`;
- `multi_horizon_huber_15_30m`;
- `distributional_quantile_15m`.

They remain `planned_not_selected` because the current common loss/head path does not yet implement those exact training semantics end-to-end. The manifest validator rejects any architecture that references a non-enabled objective, so unsupported experiments fail closed rather than being silently approximated.

## Campaign budgets

The relative rung budgets follow the already-documented campaign/scheduler design rather than inventing a new H200 policy:

- calibration: 9 representative configurations at 5% of family full budget;
- screening: 66 configurations at 15%, promote 18;
- promotion: 18 configurations at 50%, promote 6;
- objective search: 18 configurations at 50%, promote 4;
- finalists: 4 systems at full budget using all 3 finalist seeds.

This yields **123 planned fits** before runtime-adaptive scheduler behavior, inside the existing design target of approximately 100–130 fits. Phase 11 will operationalize wall-clock launch/deadline adaptation from measured runtime; Phase 10 freezes the experiment-space/rung contract, not target-hardware throughput.

## CPU verification

The Phase 10 acceptance tests verify, among other things:

- strict YAML validation and unknown-field rejection inherited from immutable Pydantic config models;
- deterministic canonical JSON and SHA-256 identity;
- campaign loading without importing PyTorch;
- independent equality of every Phase 8 advanced small/medium/large YAML preset with `advanced_model_spec`;
- independent equality of every Phase 9 custom small/medium/large YAML preset with `custom_model_spec`;
- construction/forward checks for all five neural-baseline canonical scales;
- planned objective variants cannot enter launchable architecture lists;
- batch arithmetic, rung breadth, objective references, and registry lookups fail closed;
- editing the YAML search range changes the enumerated space and manifest identity without editing Python.

The exact reconciled PR #20 head was `487e8842fdb61b1298d4c8b4e5c106e79bac7b03`. Permanent read-only CPU CI run `32438778975` / job `96645135014` tested synthetic merge `472f4c88661860f5ec803a98625db8e610fc749c` against base `286fb112c099d8911893bfc54e40ed3a26fbc613`.

Results:

- Ubuntu 24.04.4;
- Python 3.12.3;
- uv 0.10.12;
- 73-package lock resolution and locked `baseline-cpu` sync passed;
- Ruff passed;
- Ruff format passed: **116 files already formatted**;
- strict mypy passed: **no issues in 59 source files**;
- compileall passed;
- pytest: **339 passed, 1 skipped in 17.52s**;
- the sole skip is the existing opt-in Phase 2 real-S3 provider gate requiring external endpoint credentials.

PR #20 was merged with the required standard merge method as `bd6eeb634475b6ecda7684a7dacf49b6678fd042`. Comparing that actual merge commit with the CI-tested synthetic merge reports no changed files, so the content landed on `main` is identical to the content exercised by the final PR gate.

## Gate

**PASSED — VERSION-CONTROLLED CAMPAIGN SEARCH MANIFEST.** The intended v1 experiment registry, canonical size presets, objective candidates, bounded search axes, seed policy, mandatory/optional pools, and screening/promotion/finalist budgets can be loaded, validated, hashed, and enumerated from YAML without editing Python. Phase 11 can consume this contract to implement durable scheduler state and adaptive execution.
