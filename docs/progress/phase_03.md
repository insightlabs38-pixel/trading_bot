# Phase 3 Progress — CPU Data Pipeline

Last updated: **2026-08-18**

Status: **BLOCKED**

This file records validation detail for Phase 3. The authoritative task list remains
`IMPLEMENTATION_PLAN.md`.

## Vendor acquisition

- [x] Provider-independent `VendorAdapter` interface.
- [x] Canonical immutable vendor request schema with stable SHA-256 identity.
- [x] Bounded transient retry/backoff.
- [x] Request-rate limiting with deterministic test hooks.
- [x] Raw vendor payloads are preserved byte-for-byte under content-addressed immutable keys.
- [x] Exact request parameters, download timestamp, vendor source ID, response metadata, size,
  and checksum are persisted for every successful acquisition.
- [x] Repeated identical payloads reuse the immutable raw object while retaining distinct download
  audit records.
- [ ] **BLOCKED — broad-equities provider adapter:** final vendor subscription/API contract is not
  available in the sandbox, so the planned provider adapter cannot be integration-tested yet.
- [ ] **BLOCKED — execution-data provider adapter:** a Databento/equivalent selection and credentials
  are not available, so that provider-specific adapter remains deferred.

## Raw validation

- [x] Timezone-aware/UTC timestamp validation.
- [x] Duplicate asset/timestamp detection.
- [x] Impossible OHLC and invalid price/volume/VWAP detection.
- [x] Missing intraday interval and out-of-order detection.
- [x] Structured anomaly reports; raw rows are never silently repaired or deduplicated.

## Security master

- [x] Permanent security IDs, point-in-time symbol history, listing/delisting dates.
- [x] Security type, exchange, sector/issuer metadata.
- [x] Corporate actions and historical delisted-security lookup.
- [x] Overlap guards prevent ticker reuse from splicing unrelated securities.

## Adjustment/canonicalization

- [x] Raw values retained beside causal adjusted values.
- [x] Split factors use only actions effective on/before each bar date.
- [x] Future actions cannot alter historical canonical observations.
- [x] Cash dividends remain explicit and enter only explicit total-return calculations.

## Resampling

- [x] 1-minute base plus 5/15/30/60-minute and daily session aggregates.
- [x] New York regular-session alignment and no cross-session buckets.
- [x] Complete intervals required by default; bar-close timestamp causality preserved.
- [ ] **BLOCKED — production calendar validation:** exchange holidays/early closes require the
  intended calendar dependency/source.

## Point-in-time universe

- [x] Versioned policy, strict pre-rebalance liquidity history, deterministic ADV ranking.
- [x] Historical delisted securities are eligible while listed; non-common/future names excluded.
- [x] Frozen snapshots persist security-master/policy identity.
- [ ] **BLOCKED — production methodology:** exact cadence/thresholds and real vendor history remain
  intentionally unfrozen.

## Feature pipeline

- [x] Reference causal implementation covers all required OHLC/VWAP, return, volume, volatility,
  range/ATR, momentum/trend, market/sector-relative, rank, session, liquidity, regime, and identity
  feature categories.
- [x] Prefix-invariance tests prove future observations cannot alter earlier feature rows.
- [ ] **BLOCKED — production columnar/performance validation:** Polars/PyArrow/DuckDB and Python
  3.12 cannot be installed in this sandbox.

## Labels

- [x] Exact-endpoint 5/15/30/60-minute future returns, excess return, direction, rank, future
  volatility, and quantile/rank targets.
- [x] Missing endpoints are not interpolated; label generation is isolated from feature code.

## Splits

- [x] Versioned immutable chronological walk-forward split manifest.
- [x] Stable split hash and default-deny final-holdout access guard.
- [ ] **BLOCKED — production dates:** actual train/validation/holdout boundaries require the
  finalized production data period.

## Packing

- [ ] **BLOCKED — research representation:** Parquet + Zstd cannot be implemented/validated here
  because PyArrow/Polars cannot be installed and no Parquet engine is available.
- [x] Deterministic NumPy `.npy` memory-mapped reference training representation.
- [x] Features/targets, asset IDs, and nanosecond timestamps are preserved with every sample.
- [x] Dataset/split versions, feature/target names, dimensions, sizes, and SHA-256 checksums are
  persisted in deterministic metadata.
- [x] Pack publication uses a temporary directory followed by atomic rename.
- [x] Loader verifies file sizes/checksums and exposes sequential batches over memory maps.
- [x] Loader benchmark reports samples/sec and MiB/sec.
- [x] Equivalent sample sets pack deterministically independent of input order.

A sandbox test exposed and fixed a destructive cleanup bug in the first packing draft: using
`Path()` as a post-rename sentinel could direct cleanup at the current directory. Publication now
uses `None` after a successful atomic rename, and the rebuilt full suite passes.

**BLOCKED — final training format/target-hardware benchmark:** the data plan requires representative
benchmarks before freezing the production packed format. The NumPy memmap path is a validated
reference, not a claim that it is the final H200 loader representation.

## Validation performed

All implemented Phase 3 data-contract tests plus the complete Phase 2 storage suite pass in the
dedicated sandbox venv:

```text
114 passed
```

`compileall` passes and all new Python files satisfy the repository's 100-character line policy.

## Phase 3 gate

The dependency-light pipeline contracts are implemented and individually validated, but the full
production raw → packed gate is **BLOCKED** by provider-specific acquisition, production universe/
split decisions, Parquet/columnar dependencies, real exchange-calendar validation, and target-
hardware loader benchmarking. These blockers do not prevent Phase 4 leakage protections from being
implemented against the validated synthetic/reference pipeline.
