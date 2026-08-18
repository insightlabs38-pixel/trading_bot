# Phase 3 Progress — CPU Data Pipeline

Last updated: **2026-08-18**

Status: **BLOCKED — production/provider validation only**

This file records validation detail for Phase 3. The authoritative task list remains
`IMPLEMENTATION_PLAN.md`, whose Phase 3 checkboxes are currently stale relative to the implemented
code and this detailed progress record.

## Stage restartability and publication

- [x] All Phase 3 stage identities are represented explicitly from `00_raw` through
  `09_packed_training_data`.
- [x] A common `StageRunner` executes producers in temporary scratch space and supports one or many
  output artifacts per stage.
- [x] Stage outputs use content-addressed durable keys and standard artifact manifests containing
  checksum, size, schema/version, Git/config provenance, generation stage, and upstream IDs.
- [x] `_SUCCESS.json` is published only after every output and manifest has been written and
  verified.
- [x] A completed stage is reused only after its success marker, manifests, checksums, sizes, and
  requested lineage/provenance verify successfully.
- [x] Corrupt completed stages fail closed rather than silently rerunning or accepting damaged data.
- [x] Partial/unreferenced content-addressed outputs can be safely reused after checksum and
  manifest/provenance verification.

This closes the Phase 3 preamble requirement that stages be restartable and publish manifests plus a
success marker only after successful completion. Stage-specific transforms remain ordinary pure
functions and can be supplied to the common runner without embedding storage concerns in feature or
label logic.

## Vendor acquisition

- [x] Provider-independent `VendorAdapter` interface.
- [x] Canonical immutable vendor request schema with stable SHA-256 identity.
- [x] Bounded transient retry/backoff.
- [x] Request-rate limiting with deterministic test hooks.
- [x] Raw vendor payloads are preserved byte-for-byte under content-addressed immutable keys.
- [x] Exact non-secret request parameters, download timestamp, vendor source ID, response metadata,
  size, and checksum are persisted for every successful acquisition.
- [x] Repeated identical payloads reuse the immutable raw object while retaining distinct download
  audit records, including downloads occurring at the same timestamp.
- [x] Secret-like request fields and response metadata are rejected at the generic acquisition
  boundary instead of being serialized into durable audit records.
- [x] Provider-neutral HTTPS GET transport with injectable vendor URL construction.
- [x] API keys can be injected only at runtime through a configured header or query parameter;
  credential-like public URLs and sensitive committed headers are rejected.
- [x] HTTP 408/425/429/5xx responses and transport failures map to bounded-retry acquisition errors;
  ordinary 4xx responses map to permanent request failures.
- [x] Only a whitelisted HTTP response-metadata subset is retained by the shared HTTP transport.
- [ ] **BLOCKED — broad-equities provider adapter:** the data contract still describes the vendor as
  Massive/Polygon-style and explicitly makes the final subscription/API tier provisional until
  purchase/price verification. Do not freeze an endpoint/request shape before that selection.
- [ ] **BLOCKED — execution-data provider adapter:** the contract still describes Databento-style
  execution data as the preferred direction rather than a final selection; no subscription/API
  credentials are available for integration validation.

## Raw validation

- [x] Timezone-aware/UTC timestamp validation.
- [x] Duplicate asset/timestamp detection.
- [x] Impossible OHLC and invalid price/volume/VWAP detection.
- [x] Missing intraday interval and out-of-order detection.
- [x] Whole-session completeness checks when the caller supplies the expected session-date set.
- [x] Expected asset IDs can also be supplied, so a symbol with zero rows across the entire request
  cannot disappear from missing-session checks.
- [x] Expected-session validation remains calendar/provider independent; production exchange
  calendars can feed the validator without changing its contract.
- [x] Structured anomaly reports; raw rows are never silently repaired or deduplicated.

## Security master

- [x] Permanent security IDs, point-in-time symbol history, listing/delisting dates.
- [x] Every security must have symbol history beginning at listing and covering its known listing
  lifetime; active histories remain open-ended and delisted histories end at delisting.
- [x] Security type, exchange, sector/issuer metadata.
- [x] Corporate actions and historical delisted-security lookup.
- [x] Corporate-action numeric fields must remain finite; duplicate actions/source IDs are rejected.
- [x] Overlap guards prevent ticker reuse from splicing unrelated securities.
- [x] Symbol-change actions must agree with the immediately prior symbol and the new symbol's
  effective-date period; no-op symbol changes are rejected.
- [x] Chronological symbol-history lookup is exposed for downstream point-in-time consumers.

## Adjustment/canonicalization

- [x] Raw values retained beside causal adjusted values.
- [x] Split factors use only actions effective on/before each bar date.
- [x] Future actions cannot alter historical canonical observations.
- [x] Cash dividends remain explicit and enter only explicit total-return calculations.
- [x] Canonicalization independently rejects invalid/non-finite raw OHLCV/VWAP, duplicate
  security/timestamp rows, and non-finite derived adjustment state instead of assuming callers
  always invoked raw validation first.

## Resampling

- [x] 1-minute base plus 5/15/30/60-minute and daily session aggregates.
- [x] New York regular-session alignment and no cross-session buckets.
- [x] Complete intervals required by default; bar-close timestamp causality preserved.
- [x] Equal-time multi-asset output ordering is deterministic independent of input ordering.
- [x] Malformed/non-finite canonical inputs and unsupported frequencies fail closed.
- [ ] **BLOCKED — production calendar validation:** exchange holidays/early closes require the
  intended production calendar dependency/source.

## Point-in-time universe

- [x] Versioned policy, strict pre-rebalance liquidity history, deterministic ADV ranking.
- [x] Historical delisted securities are eligible while listed; non-common/future names excluded.
- [x] Frozen snapshots persist security-master/policy identity.
- [x] Non-finite prices/volumes/thresholds are rejected and duplicate same-security/day liquidity
  observations cannot be double-counted into ADV.
- [ ] **BLOCKED — production methodology:** exact cadence/thresholds and real vendor history remain
  intentionally unfrozen.

## Feature pipeline

- [x] Reference causal implementation covers all required OHLC/VWAP, return, volume, volatility,
  range/ATR, momentum/trend, market/sector-relative, rank, session, liquidity, regime, and identity
  feature categories.
- [x] Prefix-invariance tests prove future observations cannot alter earlier feature rows.
- [x] Model-ready feature inputs must be finite/valid and every derived feature is checked for
  finiteness before publication, preventing NaN/Inf propagation or overflow from reaching packing.
- [ ] **BLOCKED — production columnar/performance validation:** Polars/PyArrow/DuckDB and Python
  3.12 cannot be installed in this sandbox.

## Labels

- [x] Exact-endpoint 5/15/30/60-minute future returns, excess return, direction, rank, future
  volatility, and quantile/rank targets.
- [x] Missing endpoints are not interpolated; label generation is isolated from feature code.
- [x] Derived returns/excess returns/volatility must remain finite, including extreme-value paths.

## Splits

- [x] Versioned immutable chronological walk-forward split manifest.
- [x] Split schema version is pinned; identifiers are normalized; final-holdout ID cannot collide
  with a routine fold ID.
- [x] Stable split hash and default-deny final-holdout access guard.
- [ ] **BLOCKED — production dates:** actual train/validation/holdout boundaries require the
  finalized production data period.

## Packing

- [ ] **BLOCKED — research representation:** Parquet + Zstd cannot be implemented/validated here
  because PyArrow/Polars cannot be installed and no Parquet engine is available.
- [x] Deterministic NumPy `.npy` memory-mapped reference training representation.
- [x] Features/targets, asset IDs, and exact integer-derived nanosecond timestamps are preserved with
  every sample.
- [x] Non-finite/unrepresentable float32 values and duplicate security/timestamp samples are rejected
  before publication.
- [x] Dataset/split versions, feature/target names, dimensions, sizes, and SHA-256 checksums are
  persisted in deterministic metadata.
- [x] Fresh pack publication uses a temporary directory followed by atomic rename.
- [x] Loader verifies file integrity plus array shapes/dtypes before exposing sequential batches.
- [x] Loader benchmark reports samples/sec and MiB/sec.
- [x] Equivalent sample sets pack deterministically independent of input order.

**BLOCKED — final training format/target-hardware benchmark:** the data plan requires representative
benchmarks before freezing the production packed format. The NumPy memmap path is a validated
reference, not a claim that it is the final H200 loader representation.

## Validation performed

Historical repository/sandbox validation already recorded for earlier Phase 3 increments includes:

```text
114 passed  # earlier Phase 2 + implemented Phase 3 data-contract suite
17 passed   # provider-neutral HTTP transport focused mirror
26 passed   # Raw validation + Security master existing/new regressions
```

For this completion audit, a focused sandbox mirror exercised existing Raw validation/Security
master regressions plus adversarial checks across acquisition, canonicalization, resampling,
universe construction, features, labels, splits, packing, restartable stage publication, and a
repeated synthetic multi-asset raw → packed integration build:

```text
38 passed
```

The audit mirror also passes `python -m compileall`. The private repository is not mounted in this
sandbox, so this **is not described as a fresh full-repository pytest run**; the new repository tests
are committed alongside the implementation so the same adversarial and raw→packed contracts can be
run directly in a normal checkout/CI environment.

The synthetic integration gate runs raw validation → security-master canonicalization → 5-minute
resampling → point-in-time universe selection → causal features → future-only labels → immutable
split lookup → deterministic packed training data twice and verifies equivalent packed identity and
arrays independent of sample input order.

No live broad-equities/execution-provider request, real exchange-calendar holiday/early-close run,
production universe/split freeze, Parquet pipeline, or H200 loader benchmark is claimed as validated
because those inputs/dependencies/hardware are unavailable here.

## Remaining Phase 3 blockers

- [ ] **BLOCKED — vendor selection/integration:** concrete broad-equities and optional execution-data
  provider adapters require finalized subscriptions/API semantics and live credentials.
- [ ] **BLOCKED — production exchange calendar:** holiday/early-close behavior must be validated
  against the selected production calendar source.
- [ ] **BLOCKED — production universe methodology:** exact cadence/thresholds require the frozen
  production methodology and real history.
- [ ] **BLOCKED — production columnar research representation:** Parquet + Zstd and columnar
  performance require PyArrow/Polars or an equivalent available production dependency.
- [ ] **BLOCKED — production split dates:** actual boundaries depend on the finalized data period.
- [ ] **BLOCKED — final loader representation/H200 benchmark:** freeze only after representative
  target-hardware measurements.

## Phase 3 gate

All sandbox-verifiable provider-independent contracts are implemented and audited, including a
synthetic deterministic raw → packed composition gate and restartable manifest/success-marker
publication primitive. The **production Phase 3 gate remains BLOCKED** only by the external items
listed above. Those blockers must be resolved before the production dataset is frozen or an H200
campaign is treated as valid.
