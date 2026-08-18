# Phase 3 Progress — CPU Data Pipeline

Last updated: **2026-08-18**

Status: **IN PROGRESS**

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
  available in the sandbox, so the planned Massive/Polygon-style adapter cannot be implemented or
  integration-tested responsibly yet.
- [ ] **BLOCKED — execution-data provider adapter:** a Databento/equivalent selection and credentials
  are not available, so that provider-specific adapter remains deferred.

## Raw validation

- [x] Timezone-aware timestamp validation.
- [x] UTC-normalization contract validation.
- [x] Duplicate asset/timestamp detection.
- [x] Impossible OHLC relationship detection.
- [x] Finite/positive price and VWAP checks.
- [x] Finite/non-negative volume checks.
- [x] Missing intraday interval detection scoped by asset and UTC date.
- [x] Out-of-order row detection.
- [x] Anomalies are recorded in a structured report; input rows are never silently repaired or
  deduplicated.

The missing-interval primitive intentionally does not embed a U.S. exchange calendar yet. It
identifies gaps within an observed asset/date sequence and avoids treating cross-date boundaries as
missing intraday bars. Calendar/session semantics are handled in the later resampling/session layer.

## Security master

- [x] Permanent security-ID keyed records.
- [x] Point-in-time ticker/symbol history.
- [x] Listing and delisting dates.
- [x] Security-type classification.
- [x] Exchange metadata.
- [x] Sector/issuer reference metadata hooks.
- [x] Corporate-action records for splits, cash dividends, symbol changes, mergers/spinoffs/other.
- [x] Historical lookup retains delisted securities instead of filtering to current survivors.
- [x] Overlap guards prevent ticker reuse from splicing unrelated securities together.
- [x] Corporate actions must reference a known security and occur during its listed lifetime.

## Adjustment/canonicalization

- [x] Raw prices/volume/VWAP are preserved explicitly in every canonical bar.
- [x] Permanent security ID and point-in-time symbol are attached to canonical observations.
- [x] Cumulative split factors use only actions effective on or before the bar date.
- [x] Split-normalized prices and inverse-adjusted volume provide a continuous causal share basis.
- [x] Future split actions do not alter historical canonical observations.
- [x] Cash dividends are recorded explicitly rather than silently folded into prices.
- [x] An explicit total-return helper incorporates same-date cash dividends on the causal share basis.
- [x] Corporate-action/symbol boundaries are resolved through the point-in-time security master.

The implementation intentionally avoids a hidden back-adjusted/total-return vendor convention. Any
later research representation can derive such a view from explicit raw values and action records,
but causal features never receive future corporate-action information.

## Resampling

- [x] Canonical one-minute bars are the base-resolution input contract.
- [x] Derived 5-minute bars.
- [x] Derived 15-minute bars.
- [x] Derived 30-minute bars.
- [x] Derived 60-minute bars.
- [x] Daily session aggregates.
- [x] `America/New_York` regular-session alignment with configurable open/close/base interval.
- [x] Asset/date/bucket grouping prevents cross-session state leakage.
- [x] Complete source intervals are required by default; partial buckets must be requested explicitly.
- [x] Output timestamps use the last source observation in the bucket, preserving bar-close causality.
- [x] Adjusted OHLCV and volume-weighted VWAP aggregation are deterministic.

The reference resampler does not yet encode exchange holidays/early closes. Those require an
explicit trading-calendar source/configuration; the current contract makes session boundaries
explicit and is fully testable on synthetic regular-session fixtures.

## Validation performed

Vendor Acquisition + Raw Validation + Security Master + Canonicalization + Resampling tests plus the
complete Phase 2 storage suite pass in the dedicated sandbox venv:

```text
78 passed
```

`compileall` passes and all new files satisfy the repository's 100-character line-length policy.
Git blob hashes are checked against the exact sandbox-tested files before section merge.

The provider/calendar limitations do not prevent the next section, **Point-in-time universe**, from
being implemented and tested against synthetic daily liquidity/history fixtures.
