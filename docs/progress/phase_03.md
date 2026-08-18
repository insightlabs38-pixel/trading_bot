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

## Validation performed

Vendor Acquisition + Raw Validation + Security Master tests plus the complete Phase 2 storage suite
pass in the dedicated sandbox venv:

```text
60 passed
```

`compileall` passes and the new files satisfy the repository's 100-character line-length policy.
Git blob hashes are checked against the exact sandbox-tested files before section merge.

The provider blockers do not prevent the next section, **Adjustment/canonicalization**, from being
implemented and tested against synthetic security-master/corporate-action fixtures.
