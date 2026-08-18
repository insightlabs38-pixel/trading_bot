# Phase 4 Progress — Dataset Validation and Leakage Protection

Last updated: **2026-08-18**

Status: **BLOCKED — production-data validation only**

This file records Phase 4 validation detail. `IMPLEMENTATION_PLAN.md` remains the nominal task list,
but its Phase 4 checkboxes are stale relative to the implemented code and this detailed progress
record.

## Leakage/data-contract invariants

- [x] Feature prefix invariance: observations after a decision timestamp cannot alter earlier
  feature rows.
- [x] Cross-sectional feature panels are prefix-invariant when future timestamps/assets are added.
- [x] Feature pipeline does not import or invoke the future-label pipeline.
- [x] Labels require exact future endpoints and are generated from future-only target data.
- [x] Point-in-time universe ignores liquidity observations on/after the rebalance date.
- [x] Future corporate actions cannot alter historical canonical bars.
- [x] Resampling cannot use observations from a future bucket to change a completed prior bucket.
- [x] Adjacent trading sessions remain separate during session-aware aggregation.
- [x] Missing source intervals are not silently filled during complete-bucket resampling.
- [x] Train/validation chronology is enforced by immutable split manifests.
- [x] Routine partition lookup never identifies the protected final holdout.
- [x] `RoutineSplitManifest` physically omits final-holdout date fields while retaining the full
  split SHA-256 and holdout ID for lineage/audit references.
- [x] The full `SplitManifest` remains available only for workflows that explicitly need final
  evaluation; routine/search-facing code can consume the date-free routine view instead.
- [x] Overlapping ticker reuse across unrelated security IDs is rejected.
- [x] Legitimate non-overlapping ticker reuse resolves to the correct permanent security ID at each
  historical date.

These tests consolidate causal guarantees already individually validated during Phase 3 and form a
regression gate for later columnar/GPU optimizations. The new routine-only split type closes the
previous weakness where the guarded `final_holdout_range()` helper existed but callers holding the
full manifest could still read the final-holdout date field directly.

## Missing-data behavior

- [x] Missing raw sessions/intervals are surfaced by validation rather than silently repaired.
- [x] Complete resampling drops incomplete buckets by default instead of filling missing bars.
- [x] Labels require exact future endpoints and do not interpolate absent endpoint observations.
- [x] Dataset audits count missing close/volume values explicitly.
- [x] Non-finite/non-positive prices, non-finite/negative volumes, duplicate asset/timestamp rows,
  and non-finite derived returns are surfaced explicitly in audit diagnostics.

No implicit forward fill, interpolation, ticker splicing, or future-derived repair is part of the
Phase 4 reference contract.

## Dataset audits

- [x] Dataset summary report with row count, asset count, and observed timestamp range.
- [x] Missing close/volume counts plus duplicate-row, non-finite, non-positive-price, and
  negative-volume diagnostics.
- [x] Unique asset counts through time.
- [x] Per-asset chronological return sanity distribution.
- [x] Non-finite derived returns are counted and excluded from numeric summaries instead of
  poisoning the report with NaN/Inf.
- [x] Volume sanity distribution uses overflow-resistant summary arithmetic for extreme finite
  values.
- [x] Consecutive point-in-time universe entry/exit and bounded symmetric one-way membership
  turnover statistics.
- [x] Duplicate universe snapshot dates, duplicate security IDs, and malformed rank sequences fail
  closed rather than producing misleading turnover.
- [x] Routine train/validation split timeline report.
- [x] Protected final-holdout ID and full split hash are reportable without exposing protected
  holdout dates through the routine audit path.
- [x] Deterministic canonical JSON and human-readable Markdown report output.

## Validation performed

Historical validation recorded before this audit included the complete Phase 2 storage suite, all
then-implemented Phase 3 data-contract tests, the original Phase 4 leakage suite, and dataset-audit
tests:

```text
129 passed
```

For this completion audit, the existing Phase 4 leakage/audit behavior, Phase 3 completion/raw/
security regressions, and new Phase 4 adversarial tests were exercised together in the available
sandbox mirror:

```text
61 passed
```

The focused mirror also passes `python -m compileall`. The private repository is not mounted in this
sandbox, so this **is not described as a fresh full-repository pytest run**. The new regression tests
are committed beside the implementation for a normal checkout/CI environment.

The new adversarial coverage includes final-holdout date omission from routine views/reports,
future cross-sectional panel invariance, explicit session separation, historical ticker reuse,
duplicate/invalid audit observations, extreme finite numeric values, derived-return overflow,
changing universe sizes, duplicate universe dates/members, and malformed universe ranks.

## Remaining Phase 4 blockers

- [ ] **BLOCKED — finalized production dataset:** the leakage/data-contract and audit suites must be
  rerun on the real provider-derived dataset before architecture results on that dataset are trusted.
- [ ] **BLOCKED — production calendar validation:** session/early-close/holiday behavior must be
  validated against the selected production exchange-calendar source.
- [ ] **BLOCKED — frozen production universe/splits:** exact universe methodology and production
  split dates remain external Phase 3 decisions and must be frozen before the production Phase 4
  gate can pass.
- [ ] **BLOCKED — production columnar representation:** the same invariants/audits must be rerun on
  the finalized Parquet/columnar representation once its unavailable dependencies are present.

## Phase 4 gate

All sandbox-verifiable leakage/data-contract invariants and dataset-audit primitives are implemented
and audited. The **production Phase 4 gate remains BLOCKED** only because the finalized production
dataset and its external Phase 3 dependencies do not yet exist in this environment.

Reference/synthetic architecture work may use these regression gates, but no architecture result on
the eventual full production dataset should be treated as trustworthy until this exact suite passes
that frozen dataset and representation.
