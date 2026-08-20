# Phase 6 Progress — Canonical Evaluator and Backtester

Last updated: **2026-08-20**

Status: **IN PROGRESS — CPU/reference evaluator gate passed; real factor/BBO data acceptance pending**

`IMPLEMENTATION_PLAN.md` remains the authoritative checklist. This file records the detailed validation evidence and separates provider-independent CPU correctness from production-data acceptance.

## Canonical evaluator surface

- [x] Gross return, position changes, one-way turnover, fees, spread, slippage, impact, net return, and daily NAV are computed through one canonical return-accounting path.
- [x] Cost components are explicit and stressable rather than hidden inside a model or leaderboard score.
- [x] Portfolio weights are explicit evaluator inputs. Phase 6 does not invent or silently freeze a production portfolio-construction methodology.
- [x] Saved Phase 5 predictions are read through an evaluator-owned Parquet/Zstd reader that verifies manifest/data checksums, schema, metadata, compression, exact timestamps, and asset identity.

## Predictive and economic metrics

- [x] Cross-sectional Rank IC by timestamp with mean, median, standard deviation, ICIR, and positive-period fraction.
- [x] Rank IC breakdowns by fold, regime, horizon, and sector where labels are supplied.
- [x] Net Sharpe using the frozen annualization convention.
- [x] CAGR, Sortino, maximum drawdown, maximum drawdown duration, Calmar, ES95, and worst day.
- [x] ES95 follows the frozen contract exactly: losses are `-return`, VaR95 is the 95th loss quantile, and ES95 is the mean loss at or beyond that threshold.

## Friction, stress, and robustness

- [x] Average turnover, cumulative traded weight/notional, total modeled cost, break-even transaction cost, and trade/rebalance counts.
- [x] Frozen-grid cost stress and spread stress.
- [x] Latency/execution-delay stress over explicit delayed-return observations; production delayed-return/BBO inputs remain an external-data concern.
- [x] Fold-level statistics, positive-fold fraction, seed dispersion, and regime diagnostics.
- [x] Deflated Sharpe Ratio reference implementation with attempted-trial count accounting.
- [x] CSCV-style Probability of Backtest Overfitting diagnostic over a supplied strategy/configuration family.

## Attribution

- [x] Provider-independent OLS factor-attribution interface over caller-supplied factor observations.
- [x] Market beta, arbitrary common-factor exposures, intercept/alpha, residual volatility, and fit diagnostics are available without selecting a data vendor.
- [ ] **EXTERNAL DATA ACCEPTANCE** — rerun attribution on the selected production common-factor dataset once that dataset/provider is frozen.

## Execution-oriented finalist diagnostics

- [x] Buy/sell implementation shortfall uses the frozen sign convention where positive means worse execution.
- [x] Mean/median/p90/p95 implementation-shortfall summaries.
- [x] Participation and ADV/liquidity diagnostics.
- [x] Deterministic BBO/L1 reference simulator for market and limit orders.
- [x] The simulator enforces decision timestamp plus requested latency and never consumes pre-eligible quotes.
- [x] No-lookahead fixture proves a better pre-decision/pre-latency quote cannot be used.
- [ ] **EXTERNAL DATA ACCEPTANCE** — rerun the execution simulator/diagnostics on the selected real BBO/L1 execution dataset once available.

## Leaderboard and reports

- [x] Hard validity/disqualification gates precede ranking.
- [x] Ranking is hierarchical/lexicographic in the documented predictive/economic order rather than one opaque score.
- [x] Machine-readable JSON and human-readable Markdown reports are deterministic and SHA-256 verified.
- [x] Corruption/tampering of saved inputs or reports fails closed.

## Acceptance fixtures

- [x] Hand-calculated return/cost fixtures.
- [x] Zero-return strategy fixture.
- [x] Buy-and-hold fixture.
- [x] Known-cost fixture.
- [x] Drawdown fixture.
- [x] No-lookahead execution timing fixture.
- [x] Saved-prediction corruption validation.
- [x] Full saved-prediction → metrics → leaderboard → report gate.
- [x] Strong fresh-process gate performs the full evaluator/leaderboard/report path while asserting `trading_bot.training` and `torch` never enter `sys.modules`.

## Authoritative CPU verification

Permanent read-only GitHub Actions on Python **3.12.3** with the committed dependency lock:

```text
uv lock --check: pass
Ruff: pass
Formatting: 98 files formatted
mypy: success, no issues in 50 source files
compileall: pass
pytest: 263 passed, 1 skipped
```

The single skipped test is the existing opt-in Phase 2 real-S3 provider gate. The Phase 6 saved-Parquet and fresh-process evaluator gates execute in this environment with PyArrow available.

## Remaining Phase 6 blockers

- [ ] Validate attribution against the selected production common-factor dataset/provider.
- [ ] Validate BBO/L1 execution diagnostics against the selected real execution dataset/provider.
- [ ] Supply the frozen production portfolio-construction outputs and production stress inputs when the production campaign dataset/methodology is finalized; Phase 6 intentionally treats these as inputs rather than changing the frozen methodology.

## Phase 6 status

**CPU/REFERENCE EVALUATOR GATE PASSED.** All Phase 6 implementation and acceptance work that can be meaningfully verified with synthetic/reference data on CPU is complete. The phase remains **IN PROGRESS** only for real external factor/execution-data acceptance. No real-provider execution-quality or factor-exposure result is claimed by the CPU gate.
