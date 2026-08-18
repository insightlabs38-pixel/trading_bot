# Paper and Live Trading Plan

Status: **BASELINE**

Historical performance is not sufficient for live deployment. The production path is deliberately staged.

## Promotion path

```text
untouched final holdout
  -> shadow mode
  -> broker paper trading
  -> tiny live canary
  -> limited live capital
  -> normal live allocation
```

A material change to model weights, features, portfolio construction, risk logic, or execution logic resets the relevant paper/live validation stage.

## Phase 0 — shadow mode

Suggested minimum: roughly 5 live trading days.

The full production feature/model/portfolio/risk path runs on live data, but no broker orders are submitted.

Persist for every decision:

- source data timestamp/version/hash where practical;
- feature snapshot/reference;
- model outputs;
- target weights;
- proposed orders;
- risk decision/reason;
- timestamps at each stage.

After close, replay recorded inputs through the frozen system. Live and replay outputs should match within the expected numerical tolerance.

Purpose: detect deployment, timestamp, state, and live-feature mismatches before order APIs are involved.

## Phase 1 — broker paper trading

Use the exact intended production brokerage/execution code against a paper account.

Planning duration:

- minimum ~40 trading days;
- prefer ~60 trading days for the planned 15–30 minute primary horizon;
- enough activity to avoid evaluating a handful of fills (planning targets: >=100 rebalance events and >=250 order/fill events, adjusted to actual strategy frequency).

Paper fills are treated as operational/execution simulations, not proof of real market fill quality.

## Three P&L ledgers

Maintain simultaneously:

### A — ideal signal ledger

Uses a predefined theoretical/arrival reference price to isolate signal quality.

### B — conservative internal execution simulator

Uses explicit spread, fee, slippage, delay, impact, and partial-fill assumptions.

### C — broker paper ledger

Uses actual fills reported by the broker's paper environment.

Interpret:

- A − B: modeled execution drag;
- B − C: disagreement between internal execution assumptions and broker simulator.

Do not loosen conservative execution assumptions merely because paper fills look better.

## Operational hard gates

Before any live capital, require no unresolved critical failures in:

- duplicate unintended orders;
- stale-data trading;
- position reconciliation;
- account-balance reconciliation;
- risk-limit enforcement;
- broker disconnect recovery;
- market-data disconnect recovery;
- corporate-action handling;
- restart/recovery state;
- kill switch.

The broker's acknowledged orders/fills/positions are authoritative. Never infer a position solely from what the bot intended to submit.

## Fault injection during paper trading

Deliberately test:

- network disconnect;
- broker API disconnect;
- delayed/stale market data;
- duplicate callbacks/events;
- partial fills;
- order rejection;
- process restart;
- machine reboot;
- local-state corruption;
- model inference timeout;
- storage/logging outage.

The system must either recover correctly or fail closed.

## Statistical paper acceptance

Do not require an arbitrary short-period paper Sharpe. Instead, before paper trading starts, generate historical/block-bootstrap or walk-forward distributions for an equivalent-length period covering:

- cumulative net return;
- drawdown;
- turnover;
- volatility;
- trade/rebalance counts;
- gross/net exposures;
- Rank IC where observable.

Freeze plausibility bands before seeing the paper result. Paper performance should remain statistically/operationally consistent with the frozen historical system rather than necessarily being positive over every short window.

## Paper-to-live acceptance

### Operational — mandatory

- no critical safety failures;
- reconciled broker state remains correct;
- restart/recovery and disconnect scenarios pass;
- kill switch passes;
- stale-data rejection passes;
- shadow/replay consistency passes.

### Strategy consistency

- paper behavior falls within preregistered historical plausibility ranges;
- drawdown is not outside the predefined severe historical band;
- turnover/exposure/trade counts remain broadly consistent;
- no obvious collapse in predictive Rank IC where measurable;
- cost/execution differences are understood.

## Live canary

Start with deliberately tiny capital and conservative position/order limits. The purpose is to measure real implementation shortfall, broker/exchange behavior, and reconciliation—not to maximize profit.

Track:

- implementation shortfall;
- realized spread/slippage;
- partial-fill behavior;
- order rejection/cancel behavior;
- latency distributions;
- production prediction consistency;
- risk and reconciliation incidents.

Scale capital only after a predefined observation period and review.

## Deterministic live risk layer

The learned model does not control these safeguards. Planned controls include:

- maximum per-position notional/weight;
- gross and net exposure caps;
- leverage cap;
- maximum order notional;
- liquidity/participation limits;
- maximum outstanding orders;
- stale-data cutoff;
- duplicate-order protection/idempotency;
- daily loss / drawdown stop;
- market/session checks;
- model/inference timeout handling;
- broker-state reconciliation gate;
- unconditional kill switch.

Exact numeric limits are intentionally not frozen until the broker, account size, and paper results are known.

## Medium-frequency execution intent

The live system may evaluate opportunities approximately once per minute and can execute multiple trades per hour, but it is not designed for microsecond competition. Finalists must retain useful performance under seconds-scale execution-delay stress and realistic spread/cost assumptions.

Deeper LOB, RL execution, and multi-agent market simulation are future execution-improvement research, not prerequisites for the first production system.
