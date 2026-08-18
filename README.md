# trading_bot

Research-first medium-frequency U.S. equities trading system.

The project is intentionally planning-first: data, evaluation, scheduler behavior, paper-trading gates, and operational safety are specified before the training implementation is written.

## Intended system

- **Style:** medium-frequency intraday trading, not HFT.
- **Decision cadence:** approximately once per minute; trades are optional, not forced.
- **Primary forecast horizons:** 15 and 30 minutes, with 5- and 60-minute auxiliary horizons.
- **Universe:** a point-in-time liquid U.S. equity universe, initially targeting roughly 750–1,500 securities.
- **Research compute:** one H200-class GPU for an approximately 48-hour adaptive architecture campaign.
- **Production philosophy:** deterministic risk controls around probabilistic ML models; paper trading before limited live capital.

## Start here

1. [`PLAN.md`](PLAN.md) — concise project contract and frozen decisions.
2. [`docs/README.md`](docs/README.md) — documentation index.
3. [`docs/architecture.md`](docs/architecture.md) — system, campaign, and live-trading diagrams.
4. [`docs/evaluation_contract.md`](docs/evaluation_contract.md) — model-selection metrics and formulas.
5. [`docs/scheduler_and_recovery.md`](docs/scheduler_and_recovery.md) — 48-hour campaign controller design.
6. [`docs/paper_and_live_trading.md`](docs/paper_and_live_trading.md) — promotion path from shadow mode to limited live capital.
7. [`AGENTS.md`](AGENTS.md) — non-negotiable rules for future human/AI implementation work.

## Repository status

This scaffold contains **plans and interfaces only**. Training, preprocessing, execution, and scheduler code will be implemented after the contracts in `docs/` are reviewed and frozen.

## Safety / research note

Backtests and paper trading are research tools, not guarantees of live profitability. The project deliberately separates model research, execution simulation, risk controls, and live deployment so that a strong historical result cannot bypass operational or risk gates.
