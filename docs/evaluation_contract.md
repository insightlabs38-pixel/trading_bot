# Evaluation Contract

Status: **BASELINE — freeze before the paid campaign**

This document defines the model-selection/economic evaluation contract. The scheduler must not change these definitions because a favored architecture is losing.

## Canonical return accounting

All economic metrics derive from one canonical **net daily NAV return series**.

For portfolio weights chosen at time `t`, only returns available after the decision timestamp may be earned.

Gross portfolio return:

\[
r^{gross}_{t+1}=\sum_i w_{i,t}r_{i,t+1}
\]

Traded weight/notional change:

\[
G_t=\sum_i|w_{i,t}-w_{i,t^-}|
\]

One-way turnover:

\[
TO_t=\frac{1}{2}G_t
\]

Modeled trading cost:

\[
C_t=\sum_i|\Delta w_{i,t}|\left(fee_{i,t}+spread_{i,t}+slippage_{i,t}+impact_{i,t}\right)
\]

Net return:

\[
\boxed{r^{net}_t=r^{gross}_t-C_t}
\]

For headline risk metrics, aggregate intraday trading to daily NAV returns.

## Validity gates

A trial cannot be promoted if it violates any applicable hard gate:

- data leakage / timestamp-causality failure;
- invalid/missing cost accounting;
- non-finite predictions or returns not explained by a handled failure;
- impossible leverage/exposure under the specified portfolio rules;
- corrupted/incomplete evaluation data;
- final-holdout contamination;
- insufficient required evaluation coverage.

## Predictive metrics

### Cross-sectional Rank IC

At prediction timestamp `t`:

\[
IC_t=\rho_S(\hat r_{1,t},\ldots,\hat r_{N,t},r_{1,t+h},\ldots,r_{N,t+h})
\]

Report:

- mean Rank IC;
- median Rank IC;
- standard deviation;
- percent of periods with positive IC;
- IC Information Ratio `mean(IC)/std(IC)`;
- IC by horizon, regime, sector, and fold where useful.

Rank IC is the primary pure-prediction metric because the strategy mainly needs relative opportunity ranking, not perfectly calibrated point returns.

### Optional calibration/distributional metrics

If a model predicts uncertainty/quantiles/probabilities, evaluate with proper scoring/calibration diagnostics appropriate to the output, but do not replace economic evaluation with forecasting error alone.

## Primary economic metrics

### Net Sharpe

Using daily net excess returns `x_t`:

\[
SR=\sqrt{252}\frac{\bar x}{s_x}
\]

Net Sharpe is the primary economic ranking metric, but never the sole selection criterion.

### CAGR

For `D` trading days:

\[
CAGR=\left(\frac{NAV_D}{NAV_0}\right)^{252/D}-1
\]

### Maximum drawdown

\[
Peak_t=\max_{s\le t}NAV_s
\]

\[
DD_t=\frac{NAV_t}{Peak_t}-1
\]

\[
MDD=\min_t DD_t
\]

Also report maximum drawdown duration.

### Calmar

\[
Calmar=\frac{CAGR}{|MDD|}
\]

### Sortino

Using a consistently defined daily target/risk-free rate `T`:

\[
\sigma_d=\sqrt{\frac{1}{n}\sum_t\min(r_t-T,0)^2}
\]

\[
Sortino=\sqrt{252}\frac{\bar r-T}{\sigma_d}
\]

### Expected Shortfall

For losses `L_t=-r_t`:

\[
VaR_{95}=Q_{0.95}(L)
\]

\[
ES_{95}=E[L\mid L\ge VaR_{95}]
\]

Report ES95 and worst single day.

## Friction / capacity metrics

### Turnover

\[
\overline{TO}=\frac{1}{T}\sum_t\frac12\sum_i|\Delta w_{i,t}|
\]

Report average turnover and total/annualized traded notional.

### Break-even transaction cost

If `P_gross` is gross cumulative trading P&L in portfolio-return units and `V_trade` is cumulative traded weight/notional:

\[
c_{BE}=\frac{P_{gross}}{V_{trade}}
\]

\[
c_{BE,bps}=10{,}000\,c_{BE}
\]

### Implementation shortfall

For a buy relative to decision/arrival reference price `P_d` and average execution `P_e`:

\[
IS=\frac{P_e-P_d}{P_d}
\]

Reverse the sign convention for sells so positive values consistently represent worse execution. Report mean/median/p90/p95 in bps by ticker, trade size, volatility, time-of-day, and spread where possible.

### Participation/liquidity

Track order notional as a fraction of relevant market volume and position notional as a fraction of average daily dollar volume. Do not assume infinite capacity.

## Robustness metrics

For finalists report, at minimum:

- median fold Sharpe;
- worst fold Sharpe;
- percent of folds with positive after-cost Sharpe/return;
- seed dispersion;
- regime-specific performance;
- cost sensitivity;
- spread sensitivity;
- execution-delay sensitivity.

A baseline project gate is that at least ~70% of required walk-forward folds should be positive after costs; exact campaign thresholds must be frozen in config before execution.

## Latency sensitivity

Because this is medium-frequency rather than HFT, finalists should degrade gradually as artificial execution delay increases. Test representative delays such as:

- 0 s;
- 250 ms;
- 1 s;
- 5 s;
- 15 s;
- 30 s.

A strategy whose edge disappears at sub-second delay is likely solving a latency-sensitive problem inconsistent with the project intent.

## Cost/spread stress tests

Evaluate finalists at baseline assumptions and stressed multipliers, e.g.:

- 1.0x;
- 1.25x;
- 1.5x;
- 2.0x costs.

The exact grid is frozen before finalist evaluation.

## Multiple-testing / overfitting diagnostics

### Deflated Sharpe Ratio (DSR)

Compute DSR for serious finalists using a tested implementation consistent with the published definition, accounting for multiple trials, track-record length, skewness, and kurtosis. A planning guideline is to regard DSR >= 0.95 as strong evidence and lower values as increasing warnings; this is a project convention, not a universal law.

### Probability of Backtest Overfitting (PBO)

Use CSCV-style analysis on the relevant strategy/configuration family to estimate the probability that in-sample selection produces an out-of-sample loser. Treat PBO as a diagnostic alongside the complete trial history, not a magic pass/fail number.

## Factor attribution

As a diagnostic, regress daily excess strategy returns against common market/style factors where data is available:

\[
r_{p,t}-r_{f,t}=\alpha+\beta_M MKT_t+\beta_S SMB_t+\beta_H HML_t+\beta_R RMW_t+\beta_C CMA_t+\beta_{Mom} MOM_t+\epsilon_t
\]

Report alpha and factor exposures so an apparent AI edge is not mistaken for unrecognized beta/momentum/style exposure.

## Systems metrics

Track separately from trading quality:

- parameter count;
- training wall time;
- samples/s and steps/s;
- peak VRAM;
- compile overhead;
- checkpoint size;
- p50/p95/p99 inference latency;
- deployment memory footprint.

Systems efficiency is a tie-breaker and a deployment constraint, not a substitute for robust signal.

## Selection hierarchy

Use this order conceptually rather than collapsing everything into one opaque score:

1. validity gates;
2. Rank IC / predictive stability;
3. net Sharpe;
4. Calmar, MDD, Sortino, ES95;
5. transaction-cost and latency robustness;
6. fold/seed/regime stability;
7. DSR/PBO diagnostics;
8. factor attribution;
9. systems efficiency as a tie-breaker.

Promotion logic may use explicit threshold/lexicographic rules derived from this hierarchy, but those rules must be frozen before the paid campaign.

## Final holdout

The final holdout is accessed only after architecture, model weights/training procedure, features, targets, portfolio construction, transaction-cost assumptions, and risk settings are frozen. It is run once for the frozen system. A disappointing final holdout is a result, not permission to tune against it.
