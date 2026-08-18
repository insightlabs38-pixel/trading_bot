# Data and Storage Plan

Status: **BASELINE**, with vendor/CPU-provider details marked provisional until purchase/quote.

## Data strategy

Use two tiers of market data rather than forcing every research question onto one extremely expensive dataset.

### Tier 1 — broad alpha dataset

Purpose: architecture tournament and medium-frequency signal research.

Target:

- roughly 10 years of U.S. equity history;
- point-in-time liquid universe, initially ~750–1,500 common stocks;
- 1-minute OHLCV / aggregate data as the broad source resolution;
- corporate actions and reference/security-master information;
- derived 5m, 15m, 30m, 60m, and daily views;
- optional trades where useful for liquidity features.

Current preferred vendor direction: Massive/Polygon-style broad U.S. equities historical data, subject to final plan/price verification before purchase.

### Tier 2 — execution dataset

Purpose: finalist execution validation and later microstructure research.

Target a much smaller liquid universe using:

- trades;
- BBO/L1 or MBP-1;
- possibly MBP-10/MBO for selected microstructure/execution studies.

Current preferred vendor direction: Databento-style usage-based market microstructure data, acquired narrowly enough to control cost.

## Point-in-time universe

Do not backtest today's survivors historically.

At each universe-rebalance date, select eligible securities using information known by that date, such as trailing dollar volume, minimum price, trading-history length, listing/security type, and liquidity filters.

A representative universe statistic is trailing average dollar volume:

\[
ADV_{i,t}=\frac{1}{L}\sum_{k=1}^{L}P_{i,t-k}V_{i,t-k}
\]

Universe membership should be frozen on a regular schedule such as weekly or monthly to avoid unnecessary churn.

## Processing DAG

Each stage is independently restartable and writes a manifest plus success marker only after verification.

```text
00_raw
  -> 01_validated
  -> 02_security_master
  -> 03_adjusted/canonical
  -> 04_resampled
  -> 05_point_in_time_universe
  -> 06_features
  -> 07_labels
  -> 08_splits
  -> 09_training_pack
```

Do not make a single all-or-nothing preprocessing job whose failure requires restarting hours of completed work.

## Feature principles

Keep primitive observations as well as engineered features. Candidate categories include:

- returns across multiple horizons;
- OHLC/VWAP/range information;
- realized volatility;
- volume, dollar volume, relative volume;
- momentum/trend measures;
- market and sector context;
- cross-sectional ranks;
- liquidity proxies;
- minute-of-day/day-of-week/session state;
- regime statistics such as breadth, volatility, correlation, and dispersion.

Normalization statistics must be fitted causally using training/history available at that timestamp.

## Labels

Prepare several labels from the same frozen underlying dataset so objective experiments do not require rebuilding source data:

- future excess return;
- cross-sectional rank / ranking pairs;
- direction;
- future volatility;
- multi-horizon returns at 5m, 15m, 30m, 60m;
- distributional/quantile targets where appropriate.

Primary economic interest is expected to center on 15m and 30m horizons.

## Storage format

### Research/canonical representation

Use partitioned **Parquet + Zstandard** with explicit schemas and reasonably large objects/shards (avoid millions of tiny files).

### Training representation

Create a separate packed representation optimized for sequential/batched reads by PyTorch. The exact format can be selected after loader benchmarks (memory-mapped arrays, Arrow-based packing, or other sharded tensors).

The training format is derived and disposable; canonical Parquet plus manifests is the reproducible research source.

## Storage architecture

### GMI Cold Storage

Planned as the primary durable working store because the quoted project assumptions are:

- approximately $4/TB/month;
- no transfer charge for GPU ↔ Cold Storage traffic;
- approximately 1–10 Gbps observed/expected throughput, commonly near 10 Gbps.

These are operational assumptions supplied during planning and must be re-verified at execution time.

### Local NVMe

Use local instance NVMe only for hot scratch:

- currently active dataset shards/folds;
- compiler/cache files;
- in-progress checkpoints;
- temporary predictions.

Nothing irreplaceable should exist only on ephemeral local disk.

### External backup

Optionally mirror only critical artifacts externally: code/config manifests, security master/universe metadata, processed dataset manifests, final checkpoints, final predictions/results, and custom kernels. Full raw data may be omitted if vendor redownload is reliable and economical.

## CPU preprocessing provider

Prepare for two interchangeable paths:

1. **External CPU-heavy instance** with strong CPU/RAM/NVMe, then temporary transfer through a GMI GPU-accessible machine into Cold Storage.
2. **GMI lowest-cost GPU instance** used primarily for host CPU preprocessing and direct Cold Storage access.

Choose only after real GMI host CPU/RAM/NVMe specs and pricing are available. The pipeline must use S3-compatible abstractions/configuration so this choice requires no code rewrite.

## CPU target characteristics

Prefer a balanced machine roughly in the range of:

- 32–64 vCPU;
- 128–256 GB RAM if economical;
- 2–4 TB fast local NVMe;
- high network bandwidth.

Benchmark the actual preprocessing DAG rather than comparing hourly prices alone.

## Integrity

Each stored shard should be represented in a manifest with at least:

- logical path;
- byte size;
- row count or tensor shape;
- checksum (BLAKE3 or SHA-256);
- schema/version;
- generation stage;
- upstream dataset/version IDs.

Transfers are considered complete only after destination verification.
