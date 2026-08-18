# Source Layout

Implementation is intentionally deferred, but modules should grow under this structure:

```text
src/trading_bot/
  data/          # vendor adapters, security master, universe, features, labels, packing
  models/        # baseline and advanced architectures
  kernels/       # PyTorch references + Triton kernels
  training/      # trainer, checkpoints, precision/compile helpers
  evaluation/    # canonical backtester and metrics
  campaign/      # scheduler/controller/runtime estimation/recovery
  storage/       # local + S3-compatible storage abstraction/manifests
  execution/     # target-to-order logic and execution simulator
  risk/          # deterministic portfolio/live safety controls
  monitoring/    # structured telemetry, Discord events, status reports
  brokers/       # paper/live broker adapters (later stage)
```

Rules:

- no architecture-specific economic accounting;
- no secrets/constants embedded in modules;
- no direct final-holdout access from training/model modules;
- custom kernels require clear reference implementations and tests;
- live risk/broker logic remains independent from learned models.
