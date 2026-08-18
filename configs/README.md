# Configuration Layout

Planned structure:

```text
configs/
  data/          # source/universe/features/labels/splits
  models/        # architecture defaults and search spaces
  campaigns/     # 48h scheduler phases, budgets, deadline policies
  evaluation/    # frozen metrics/gates/cost/latency stress assumptions
  storage/       # endpoint-agnostic S3/local settings (no secrets)
  paper/         # shadow/paper acceptance and fault-test settings
  live/          # risk/execution settings once approved
```

Configuration files should be complete enough that a trial can be reproduced from config + Git SHA + dataset manifest.

Do not store credentials, webhook URLs, broker tokens, or final-holdout access secrets here.
