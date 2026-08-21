# Configuration Layout

Current structure:

```text
configs/
  data/          # source/universe/features/labels/splits
  models/        # architecture-specific config material where needed
  campaigns/     # campaign registry/search spaces, budgets, deadline policies
  evaluation/    # frozen metrics/gates/cost/latency stress assumptions
  storage/       # endpoint-agnostic S3/local settings (no secrets)
  paper/         # shadow/paper acceptance and fault-test settings
  live/          # risk/execution settings once approved
```

The Phase 10 frozen experiment/search contract lives at
`configs/campaigns/h200_tournament_v1.yaml`. It is strictly validated and hashed by
`trading_bot.campaign` and can be enumerated without importing model/PyTorch code.
Changing a search range, family pool, objective selection, canonical model preset,
seed policy, or rung budget must therefore be a visible version-controlled config
change rather than an implicit Python edit.

Configuration files should be complete enough that a trial can be reproduced from config + Git SHA + dataset manifest.

Do not store credentials, webhook URLs, broker tokens, or final-holdout access secrets here.
