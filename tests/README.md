# Test Layout

Planned test groups:

```text
tests/
  unit/
  data_contracts/
  models/
  kernels/
  evaluation/
  campaign/
  integration/
  fault_injection/
  paper_trading/
```

Highest-priority tests before paid GPU time:

- timestamp/causality and split leakage checks;
- point-in-time universe behavior;
- canonical return/cost-accounting tests;
- checkpoint save/resume equivalence;
- model common-interface tests;
- Triton/reference numerical and gradient tests;
- scheduler state/restart/retry/deadline tests;
- simulated campaign fault injection;
- storage checksum/recovery tests;
- shadow/live replay consistency tests later.

Tests use synthetic/tiny redistributable fixtures, not licensed production datasets.
