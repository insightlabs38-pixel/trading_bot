# Architecture Decision Records

Use this directory for material design changes after the initial plan.

Suggested format:

```text
0001-short-title.md
0002-short-title.md
...
```

Each ADR should contain:

- **Status:** Proposed / Accepted / Superseded / Rejected
- **Date**
- **Context**
- **Decision**
- **Alternatives considered**
- **Consequences / trade-offs**
- **Impact on existing datasets/campaigns/reproducibility**

Examples of decisions worth an ADR:

- changing primary data vendor;
- changing the point-in-time universe definition;
- moving from Compose to another orchestration model;
- changing the canonical evaluation hierarchy;
- adding a new production broker;
- changing the default training precision/runtime stack;
- introducing multi-agent execution research into the main system.

Do not rewrite old ADRs to pretend a prior decision never existed; supersede them so the project history stays auditable.
