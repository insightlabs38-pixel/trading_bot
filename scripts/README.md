# Scripts / Entry Points

This directory will contain thin operational entry points only. Business logic belongs under `src/`.

Expected future commands include wrappers for:

- CPU preprocessing pipeline;
- data verification/staging;
- H200 bootstrap/smoke test;
- campaign simulation/dress rehearsal;
- production H200 campaign;
- storage sync/verification;
- final report generation;
- shadow/paper trading processes.

The desired end state is that paid compute starts with a small number of tested commands rather than interactive environment setup or last-minute script editing.
