from __future__ import annotations

from pathlib import Path


def replace_exact(path_text: str, old: str, new: str) -> None:
    path = Path(path_text)
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"expected reconciliation text not found in {path_text}: {old!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")


plan = "IMPLEMENTATION_PLAN.md"
replace_exact(
    plan,
    "The Phase 3 and Phase 4 master statuses/checklists had fallen behind the repository. They are reconciled below against the code and detailed progress records on `main` through commit `c2089851bcf77b6d988b67467831ca2e8e05c279`, plus the current foundation-hardening branch. Provider-independent Phase 3/4 work is substantially implemented; their production gates remain blocked by real provider/data/hardware dependencies.\n\nCPU-only GitHub Actions verification is being added on standard `ubuntu-latest` hosted runners using the pinned Python 3.12 target. GPU/Triton/H200 checks are intentionally excluded until compatible GPU infrastructure is available.",
    "The Phase 0–4 tracker is reconciled below against the current foundation-hardening branch and its supported-runtime CI evidence. Phase 1 is complete; Phase 2 is blocked only on real S3/provider validation; the provider-independent Phase 3/4 CPU implementation is substantially complete, including production exchange-calendar support and the Parquet + Zstd research representation. Remaining Phase 3/4 production gates require external provider choices/data, frozen production methodology/dates, finalized production-dataset validation, or H200 benchmarking.\n\nCPU-only GitHub Actions verification now runs on one standard `ubuntu-latest` hosted runner using Python 3.12 and the committed `uv.lock`. The permanent read-only gate runs Ruff, Ruff format checking, strict mypy, `compileall`, and the full pytest suite. GPU/Triton/H200 checks remain intentionally excluded until compatible GPU infrastructure is available.",
)
replace_exact(
    plan,
    "| 1. Project/config foundations | **IN PROGRESS — Python 3.12 CI verification being added** | Yes |",
    "| 1. Project/config foundations | **COMPLETE** | Yes |",
)
replace_exact(
    plan,
    "| 3. CPU data pipeline | **BLOCKED — external/provider/production-format gates only** | Yes |",
    "| 3. CPU data pipeline | **BLOCKED — provider/frozen-methodology/H200 gates only** | Yes |",
)
replace_exact(
    plan,
    "- [ ] Commit a dependency lock once the resolved CPU/GPU environment strategy is finalized.",
    "- [x] Commit a Python 3.12-resolved dependency lock for the CPU verification environment.",
)
replace_exact(
    plan,
    "- Completed: Python project/configuration/common-metadata implementation and a reusable `scripts/verify_cpu.sh` verification gate.\n- Added: `.github/workflows/cpu-ci.yml` using one standard `ubuntu-latest` runner, Python 3.12, the CPU dependency group, Ruff, Ruff format checking, mypy, `compileall`, and the full pytest suite.\n- Cost control: one job only, 20-minute timeout, and concurrency cancellation to avoid wasting hosted-runner minutes on superseded commits.\n- Previous sandbox evidence remains useful but is not the target-environment gate because it used Python 3.13.\n- Remaining gate: obtain a green Python 3.12 GitHub Actions run from the repository branch/PR.\n- Remaining reproducibility gap: a committed `uv.lock` is still desirable once the intended resolved environment policy is frozen.",
    "- Completed: Python project/configuration/common-metadata implementation and a reusable `scripts/verify_cpu.sh` verification gate.\n- CI: `.github/workflows/cpu-ci.yml` uses one standard `ubuntu-latest` runner, Python 3.12, the locked CPU dependency group, Ruff, Ruff format checking, strict mypy, `compileall`, and the full pytest suite.\n- Reproducibility: a Python 3.12-resolved `uv.lock` is committed and CI verifies it with `uv sync --locked --group cpu`.\n- Cost control: one job only, 20-minute timeout, concurrency cancellation, no GPU/larger runner, and read-only repository permissions.\n- Supported-environment verification is green; later Phase 3 columnar additions increased the authoritative full-suite result to 241 passed with only the opt-in real S3 provider gate skipped.",
)
replace_exact(
    plan,
    "**IN PROGRESS — target-environment CI verification.** The functional gate already passes in sandbox mirrors. The new repository CI is intended to close the supported Python 3.12/Ruff/mypy/full-test confirmation.",
    "**PASSED.** Python 3.12 CPU CI is green, strict lint/type/test verification passes, and the dependency lock is committed.",
)
replace_exact(
    plan,
    "- [ ] Research representation: Parquet + Zstd. — **BLOCKED until the production columnar dependency/format path is validated.**",
    "- [x] Research representation: Parquet + Zstd with deterministic ordering, exact timestamps, float32 feature/target columns, semantic metadata, checksummed manifests, and fail-closed validation.",
)
replace_exact(
    plan,
    "- The Phase 3 checklist is now reconciled with the implemented reference pipeline and the detailed `docs/progress/phase_03.md` record.\n- New hardening: production `exchange_calendars` support resolves actual XNYS holidays and early closes and can drive per-date resampling session lengths.\n- New hardening: packed dataset metadata now has an independently verified SHA-256 sidecar; loader validation is stricter for metadata dimensions, names, file records, dtypes, and checksums.\n- Remaining blockers are external or production-freeze items: concrete vendor adapters/credentials, final universe methodology, final split dates, Parquet/Zstd production representation, and H200 loader benchmarking.",
    "- The Phase 3 checklist is reconciled with the implemented reference pipeline and `docs/progress/phase_03.md`.\n- Production `exchange_calendars` support resolves actual XNYS holidays and early closes and drives per-date resampling session lengths.\n- The NumPy memmap pack verifies array and semantic metadata integrity with SHA-256 sidecars.\n- The Parquet + Zstd research representation is implemented and CPU-CI validated through PyArrow, Polars, and DuckDB with deterministic ordering, exact timestamps, semantic metadata, and checksum/tamper detection.\n- The supported Python 3.12 CPU gate passes 241 tests; the only skip is the opt-in real S3 provider test.\n- Remaining blockers are external/frozen/hardware items: concrete vendor adapters/credentials, final universe methodology, final split dates, and H200 loader-format benchmarking.",
)
replace_exact(
    plan,
    "**REFERENCE/SYNTHETIC GATE PASSED IN PRIOR SANDBOX AUDITS. PRODUCTION PHASE BLOCKED on the external/finalization items above.**",
    "**REFERENCE/SYNTHETIC CPU GATE PASSED IN PYTHON 3.12 CI. PRODUCTION PHASE BLOCKED only on the external/finalization/H200 items above.**",
)
replace_exact(
    plan,
    "- The Phase 4 checklist is reconciled with the implemented leakage/audit suite and `docs/progress/phase_04.md`.\n- The production calendar dependency gap is now addressed in code for XNYS-style session/early-close validation; it still needs to be exercised on the finalized provider-derived production dataset.\n- The production Phase 4 gate remains blocked because the final provider dataset, frozen universe/split definitions, and final columnar representation do not yet exist.",
    "- The Phase 4 checklist is reconciled with the implemented leakage/audit suite and `docs/progress/phase_04.md`.\n- Exchange-calendar holiday/early-close behavior and the Parquet + Zstd research representation are now exercised by the supported Python 3.12 CPU environment.\n- The complete CPU suite passes 241 tests with only the opt-in real S3 provider test skipped.\n- The production Phase 4 gate remains blocked because the final provider-derived dataset and frozen production universe/split definitions do not yet exist; the exact leakage/audit suite must be rerun on that frozen dataset and final representation.",
)

phase1 = "docs/progress/phase_01.md"
replace_exact(
    phase1,
    "mypy: success, no issues in 33 source files\npytest: 234 passed, 1 skipped in 4.22s",
    "mypy: success, no issues in 34 source files\npytest: 241 passed, 1 skipped",
)
replace_exact(
    phase1,
    "The single skipped test is the opt-in real S3 provider gate in\n`tests/integration/test_phase2_s3_provider_gate.py`; it requires a real S3-compatible test endpoint\nand credentials and therefore does not block the Phase 1 project/configuration gate.",
    "The single skipped test is the opt-in real S3 provider gate in\n`tests/integration/test_phase2_s3_provider_gate.py`; it requires a real S3-compatible test endpoint\nand credentials and therefore does not block the Phase 1 project/configuration gate. The later\nPhase 3 Parquet/Zstd additions are included in the 241-test result above.",
)

phase2 = "docs/progress/phase_02.md"
replace_exact(phase2, "Last updated: **2026-08-18**", "Last updated: **2026-08-20**")
replace_exact(phase2, "Status: **BLOCKED**", "Status: **BLOCKED — real S3/GMI integration only**")
replace_exact(
    phase2,
    "## Sandbox test environment\n\nA dedicated test virtual environment exists at `/mnt/data/trading_bot_test_venv`.\n\nThe sandbox is itself hosted inside a managed Python environment, so the isolated venv uses a\nsandbox-only `.pth` bridge to the harness's preinstalled site-packages. This permits repeatable\npytest execution without downloading dependencies.\n\nAttempts to upgrade the venv were made on 2026-08-18:\n\n- Python 3.12 is not installed anywhere in the sandbox.\n- `uv python install 3.12` cannot resolve/download its Python distribution because outbound DNS\n  is blocked.\n- `pip install ruff mypy` likewise cannot reach a package index.\n- no useful cached wheels/interpreters were found for the missing tools.\n\nThis does **not** change the repository dependency policy and does not resolve the Phase 1 target-\nenvironment blocker; the venv still uses Python 3.13.5.",
    "## Supported CPU CI environment\n\nThe public repository now runs the permanent CPU verification workflow on a standard free\n`ubuntu-latest` GitHub-hosted runner with Python 3.12. The committed `uv.lock` resolves the CPU\ndependency set, including boto3 and the storage test dependencies. Ruff, formatting, strict mypy,\n`compileall`, and the full pytest suite execute on every PR update.\n\nThe authoritative current full-suite result is **241 passed, 1 skipped**. The single skip is this\nphase's opt-in real S3 provider gate because no real GMI/test endpoint or credentials are configured\nin ordinary CI. All local/fake-client storage tests execute and pass in the supported environment.",
)
replace_exact(
    phase2,
    "Combined Phase 2 storage suite:\n\n```text\n34 passed\n```",
    "The current Python 3.12 CPU CI executes the Phase 2 storage suite as part of the complete\nrepository gate. Authoritative full-suite result:\n\n```text\n241 passed, 1 skipped\n```\n\nThe one skip is the intentionally opt-in real-provider test. Historical focused Phase 2 validation\nrecorded 34 passing storage tests before repository-wide CI became available.",
)

phase3 = "docs/progress/phase_03.md"
replace_exact(phase3, "Last updated: **2026-08-18**", "Last updated: **2026-08-20**")
replace_exact(
    phase3,
    "Status: **BLOCKED — production/provider validation only**",
    "Status: **BLOCKED — vendor/frozen-methodology/H200 gates only**",
)
replace_exact(
    phase3,
    "This file records validation detail for Phase 3. The authoritative task list remains\n`IMPLEMENTATION_PLAN.md`, whose Phase 3 checkboxes are currently stale relative to the implemented\ncode and this detailed progress record.",
    "This file records validation detail for Phase 3. The authoritative task list remains\n`IMPLEMENTATION_PLAN.md`; this record is reconciled with the current implementation and supported\nPython 3.12 CPU CI evidence.",
)
replace_exact(
    phase3,
    "- [ ] **BLOCKED — production calendar validation:** exchange holidays/early closes require the\n  intended production calendar dependency/source.",
    "- [x] Production exchange-calendar support uses `exchange_calendars`; XNYS holidays and early\n  closes are covered by CPU-CI regressions and feed exact per-session resampling specifications.",
)
replace_exact(
    phase3,
    "- [ ] **BLOCKED — production columnar/performance validation:** Polars/PyArrow/DuckDB and Python\n  3.12 cannot be installed in this sandbox.",
    "- [x] The supported Python 3.12 CPU environment includes PyArrow, Polars, and DuckDB; the\n  reference feature/packing path is exercised with the production columnar dependency set.",
)
replace_exact(
    phase3,
    "- [ ] **BLOCKED — research representation:** Parquet + Zstd cannot be implemented/validated here\n  because PyArrow/Polars cannot be installed and no Parquet engine is available.",
    "- [x] Parquet + Zstd research representation with deterministic row ordering, exact int64\n  nanosecond timestamps, float32 feature/target columns, semantic schema metadata, checksummed\n  manifests, atomic publication, and fail-closed corruption/tamper validation.",
)
replace_exact(
    phase3,
    "## Validation performed\n\nHistorical repository/sandbox validation already recorded for earlier Phase 3 increments includes:",
    "## Validation performed\n\nAuthoritative supported-environment verification now runs on Ubuntu 24.04 / Python 3.12 through\nthe permanent CPU GitHub Actions gate. Current repository-wide result:\n\n```text\nRuff: all checks passed\nFormatting: all files formatted\nmypy: success, no issues in 34 source files\npytest: 241 passed, 1 skipped\n```\n\nThe only skipped test is the opt-in real S3 provider gate. The Parquet/Zstd tests independently\nround-trip through PyArrow and verify readability through Polars and DuckDB, while corruption,\nsemantic-metadata tampering, exact timestamps, float32 representability, deterministic output, and\nZstd compression are covered by regressions.\n\nHistorical repository/sandbox validation recorded for earlier Phase 3 increments includes:",
)
replace_exact(
    phase3,
    "For this completion audit, a focused sandbox mirror exercised existing Raw validation/Security\nmaster regressions plus adversarial checks across acquisition, canonicalization, resampling,\nuniverse construction, features, labels, splits, packing, restartable stage publication, and a\nrepeated synthetic multi-asset raw → packed integration build:",
    "An earlier focused sandbox audit exercised Raw validation/Security master regressions plus\nadversarial checks across acquisition, canonicalization, resampling, universe construction,\nfeatures, labels, splits, packing, restartable stage publication, and a repeated synthetic\nmulti-asset raw → packed integration build:",
)
replace_exact(
    phase3,
    "The audit mirror also passes `python -m compileall`. The private repository is not mounted in this\nsandbox, so this **is not described as a fresh full-repository pytest run**; the new repository tests\nare committed alongside the implementation so the same adversarial and raw→packed contracts can be\nrun directly in a normal checkout/CI environment.",
    "That historical audit also passed `python -m compileall`. It is retained as focused regression\nevidence; the repository-wide Python 3.12 CI result above is now the authoritative verification.",
)
replace_exact(
    phase3,
    "No live broad-equities/execution-provider request, real exchange-calendar holiday/early-close run,\nproduction universe/split freeze, Parquet pipeline, or H200 loader benchmark is claimed as validated\nbecause those inputs/dependencies/hardware are unavailable here.",
    "No live broad-equities/execution-provider request, production universe/split freeze, or H200\nloader benchmark is claimed as validated. Exchange-calendar behavior and the Parquet/Zstd research\nrepresentation are now covered by the supported CPU environment.",
)
replace_exact(
    phase3,
    "- [ ] **BLOCKED — production exchange calendar:** holiday/early-close behavior must be validated\n  against the selected production calendar source.\n- [ ] **BLOCKED — production universe methodology:** exact cadence/thresholds require the frozen\n  production methodology and real history.\n- [ ] **BLOCKED — production columnar research representation:** Parquet + Zstd and columnar\n  performance require PyArrow/Polars or an equivalent available production dependency.\n- [ ] **BLOCKED — production split dates:** actual boundaries depend on the finalized data period.",
    "- [ ] **BLOCKED — production universe methodology:** exact cadence/thresholds require the frozen\n  production methodology and real history.\n- [ ] **BLOCKED — production split dates:** actual boundaries depend on the finalized data period.",
)
replace_exact(
    phase3,
    "All sandbox-verifiable provider-independent contracts are implemented and audited, including a\nsynthetic deterministic raw → packed composition gate and restartable manifest/success-marker\npublication primitive. The **production Phase 3 gate remains BLOCKED** only by the external items\nlisted above. Those blockers must be resolved before the production dataset is frozen or an H200\ncampaign is treated as valid.",
    "All CPU-verifiable provider-independent contracts are implemented and pass the supported Python\n3.12 CI gate, including deterministic raw → packed composition, restartable publication, real\nexchange-calendar regressions, and the Parquet/Zstd research representation. The **production\nPhase 3 gate remains BLOCKED** only by vendor selection/integration, frozen production universe and\nsplit decisions, and final H200 loader benchmarking. Those blockers must be resolved before the\nproduction dataset is frozen or an H200 campaign is treated as valid.",
)

phase4 = "docs/progress/phase_04.md"
replace_exact(phase4, "Last updated: **2026-08-18**", "Last updated: **2026-08-20**")
replace_exact(
    phase4,
    "This file records Phase 4 validation detail. `IMPLEMENTATION_PLAN.md` remains the nominal task list,\nbut its Phase 4 checkboxes are stale relative to the implemented code and this detailed progress\nrecord.",
    "This file records Phase 4 validation detail. `IMPLEMENTATION_PLAN.md` remains the authoritative\ntask list and is reconciled with this record and the supported Python 3.12 CPU CI evidence.",
)
replace_exact(
    phase4,
    "## Validation performed\n\nHistorical validation recorded before this audit included the complete Phase 2 storage suite, all\nthen-implemented Phase 3 data-contract tests, the original Phase 4 leakage suite, and dataset-audit\ntests:",
    "## Validation performed\n\nAuthoritative supported-environment verification now runs on Ubuntu 24.04 / Python 3.12 through\nthe permanent CPU GitHub Actions gate. Current repository-wide result is **241 passed, 1 skipped**\nafter Ruff, format, strict mypy, and `compileall` all pass. The one skip is the opt-in real S3\nprovider test. Exchange-calendar regressions and the Parquet/Zstd research representation are\nincluded in this gate.\n\nHistorical validation recorded before repository-wide CI included the complete Phase 2 storage\nsuite, then-implemented Phase 3 data-contract tests, the original Phase 4 leakage suite, and\ndataset-audit tests:",
)
replace_exact(
    phase4,
    "For this completion audit, the existing Phase 4 leakage/audit behavior, Phase 3 completion/raw/\nsecurity regressions, and new Phase 4 adversarial tests were exercised together in the available\nsandbox mirror:",
    "An earlier focused completion audit exercised the existing Phase 4 leakage/audit behavior, Phase\n3 completion/raw/security regressions, and new Phase 4 adversarial tests together in the sandbox\nmirror:",
)
replace_exact(
    phase4,
    "The focused mirror also passes `python -m compileall`. The private repository is not mounted in this\nsandbox, so this **is not described as a fresh full-repository pytest run**. The new regression tests\nare committed beside the implementation for a normal checkout/CI environment.",
    "That focused mirror also passed `python -m compileall`; it is retained as historical focused\nevidence. The repository-wide Python 3.12 CI result above is now the authoritative verification.",
)
replace_exact(
    phase4,
    "- [ ] **BLOCKED — production calendar validation:** session/early-close/holiday behavior must be\n  validated against the selected production exchange-calendar source.\n- [ ] **BLOCKED — frozen production universe/splits:** exact universe methodology and production\n  split dates remain external Phase 3 decisions and must be frozen before the production Phase 4\n  gate can pass.\n- [ ] **BLOCKED — production columnar representation:** the same invariants/audits must be rerun on\n  the finalized Parquet/columnar representation once its unavailable dependencies are present.",
    "- [ ] **BLOCKED — frozen production universe/splits:** exact universe methodology and production\n  split dates remain external Phase 3 decisions and must be frozen before the production Phase 4\n  gate can pass.\n- [ ] **BLOCKED — final production representation/data:** the same invariants/audits must be rerun on\n  the exact frozen provider-derived dataset and final representation used for production research.",
)
replace_exact(
    phase4,
    "All sandbox-verifiable leakage/data-contract invariants and dataset-audit primitives are implemented\nand audited. The **production Phase 4 gate remains BLOCKED** only because the finalized production\ndataset and its external Phase 3 dependencies do not yet exist in this environment.",
    "All CPU-verifiable leakage/data-contract invariants and dataset-audit primitives are implemented\nand pass the supported Python 3.12 CI gate. The **production Phase 4 gate remains BLOCKED** only\nbecause the finalized provider-derived dataset and frozen Phase 3 production decisions do not yet\nexist; the suite must be rerun on that exact frozen data/representation.",
)
