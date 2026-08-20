# Phase 2 Progress — Storage and Artifact Primitives

Last updated: **2026-08-20**

Status: **BLOCKED — real S3/GMI integration only**

This file records validation detail for Phase 2. The authoritative task list remains
`IMPLEMENTATION_PLAN.md`.

## Status convention

- `[x]` — complete and acceptance criteria met.
- `[ ]` — not complete.
- `IN PROGRESS` — work has started but the acceptance gate is not met.
- `BLOCKED` — cannot proceed until the stated dependency/decision is resolved.
- `OPTIONAL` — useful but not on the critical path.

## Storage abstraction

- [x] Common `StorageBackend` protocol and portable object metadata.
- [x] Local filesystem backend.
- [x] Local and S3 operations for list, exists, upload, multipart upload, download, copy,
  delete, metadata/head, and checksum verification.
- [x] Safe relative object-key normalization with path-escape rejection.
- [x] Atomic publication through unique temporary local paths/S3 keys.
- [x] S3 multipart upload with abort-on-failure cleanup and automatic threshold selection.
- [x] Bounded exponential retry policy for transient S3/network failures.
- [x] Explicit connect/read timeout policy applied to the boto3 client.
- [x] Config-to-backend factory; the same S3 implementation can represent independent durable
  and staging buckets/prefixes.
- [x] SHA-256 metadata/stream verification where practical.

**BLOCKED — real S3/GMI integration validation:** the sandbox has no reachable S3-compatible
bucket/emulator or GMI Cold Storage credentials. The S3 backend is implemented and unit-tested
against an in-memory client, and boto3 client timeout construction is tested without network I/O,
but GMI/external-provider compatibility must be confirmed against a real endpoint before those
provider-specific tracker items are checked.

## Artifact manifests

- [x] Immutable versioned artifact-manifest schema.
- [x] Logical artifact path/key, byte size, SHA-256, artifact schema/version, and UTC creation time.
- [x] Producer Git SHA and producer config SHA-256 provenance fields.
- [x] Optional row-count, tensor-shape, generation-stage, upstream-ID, and JSON metadata lineage.
- [x] Stable canonical JSON plus a manifest-document SHA-256.
- [x] Backend-independent manifest build/write/load/verify helpers.
- [x] Manifest publication occurs only after artifact checksum/size verification succeeds.
- [x] Local verification command: `python -m trading_bot.storage.manifests verify-local ...`.

## Bulk transfer

- [x] Backend-native bulk-transfer implementation used as the plan's `rclone`, `s5cmd`, or
  equivalent path when external transfer binaries are unavailable.
- [x] Upload and download batches over the common storage protocol.
- [x] Durable versioned JSON journal for object-level resume after interruption/restart.
- [x] Verified existing objects/files are adopted/skipped rather than retransferred.
- [x] Plan-hash protection prevents accidentally reusing a journal for a different transfer set.
- [x] Automatic multipart behavior is inherited from the S3 backend for large uploads.
- [x] Per-run JSONL throughput/counter statistics, including bytes and MiB/s.
- [x] Journal writes are atomic and fsynced; ordinary failures record partial statistics.

The sandbox does not provide `rclone` or `s5cmd`, and cannot download them because outbound DNS
and package/tool downloads are blocked. The backend-native implementation therefore satisfies the
allowed "or equivalent" implementation path. Provider-scale performance remains an external
benchmark item.

## Real-provider gate harness

- [x] Added `tests/integration/test_phase2_s3_provider_gate.py` as the opt-in real-provider gate.
- [x] Uses a unique per-run provider prefix and deletes the generated remote artifact/manifest.
- [x] Exercises artifact publication, local-source deletion, resumable restore, byte equality,
  checksum verification, and manifest verification against the actual S3-compatible endpoint.
- [x] Requires explicit `TRADING_BOT_S3_TEST_ENABLED=1` plus bucket and endpoint variables so the
  integration test cannot accidentally mutate a default AWS account.
- [x] Supports optional region, access key, secret key, session token, and test-prefix variables.

Required activation variables:

```text
TRADING_BOT_S3_TEST_ENABLED=1
TRADING_BOT_S3_TEST_BUCKET=<test bucket>
TRADING_BOT_S3_TEST_ENDPOINT_URL=<S3-compatible endpoint>
```

Optional variables:

```text
TRADING_BOT_S3_TEST_REGION
TRADING_BOT_S3_TEST_ACCESS_KEY
TRADING_BOT_S3_TEST_SECRET_KEY
TRADING_BOT_S3_TEST_SESSION_TOKEN
TRADING_BOT_S3_TEST_PREFIX
```

The harness is implemented now; the provider-specific checkbox remains blocked until this test is
actually run successfully against GMI Cold Storage or the selected S3-compatible test provider.

## Supported CPU CI environment

The public repository now runs the permanent CPU verification workflow on a standard free
`ubuntu-latest` GitHub-hosted runner with Python 3.12. The committed `uv.lock` resolves the CPU
dependency set, including boto3 and the storage test dependencies. Ruff, formatting, strict mypy,
`compileall`, and the full pytest suite execute on every PR update.

The authoritative current full-suite result is **241 passed, 1 skipped**. The single skip is this
phase's opt-in real S3 provider gate because no real GMI/test endpoint or credentials are configured
in ordinary CI. All local/fake-client storage tests execute and pass in the supported environment.

## Validation performed

The current Python 3.12 CPU CI executes the Phase 2 storage suite as part of the complete
repository gate. Authoritative full-suite result:

```text
241 passed, 1 skipped
```

The one skip is the intentionally opt-in real-provider test. Historical focused Phase 2 validation
recorded 34 passing storage tests before repository-wide CI became available.

Also passed:

- `python -m compileall` for storage implementation/tests;
- repository 100-character line-length check for all new Python files;
- Git blob-hash comparisons confirming feature-branch source matches the exact venv-tested files.

For the new real-provider gate harness in the current sandbox:

- `python -m py_compile tests/integration/test_phase2_s3_provider_gate.py` passes;
- the 100-character line-length check passes;
- with no provider opt-in variables configured, pytest safely reports the module as skipped;
- live provider execution is intentionally **BLOCKED** because this sandbox has no GMI/S3 endpoint
  credentials or reachable provider endpoint.

Validated failure/recovery behavior includes checksum mismatch without publication, atomic local
replacement, bounded transient S3 retry, retry after partial upload-stream consumption, temporary
S3-key cleanup, multipart cleanup/fallback, artifact tamper and size-mismatch detection, invalid
manifest rejection, manifest-last publication, and interrupted bulk upload followed by restart
without retransferring the already verified object.

## Phase 2 test checklist status

- [x] Local backend unit tests.
- [ ] **BLOCKED** — real S3 integration test harness is implemented, but successful execution
  still requires a GMI/test S3 endpoint and credentials.
- [x] Interrupted upload recovery/resume test at the bulk-transfer level.
- [x] Checksum mismatch detection test.
- [x] Manifest verification test.

## Gate

The complete local functional gate passes: a generated artifact can be published to durable local
storage with a manifest, the source can be deleted, the bytes can be restored through the resumable
bulk-transfer path, and the restored/stored artifact can be checksum/manifest verified without
manual steps.

**BLOCKED — provider integration only:** run
`tests/integration/test_phase2_s3_provider_gate.py` successfully against GMI Cold Storage and, if
used, the selected external S3-compatible staging provider before Phase 2 is declared fully
complete.
