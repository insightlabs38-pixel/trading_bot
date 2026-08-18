# Phase 2 Progress — Storage and Artifact Primitives

Last updated: **2026-08-18**

Status: **BLOCKED**

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

## Sandbox test environment

A dedicated test virtual environment exists at `/mnt/data/trading_bot_test_venv`.

The sandbox is itself hosted inside a managed Python environment, so the isolated venv uses a
sandbox-only `.pth` bridge to the harness's preinstalled site-packages. This permits repeatable
pytest execution without downloading dependencies.

Attempts to upgrade the venv were made on 2026-08-18:

- Python 3.12 is not installed anywhere in the sandbox.
- `uv python install 3.12` cannot resolve/download its Python distribution because outbound DNS
  is blocked.
- `pip install ruff mypy` likewise cannot reach a package index.
- no useful cached wheels/interpreters were found for the missing tools.

This does **not** change the repository dependency policy and does not resolve the Phase 1 target-
environment blocker; the venv still uses Python 3.13.5.

## Validation performed

Combined Phase 2 storage suite:

```text
34 passed
```

Also passed:

- `python -m compileall` for storage implementation/tests;
- repository 100-character line-length check for all new Python files;
- Git blob-hash comparisons confirming feature-branch source matches the exact venv-tested files.

Validated failure/recovery behavior includes checksum mismatch without publication, atomic local
replacement, bounded transient S3 retry, retry after partial upload-stream consumption, temporary
S3-key cleanup, multipart cleanup/fallback, artifact tamper and size-mismatch detection, invalid
manifest rejection, manifest-last publication, and interrupted bulk upload followed by restart
without retransferring the already verified object.

## Phase 2 test checklist status

- [x] Local backend unit tests.
- [ ] **BLOCKED** — S3 integration tests against a real test bucket or S3-compatible emulator.
- [x] Interrupted upload recovery/resume test at the bulk-transfer level.
- [x] Checksum mismatch detection test.
- [x] Manifest verification test.

## Gate

The complete local functional gate passes: a generated artifact can be published to durable local
storage with a manifest, the source can be deleted, the bytes can be restored through the resumable
bulk-transfer path, and the restored/stored artifact can be checksum/manifest verified without
manual steps.

**BLOCKED — provider integration only:** repeat the same round-trip against GMI Cold Storage and,
if used, an external S3-compatible staging provider before Phase 2 is declared fully complete.
