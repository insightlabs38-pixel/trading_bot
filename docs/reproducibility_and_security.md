# Reproducibility and Security Plan

Status: **BASELINE**

## Reproducibility modes

### Deterministic debug mode

Use fixed seeds and deterministic algorithms where feasible for unit tests, debugging, regression tests, and minimal reproductions. This mode may sacrifice speed.

### Campaign-fast mode

Allow validated high-performance kernels/algorithms that may not be bitwise deterministic, while recording enough environment/RNG/version information to reproduce the statistical experiment as closely as practical.

Complete bit-for-bit reproducibility across GPU driver/framework/hardware revisions is not assumed.

## Run manifest

Every trial should automatically write a manifest containing at least:

- trial ID / parent trial ID;
- Git commit SHA;
- Docker image digest/tag;
- Python/PyTorch/CUDA/Triton/Transformer Engine versions where relevant;
- GPU model/driver/memory;
- full frozen config hash and file;
- dataset version/hash;
- split IDs;
- seed and RNG-state metadata;
- precision/compile mode;
- model parameter count;
- start/end timestamps/status;
- checkpoint and prediction artifact IDs.

## Dataset lineage

Every processed dataset version requires:

- vendor/source snapshot identifiers;
- raw-file checksums/manifests;
- security-master/corporate-action versions;
- transformation/config version;
- feature schema;
- label schema;
- point-in-time universe definition/version;
- train/validation/final-holdout boundaries;
- packed-training representation version.

Raw data remains immutable.

## Campaign change control

At paid campaign start, tag/freeze the repository and campaign config.

If a bug fix changes behavior:

1. commit the patch;
2. record new Git SHA/image if rebuilt;
3. create a child/new trial;
4. preserve the old failed/prior trial record.

Do not silently replace code under an existing trial ID.

## Protected final holdout

Implementation should make accidental access difficult, not merely discouraged. The final-holdout partition should have a distinct split ID/path and should not be mounted/available to normal architecture-search jobs if practical.

Access events should be logged. Final holdout is evaluated only after the complete system is frozen.

## Secrets

Never commit or bake into images:

- `.env`;
- market-data API keys;
- DeepSeek/other AI API keys;
- GMI/S3 credentials;
- Discord webhook URLs;
- broker API keys/tokens;
- SSH/private keys.

Inject secrets at runtime through environment/secrets mechanisms. Keep secret-bearing config out of logs.

## AI debugging privacy/safety

Before any external AI debugging request:

- select only relevant source/config/log fragments;
- sanitize credential patterns;
- exclude `.env` and storage/broker configs containing secrets;
- prefer synthetic/minimal tensors over licensed market data;
- do not send final-holdout results unless a deliberate human-approved use case exists;
- record request/response hashes and patch results for auditability.

## Storage integrity

All important artifacts use checksums and verification before local deletion. Object upload initiation is not proof of durability.

Checkpoints use atomic local creation and verified remote sync.

## Dependency/runtime pinning

- Pin CPU dependencies with a lockfile.
- Pin the validated GPU base image/tag/digest before campaign day.
- Avoid inadvertently pip-installing replacements for the NGC-provided PyTorch/CUDA/Triton stack.
- Record actual resolved package versions in run manifests.

## Licensed data

Do not commit raw or derived licensed market datasets to GitHub. Repository fixtures/tests use tiny synthetic or clearly redistributable samples only.

## Live trading security

Production deployment should use least-privilege broker/API credentials where supported, strict separation between research and live secrets, encrypted secret storage, and explicit kill/revoke procedures. Detailed broker security is finalized when the broker is selected.
