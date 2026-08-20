"""One-shot reconciliation of Phase 5 CPU progress into IMPLEMENTATION_PLAN.md."""

from __future__ import annotations

from pathlib import Path

PLAN = Path("IMPLEMENTATION_PLAN.md")


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one tracker occurrence: {old!r}")
    return text.replace(old, new, 1)


def main() -> None:
    text = PLAN.read_text(encoding="utf-8")
    text = _replace_once(
        text,
        "| 5. Common training framework | Not started | Yes |",
        "| 5. Common training framework | **IN PROGRESS — CPU gate passed; GPU acceptance pending** | Yes |",
    )
    text = _replace_once(
        text,
        "`uv sync --locked --group cpu`.",
        "`uv sync --locked --group training-cpu`.",
    )

    phase5_start = text.index("# 7. Phase 5 — common training framework")
    phase6_marker = "\n---\n\n# 8. Phase 6 — canonical evaluator and backtester"
    phase6_start = text.index(phase6_marker, phase5_start)
    phase5 = """# 7. Phase 5 — common training framework

## Goal

Every architecture must train through the same interfaces so architecture comparisons are meaningful.

## Model interface

- [x] Common base/protocol for models.
- [x] Standard batch structure.
- [x] Standard model output containing applicable fields such as:
  - [x] expected return;
  - [x] rank score;
  - [x] direction probability;
  - [x] volatility;
  - [x] uncertainty/quantiles.
- [x] Model parameter-count reporting.
- [x] Inference timing interface.

## Trainer

- [x] BF16 default path — CPU autocast is covered by the supported Python 3.12 CI path; GPU BF16 acceptance remains hardware-specific.
- [x] FP32/debug path.
- [ ] Optional FP8 finalist path where supported. — **GPU-DEPENDENT; intentionally deferred until compatible hardware/software is configured.**
- [x] Gradient accumulation.
- [x] Gradient clipping.
- [x] LR scheduling.
- [x] Early stopping hooks controlled by scheduler rather than hidden model logic.
- [x] NaN/Inf detection.
- [ ] GPU memory telemetry. — typed CUDA allocator telemetry is implemented in the heartbeat contract, but real CUDA measurements remain **GPU-DEPENDENT** and unvalidated.
- [x] Step/time progress heartbeat.
- [x] Deterministic debug mode.
- [x] Fast campaign mode.

## Checkpointing

Checkpoint must contain enough information for true continuation:

- [x] model state;
- [x] optimizer state;
- [x] LR scheduler state;
- [x] training cursor/step;
- [x] RNG state;
- [x] precision/scaler state where relevant — generic precision-state plumbing is persisted; CUDA-specific scaler acceptance remains a GPU follow-up;
- [x] model/training config hashes;
- [x] dataset/split IDs.

Also implement:

- [x] atomic temporary-write → verify → rename protocol;
- [x] latest/best bookkeeping;
- [x] resume validation;
- [x] checkpoint corruption detection.

## Prediction artifacts

- [x] Save validation predictions for promoted/final models.
- [x] Include timestamp, asset ID, target, prediction, relevant metadata.
- [x] Allow evaluator to rerun without retraining.

### Progress note — 2026-08-20

- Common contracts: `TrainingBatch`, `ModelOutput`, and `TradingModel` standardize architecture inputs/outputs; shared parameter-count and inference-timing interfaces are implemented.
- Trainer: BF16 is the default, FP32/debug is explicit, and the common loop covers gradient accumulation/clipping, stateful LR schedulers, external early-stop control, non-finite checks, heartbeat reporting, deterministic-debug mode, and fast-campaign mode.
- Telemetry: heartbeat plumbing includes allocated/reserved/peak CUDA allocator counters when running on CUDA and explicitly reports no GPU telemetry on CPU. The GPU-memory checklist remains unchecked until exercised on real CUDA hardware.
- Checkpointing: model/optimizer/scheduler/cursor/RNG/precision state and run identity are checksummed and atomically published; latest/best pointers, resume validation, and corruption detection are covered. A dropout regression reproduces the exact next optimizer-step parameters after restore on CPU.
- Predictions: evaluator-independent immutable Parquet + Zstd artifacts preserve exact asset/timestamp/target/prediction lineage and fail closed on tampering.
- Gate models: affine linear, nonlinear GELU/dropout MLP, and residual gated mixer models all train, checkpoint, reconstruct, resume, predict, and pass a trainer-independent saved-artifact MSE evaluation smoke through the same framework.
- Dependency isolation: the `training-cpu` group uses the official CPU-only PyTorch index for CPU CI while the production `gpu` group remains PyTorch-free so the pinned NGC image retains ownership of the CUDA stack.
- Supported environment: Python 3.12 CI resolved PyTorch 2.13.0+cpu and passed Ruff, format, strict mypy, compileall, and 252 tests with only the opt-in Phase 2 real-S3 provider gate skipped before final tracker reconciliation.
- Detailed status and remaining GPU-only acceptance items are recorded in `docs/progress/phase_05.md`.

## Gate

At least three architecturally different toy/baseline models train, checkpoint, resume, write predictions, and evaluate through the exact same framework.

**CPU FRAMEWORK GATE PASSED.** Three distinct toy architectures satisfy the CPU-verifiable gate, including true checkpoint continuation and trainer-independent saved-prediction evaluation. Phase 5 remains **IN PROGRESS** only for real CUDA memory telemetry, optional FP8, CUDA-specific precision/scaler acceptance, and representative GPU/H200 performance validation. Canonical predictive/economic/backtest metrics remain Phase 6.
"""
    text = text[:phase5_start] + phase5 + text[phase6_start:]
    PLAN.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
