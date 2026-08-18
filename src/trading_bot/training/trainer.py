"""Common architecture-agnostic training loop with deterministic recovery hooks."""

from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Literal

import numpy as np
import torch
from torch import Tensor, nn

from trading_bot.training.contracts import ModelBatch, ModelOutput


class TrainingError(RuntimeError):
    """Base error for common-trainer failures."""


class NonFiniteTrainingError(TrainingError):
    """Raised when loss or gradients become NaN/Inf."""


class UnsupportedPrecisionError(TrainingError):
    """Raised when a requested precision path is unavailable."""


@dataclass(frozen=True, slots=True)
class TrainerSettings:
    max_steps: int
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float | None = 1.0
    precision: Literal["fp32", "bf16", "fp8"] = "bf16"
    deterministic: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when set")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True, slots=True)
class TrainingHeartbeat:
    step: int
    micro_steps: int
    loss: float
    learning_rate: float
    device_type: str
    gpu_memory_allocated_bytes: int
    gpu_memory_reserved_bytes: int


@dataclass(frozen=True, slots=True)
class TrainingResult:
    steps_completed: int
    micro_steps_completed: int
    stopped_early: bool
    final_loss: float


LossFunction = Callable[[ModelOutput, ModelBatch], Tensor]
HeartbeatCallback = Callable[[TrainingHeartbeat], None]
StopCallback = Callable[[TrainingHeartbeat], bool]


def configure_reproducibility(*, seed: int, deterministic: bool) -> None:
    """Configure process RNGs for deterministic debug mode or seeded fast mode."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: Iterable[ModelBatch],
    loss_fn: LossFunction,
    *,
    settings: TrainerSettings,
    lr_scheduler: object | None = None,
    heartbeat: HeartbeatCallback | None = None,
    should_stop: StopCallback | None = None,
) -> TrainingResult:
    """Train through one shared loop; early-stop decisions remain external to model logic."""
    configure_reproducibility(seed=settings.seed, deterministic=settings.deterministic)
    if settings.precision == "fp8":
        raise UnsupportedPrecisionError(
            "fp8 training requires the validated GPU/Transformer Engine environment"
        )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = _repeat_batches(batches)
    step = 0
    micro_steps = 0
    stopped_early = False
    final_loss = float("nan")
    device = _model_device(model)

    while step < settings.max_steps:
        accumulated_loss = 0.0
        for _ in range(settings.gradient_accumulation_steps):
            batch = next(iterator).to(device)
            with _autocast_context(device, settings.precision):
                output = model(batch)
                if not isinstance(output, ModelOutput):
                    raise TrainingError("models must return ModelOutput")
                output.validate(batch.batch_size)
                loss = loss_fn(output, batch)
            if loss.ndim != 0:
                raise TrainingError("loss function must return a scalar tensor")
            if not torch.isfinite(loss):
                raise NonFiniteTrainingError(f"non-finite loss at optimizer step {step}")
            scaled_loss = loss / settings.gradient_accumulation_steps
            scaled_loss.backward()
            accumulated_loss += float(loss.detach().cpu())
            micro_steps += 1

        _assert_finite_gradients(model, step)
        if settings.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
            _assert_finite_gradients(model, step)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if lr_scheduler is not None:
            scheduler_step = getattr(lr_scheduler, "step", None)
            if not callable(scheduler_step):
                raise TrainingError("lr_scheduler must expose a callable step()")
            scheduler_step()
        step += 1
        final_loss = accumulated_loss / settings.gradient_accumulation_steps
        event = TrainingHeartbeat(
            step=step,
            micro_steps=micro_steps,
            loss=final_loss,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            device_type=device.type,
            gpu_memory_allocated_bytes=_gpu_memory_allocated(device),
            gpu_memory_reserved_bytes=_gpu_memory_reserved(device),
        )
        if heartbeat is not None:
            heartbeat(event)
        if should_stop is not None and should_stop(event):
            stopped_early = True
            break

    return TrainingResult(
        steps_completed=step,
        micro_steps_completed=micro_steps,
        stopped_early=stopped_early,
        final_loss=final_loss,
    )


def _repeat_batches(batches: Iterable[ModelBatch]) -> Iterator[ModelBatch]:
    cached = tuple(batches)
    if not cached:
        raise TrainingError("at least one training batch is required")
    while True:
        yield from cached


def _autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    raise UnsupportedPrecisionError(f"unsupported precision {precision!r}")


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _assert_finite_gradients(model: nn.Module, step: int) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.all(torch.isfinite(parameter.grad)):
            raise NonFiniteTrainingError(f"non-finite gradient in {name} at optimizer step {step}")


def _gpu_memory_allocated(device: torch.device) -> int:
    return 0 if device.type != "cuda" else int(torch.cuda.memory_allocated(device))


def _gpu_memory_reserved(device: torch.device) -> int:
    return 0 if device.type != "cuda" else int(torch.cuda.memory_reserved(device))
