"""Device-agnostic common trainer with CPU-verifiable safety and control hooks."""

from __future__ import annotations

import contextlib
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Literal, Protocol, cast

import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer

from trading_bot.training.contracts import ModelOutput, TradingModel, TrainingBatch


class TrainerError(RuntimeError):
    """Base error for common training failures."""


class UnsupportedPrecisionError(TrainerError):
    """Raised when a requested precision path is not available on this runtime."""


class NonFiniteTrainingError(TrainerError):
    """Raised when a loss or gradient contains NaN or infinity."""


class Scheduler(Protocol):
    """Minimal stateful learning-rate scheduler boundary used by checkpoints."""

    def step(self) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...


@dataclass(frozen=True, slots=True)
class TrainerRuntimeOptions:
    """Execution-only controls that do not redefine the scientific training config."""

    mode: Literal["deterministic_debug", "fast_campaign"] = "deterministic_debug"
    heartbeat_interval_steps: int = 10
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.heartbeat_interval_steps < 1:
            raise ValueError("heartbeat_interval_steps must be positive")


@dataclass(frozen=True, slots=True)
class TrainingState:
    """Serializable cursor required for exact checkpoint continuation."""

    optimizer_step: int = 0
    micro_step: int = 0
    samples_seen: int = 0
    last_loss: float | None = None
    stopped_early: bool = False

    def __post_init__(self) -> None:
        if self.optimizer_step < 0 or self.micro_step < 0 or self.samples_seen < 0:
            raise ValueError("training cursor values must be non-negative")


@dataclass(frozen=True, slots=True)
class TrainingHeartbeat:
    """Progress payload suitable for scheduler/monitoring integration."""

    state: TrainingState
    elapsed_seconds: float
    samples_per_second: float
    learning_rate: float


LossFunction = Callable[[ModelOutput, TrainingBatch], Tensor]
StopController = Callable[[TrainingState], bool]
HeartbeatCallback = Callable[[TrainingHeartbeat], None]


class Trainer:
    """One common optimizer loop shared by all PyTorch architectures."""

    def __init__(
        self,
        *,
        model: TradingModel,
        optimizer: Optimizer,
        loss_fn: LossFunction,
        max_steps: int,
        gradient_accumulation_steps: int = 1,
        gradient_clip_norm: float | None = 1.0,
        precision: Literal["fp32", "bf16", "fp8"] = "fp32",
        seed: int = 42,
        scheduler: Scheduler | None = None,
        options: TrainerRuntimeOptions | None = None,
    ) -> None:
        if max_steps < 1 or gradient_accumulation_steps < 1:
            raise ValueError("max_steps and gradient accumulation must be positive")
        if gradient_clip_norm is not None and gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_norm = gradient_clip_norm
        self.precision = precision
        self.seed = seed
        self.scheduler = scheduler
        self.options = options or TrainerRuntimeOptions()
        self.device = torch.device(self.options.device)
        if precision == "fp8":
            raise UnsupportedPrecisionError(
                "FP8 requires a GPU-specific implementation and verification path"
            )
        self.model.to(self.device)
        _seed_process(seed)

    def fit(
        self,
        batches: Iterable[TrainingBatch],
        *,
        initial_state: TrainingState | None = None,
        should_stop: StopController | None = None,
        heartbeat: HeartbeatCallback | None = None,
    ) -> TrainingState:
        """Train until max optimizer steps, data exhaustion, or an external stop hook."""
        state = initial_state or TrainingState()
        if state.optimizer_step > self.max_steps:
            raise ValueError("initial training step exceeds max_steps")
        previous_determinism = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(self.options.mode == "deterministic_debug")
        started = perf_counter()
        pending_micro_steps = 0
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        try:
            for original_batch in batches:
                if state.optimizer_step >= self.max_steps:
                    break
                batch = original_batch.to(self.device)
                with self._autocast_context():
                    output = cast(ModelOutput, self.model(batch))
                    output.validate(batch.batch_size)
                    raw_loss = self.loss_fn(output, batch)
                self._validate_loss(raw_loss)
                loss_value = float(raw_loss.detach().float().item())
                torch.autograd.backward(raw_loss / self.gradient_accumulation_steps)
                pending_micro_steps += 1
                state = replace(
                    state,
                    micro_step=state.micro_step + 1,
                    samples_seen=state.samples_seen + batch.batch_size,
                    last_loss=loss_value,
                )
                if pending_micro_steps < self.gradient_accumulation_steps:
                    continue
                self._optimizer_step()
                pending_micro_steps = 0
                state = replace(state, optimizer_step=state.optimizer_step + 1)
                if heartbeat is not None and (
                    state.optimizer_step % self.options.heartbeat_interval_steps == 0
                ):
                    heartbeat(self._heartbeat(state, started))
                if should_stop is not None and should_stop(state):
                    state = replace(state, stopped_early=True)
                    break
            if pending_micro_steps:
                # Never apply an under-scaled partial accumulation window.
                self.optimizer.zero_grad(set_to_none=True)
            return state
        finally:
            torch.use_deterministic_algorithms(previous_determinism)

    def _autocast_context(self) -> contextlib.AbstractContextManager[Any]:
        if self.precision == "fp32":
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)

    def _validate_loss(self, loss: Tensor) -> None:
        if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
            raise NonFiniteTrainingError("loss is not a finite scalar")

    def _optimizer_step(self) -> None:
        self._verify_gradients()
        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip_norm,
                error_if_nonfinite=True,
            )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()

    def _verify_gradients(self) -> None:
        for name, parameter in self.model.named_parameters():
            gradient = parameter.grad
            if gradient is not None and not bool(torch.isfinite(gradient).all().item()):
                raise NonFiniteTrainingError(f"gradient for {name} contains NaN/Inf")

    def _heartbeat(self, state: TrainingState, started: float) -> TrainingHeartbeat:
        elapsed = perf_counter() - started
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        return TrainingHeartbeat(
            state=state,
            elapsed_seconds=elapsed,
            samples_per_second=state.samples_seen / max(elapsed, 1e-12),
            learning_rate=learning_rate,
        )


def _seed_process(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
