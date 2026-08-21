"""Infrastructure-failure circuit breaker and deterministic health gates."""

from __future__ import annotations

from collections import deque
from enum import StrEnum

from trading_bot.recovery.policy import RecoveryPolicy
from trading_bot.recovery.types import FailureClassification, GateResult


class CircuitState(StrEnum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


_REQUIRED_HEALTH_CHECKS = ("disk", "dataset", "storage", "gpu_smoke")


class CircuitBreaker:
    """Pause launches after clustered infrastructure-like failures."""

    def __init__(self, policy: RecoveryPolicy) -> None:
        self.policy = policy
        self.state = CircuitState.CLOSED
        self._failures: deque[float] = deque()
        self._opened_at: float | None = None

    def record_failure(self, classification: FailureClassification, *, now: float) -> bool:
        if not classification.infrastructure_like:
            return False
        self._prune(now)
        self._failures.append(now)
        if len(self._failures) >= self.policy.circuit_failure_threshold:
            self.state = CircuitState.OPEN
            self._opened_at = now
            return True
        return False

    def can_launch(self, *, now: float) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            assert self._opened_at is not None
            if now - self._opened_at >= self.policy.circuit_cooldown_seconds:
                self.state = CircuitState.HALF_OPEN
            return False
        return False

    def apply_health_gate(self, results: tuple[GateResult, ...]) -> bool:
        by_name = {result.name: result for result in results}
        if set(by_name) != set(_REQUIRED_HEALTH_CHECKS):
            return False
        if not all(by_name[name].passed for name in _REQUIRED_HEALTH_CHECKS):
            self.state = CircuitState.OPEN
            return False
        self.state = CircuitState.CLOSED
        self._opened_at = None
        self._failures.clear()
        return True

    def _prune(self, now: float) -> None:
        cutoff = now - self.policy.circuit_window_seconds
        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()
