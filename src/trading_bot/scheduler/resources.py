"""Resource-slot policy for GPU trial exclusivity and concurrent CPU evaluation."""

from __future__ import annotations

from trading_bot.scheduler.policy import SchedulerResourcePolicy


class ResourceAllocator:
    """In-memory launch guard; durable usage samples are recorded separately in SQLite."""

    def __init__(self, policy: SchedulerResourcePolicy) -> None:
        self.policy = policy
        self._gpu_trials: dict[str, bool] = {}
        self._cpu_evaluators: set[str] = set()
        self._tiny_concurrency_calibrated = False

    @property
    def active_gpu_trials(self) -> tuple[str, ...]:
        return tuple(sorted(self._gpu_trials))

    @property
    def active_cpu_evaluators(self) -> tuple[str, ...]:
        return tuple(sorted(self._cpu_evaluators))

    def record_tiny_concurrency_calibration(self, throughput_gain: float) -> None:
        if throughput_gain <= 0:
            raise ValueError("throughput gain must be positive")
        self._tiny_concurrency_calibrated = (
            self.policy.allow_concurrent_tiny_trials_after_calibration
            and throughput_gain >= self.policy.minimum_tiny_trial_throughput_gain
        )

    def can_start_gpu_trial(self, *, is_tiny: bool) -> bool:
        if not self._gpu_trials:
            return True
        if self.policy.exclusive_gpu_trials:
            return False
        if not is_tiny or not self._tiny_concurrency_calibrated:
            return False
        if not all(self._gpu_trials.values()):
            return False
        return len(self._gpu_trials) < self.policy.max_gpu_trials

    def start_gpu_trial(self, trial_id: str, *, is_tiny: bool) -> None:
        if trial_id in self._gpu_trials:
            raise ValueError(f"GPU trial {trial_id!r} already owns a slot")
        if not self.can_start_gpu_trial(is_tiny=is_tiny):
            raise RuntimeError("GPU launch would violate resource policy")
        self._gpu_trials[trial_id] = is_tiny

    def finish_gpu_trial(self, trial_id: str) -> None:
        if self._gpu_trials.pop(trial_id, None) is None:
            raise KeyError(f"GPU trial {trial_id!r} does not own a slot")

    def can_start_cpu_evaluator(self) -> bool:
        return len(self._cpu_evaluators) < self.policy.max_cpu_evaluators

    def start_cpu_evaluator(self, evaluator_id: str) -> None:
        if evaluator_id in self._cpu_evaluators:
            raise ValueError(f"evaluator {evaluator_id!r} already owns a slot")
        if not self.can_start_cpu_evaluator():
            raise RuntimeError("CPU evaluator capacity exhausted")
        self._cpu_evaluators.add(evaluator_id)

    def finish_cpu_evaluator(self, evaluator_id: str) -> None:
        if evaluator_id not in self._cpu_evaluators:
            raise KeyError(f"evaluator {evaluator_id!r} does not own a slot")
        self._cpu_evaluators.remove(evaluator_id)
