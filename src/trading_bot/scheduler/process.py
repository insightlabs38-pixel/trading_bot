"""Fresh-process trial launching with process-group termination and log capture."""

from __future__ import annotations

import os
import re
import signal
import subprocess
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

_TRIAL_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True, slots=True)
class WorkerProcess:
    trial_id: str
    process: subprocess.Popen[bytes]
    stdout_path: Path
    stderr_path: Path


class SubprocessTrialRunner:
    """Launch every worker in a new process group; never execute trial code in-controller."""

    def __init__(self, log_root: str | Path) -> None:
        self.log_root = Path(log_root)
        self.log_root.mkdir(parents=True, exist_ok=True)

    def start(
        self,
        trial_id: str,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> WorkerProcess:
        if not _TRIAL_ID_RE.fullmatch(trial_id):
            raise ValueError("trial_id contains unsafe log-path characters")
        if not command:
            raise ValueError("worker command must not be empty")
        stdout_path = self.log_root / f"{trial_id}.stdout.log"
        stderr_path = self.log_root / f"{trial_id}.stderr.log"
        child_env = os.environ.copy()
        if env is not None:
            child_env.update(env)
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                list(command),
                cwd=str(cwd) if cwd is not None else None,
                env=child_env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
        return WorkerProcess(
            trial_id=trial_id,
            process=process,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    @staticmethod
    def wait(worker: WorkerProcess, *, timeout_seconds: float | None = None) -> int:
        return worker.process.wait(timeout=timeout_seconds)

    @staticmethod
    def terminate(worker: WorkerProcess, *, grace_seconds: float) -> int:
        if grace_seconds < 0:
            raise ValueError("termination grace must be non-negative")
        process = worker.process
        if process.poll() is not None:
            return int(process.returncode)
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return int(process.wait())
        try:
            return int(process.wait(timeout=grace_seconds))
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            return int(process.wait())
