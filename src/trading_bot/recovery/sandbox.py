"""Isolated Git worktree repair application and deterministic validation gates."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import time
from pathlib import Path

from trading_bot.recovery.repair import ProtectedFilePolicy
from trading_bot.recovery.types import GateResult, RepairProposal, RepairValidationResult


class RepairSandbox:
    """A detached Git worktree that prevents AI proposals from editing the live checkout."""

    def __init__(self, *, repository_root: str | Path, worktree_path: str | Path) -> None:
        self.repository_root = Path(repository_root).resolve()
        self.worktree_path = Path(worktree_path).resolve()

    @classmethod
    def create(
        cls,
        *,
        repository_root: str | Path,
        worktree_path: str | Path,
        base_ref: str = "HEAD",
    ) -> RepairSandbox:
        repository = Path(repository_root).resolve()
        worktree = Path(worktree_path).resolve()
        if worktree.exists():
            raise FileExistsError(f"repair worktree already exists: {worktree}")
        subprocess.run(
            ["git", "-C", str(repository), "worktree", "add", "--detach", str(worktree), base_ref],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return cls(repository_root=repository, worktree_path=worktree)

    def apply(self, proposal: RepairProposal, policy: ProtectedFilePolicy) -> tuple[str, ...]:
        paths = tuple(change.path for change in proposal.changes)
        policy.require_allowed(paths)
        for change in proposal.changes:
            destination = self._safe_path(change.path)
            if not destination.is_file():
                raise FileNotFoundError(f"repair target does not exist: {change.path}")
            current = destination.read_bytes()
            actual_sha256 = hashlib.sha256(current).hexdigest()
            if actual_sha256 != change.expected_sha256:
                raise ValueError(
                    f"repair target changed before patch application: {change.path}; "
                    f"expected {change.expected_sha256}, got {actual_sha256}"
                )
            destination.write_text(change.replacement_text, encoding="utf-8")
        return paths

    def diff(self) -> str:
        result = subprocess.run(
            ["git", "-C", str(self.worktree_path), "diff", "--no-ext-diff", "--"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout

    def close(self) -> None:
        subprocess.run(
            ["git", "-C", str(self.repository_root), "worktree", "remove", "--force", str(self.worktree_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if self.worktree_path.exists():
            shutil.rmtree(self.worktree_path)

    def _safe_path(self, relative: str) -> Path:
        destination = (self.worktree_path / relative).resolve()
        if self.worktree_path not in destination.parents:
            raise PermissionError(f"repair target escapes sandbox: {relative}")
        return destination


def run_command_gate(
    name: str,
    commands: tuple[tuple[str, ...], ...],
    *,
    cwd: str | Path,
    timeout_seconds: float,
) -> GateResult:
    started = time.perf_counter()
    details: list[str] = []
    for command in commands:
        try:
            result = subprocess.run(
                list(command),
                cwd=cwd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return GateResult(
                name=name,
                passed=False,
                detail=f"command timed out: {' '.join(command)}",
                duration_seconds=time.perf_counter() - started,
            )
        details.append(f"{' '.join(command)} -> {result.returncode}")
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or details[-1]
            return GateResult(
                name=name,
                passed=False,
                detail=detail[-4000:],
                duration_seconds=time.perf_counter() - started,
            )
    return GateResult(
        name=name,
        passed=True,
        detail="; ".join(details) if details else "no commands configured",
        duration_seconds=time.perf_counter() - started,
    )


def validate_repair(
    sandbox: RepairSandbox,
    *,
    static_commands: tuple[tuple[str, ...], ...],
    unit_commands: tuple[tuple[str, ...], ...],
    regression_commands: tuple[tuple[str, ...], ...],
    gpu_smoke_gate: GateResult,
    timeout_seconds: float = 120.0,
) -> RepairValidationResult:
    """Run deterministic CPU gates and require an explicit external GPU gate."""
    static = run_command_gate(
        "static", static_commands, cwd=sandbox.worktree_path, timeout_seconds=timeout_seconds
    )
    unit = (
        run_command_gate(
            "unit", unit_commands, cwd=sandbox.worktree_path, timeout_seconds=timeout_seconds
        )
        if static.passed
        else GateResult(name="unit", passed=False, detail="blocked by static gate")
    )
    regression = (
        run_command_gate(
            "regression",
            regression_commands,
            cwd=sandbox.worktree_path,
            timeout_seconds=timeout_seconds,
        )
        if static.passed and unit.passed
        else GateResult(name="regression", passed=False, detail="blocked by earlier gate")
    )
    return RepairValidationResult(
        static_gate=static,
        unit_gate=unit,
        regression_gate=regression,
        gpu_smoke_gate=gpu_smoke_gate,
    )
