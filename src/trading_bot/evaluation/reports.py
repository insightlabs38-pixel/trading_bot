"""Deterministic machine-readable and human-readable evaluation reports."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from trading_bot.evaluation.leaderboard import LeaderboardRow, TrialEvaluation


class EvaluationReportError(RuntimeError):
    """Raised when an evaluation report cannot be published safely."""


@dataclass(frozen=True, slots=True)
class ReportWriteResult:
    path: Path
    json_sha256: str
    markdown_sha256: str
    manifest_sha256: str


def write_evaluation_report(
    evaluations: Sequence[TrialEvaluation],
    leaderboard: Sequence[LeaderboardRow],
    destination: str | Path,
    *,
    metadata: Mapping[str, str] | None = None,
) -> ReportWriteResult:
    """Atomically publish a canonical JSON report plus deterministic Markdown summary."""
    if not evaluations or not leaderboard:
        raise ValueError("evaluation report requires trials and leaderboard rows")
    destination = Path(destination)
    if destination.exists():
        raise EvaluationReportError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    evaluation_ids = {item.trial_id for item in evaluations}
    leaderboard_ids = {row.trial_id for row in leaderboard}
    if evaluation_ids != leaderboard_ids:
        raise ValueError("leaderboard and evaluation trial IDs do not match")

    payload = {
        "schema_version": 1,
        "metadata": dict(sorted((metadata or {}).items())),
        "leaderboard": [asdict(row) for row in leaderboard],
        "trials": [asdict(item) for item in sorted(evaluations, key=lambda item: item.trial_id)],
    }
    json_bytes = _canonical_json_bytes(payload)
    markdown_bytes = _markdown_report(leaderboard, metadata=metadata).encode("utf-8")
    json_sha = hashlib.sha256(json_bytes).hexdigest()
    markdown_sha = hashlib.sha256(markdown_bytes).hexdigest()
    manifest = {
        "schema_version": 1,
        "files": {
            "report.json": {"size": len(json_bytes), "sha256": json_sha},
            "report.md": {"size": len(markdown_bytes), "sha256": markdown_sha},
        },
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=str(destination.parent))
    )
    try:
        files = {
            "report.json": json_bytes,
            "report.md": markdown_bytes,
            "manifest.json": manifest_bytes,
            "manifest.sha256": f"{manifest_sha}\n".encode("ascii"),
        }
        for name, content in files.items():
            path = temporary / name
            path.write_bytes(content)
            _fsync_file(path)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return ReportWriteResult(
        path=destination,
        json_sha256=json_sha,
        markdown_sha256=markdown_sha,
        manifest_sha256=manifest_sha,
    )


def verify_evaluation_report(path: str | Path) -> dict[str, Any]:
    """Verify report manifest and file hashes before downstream leaderboard use."""
    root = Path(path)
    try:
        manifest_bytes = (root / "manifest.json").read_bytes()
        expected_manifest_sha = (root / "manifest.sha256").read_text(
            encoding="ascii"
        ).strip()
        manifest = json.loads(manifest_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationReportError("invalid evaluation report manifest") from exc
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha:
        raise EvaluationReportError("evaluation report manifest checksum mismatch")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise EvaluationReportError("unsupported evaluation report manifest")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise EvaluationReportError("evaluation report file manifest is invalid")
    for name in ("report.json", "report.md"):
        record = files.get(name)
        if not isinstance(record, dict):
            raise EvaluationReportError(f"evaluation report manifest missing {name}")
        expected_size = record.get("size")
        expected_sha = record.get("sha256")
        file_path = root / name
        try:
            content = file_path.read_bytes()
        except OSError as exc:
            raise EvaluationReportError(f"evaluation report file missing: {name}") from exc
        if len(content) != expected_size or hashlib.sha256(content).hexdigest() != expected_sha:
            raise EvaluationReportError(f"evaluation report file checksum mismatch: {name}")
    try:
        payload = json.loads((root / "report.json").read_bytes())
    except json.JSONDecodeError as exc:
        raise EvaluationReportError("evaluation JSON report is invalid") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise EvaluationReportError("unsupported evaluation JSON report")
    return payload


def _markdown_report(
    leaderboard: Sequence[LeaderboardRow],
    *,
    metadata: Mapping[str, str] | None,
) -> str:
    lines = ["# Canonical Evaluation Leaderboard", ""]
    if metadata:
        for key, value in sorted(metadata.items()):
            lines.append(f"- **{key}**: `{value}`")
        lines.append("")
    lines.extend(
        [
            "| Rank | Trial | Eligible | Mean Rank IC | Net Sharpe | Calmar | MDD | Avg Turnover |",
            "|---:|---|:---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in leaderboard:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.rank),
                    row.trial_id,
                    "yes" if row.eligible else "no",
                    _format_float(row.mean_rank_ic),
                    _format_optional(row.net_sharpe),
                    _format_optional(row.calmar),
                    _format_float(row.maximum_drawdown),
                    _format_float(row.average_turnover),
                ]
            )
            + " |"
        )
    disqualified = [row for row in leaderboard if not row.eligible]
    if disqualified:
        lines.extend(["", "## Disqualifications", ""])
        for row in disqualified:
            lines.append(f"- **{row.trial_id}**: {', '.join(row.disqualification_reasons)}")
    lines.append("")
    return "\n".join(lines)


def _format_float(value: float) -> str:
    return f"{value:.8f}"


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else _format_float(value)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
