"""Dataset audit reports for missingness, panel coverage, sanity, universe, and splits."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Iterable

from trading_bot.data.splits import SplitManifest
from trading_bot.data.universe import UniverseSnapshot


@dataclass(frozen=True, slots=True)
class AuditObservation:
    security_id: str
    timestamp: datetime
    close: float | None
    volume: float | None

    def __post_init__(self) -> None:
        if not self.security_id.strip():
            raise ValueError("security_id must not be blank")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")


@dataclass(frozen=True, slots=True)
class NumericSummary:
    count: int
    mean: float | None
    std: float | None
    minimum: float | None
    maximum: float | None


@dataclass(frozen=True, slots=True)
class UniverseTurnover:
    previous_date: date
    current_date: date
    previous_count: int
    current_count: int
    entered: tuple[str, ...]
    exited: tuple[str, ...]
    one_way_turnover: float


@dataclass(frozen=True, slots=True)
class SplitTimelineEntry:
    fold_id: str
    partition: str
    start: date
    end: date


@dataclass(frozen=True, slots=True)
class DatasetAuditReport:
    total_rows: int
    unique_assets: int
    timestamp_start: datetime | None
    timestamp_end: datetime | None
    missing_close: int
    missing_volume: int
    nonfinite_close: int
    nonfinite_volume: int
    negative_volume: int
    asset_counts_by_date: dict[str, int]
    return_summary: NumericSummary
    volume_summary: NumericSummary
    universe_turnover: tuple[UniverseTurnover, ...]
    split_timeline: tuple[SplitTimelineEntry, ...]
    final_holdout_id: str | None

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=str)

    def to_markdown(self) -> str:
        lines = [
            "# Dataset Audit Report",
            "",
            f"- Rows: {self.total_rows}",
            f"- Unique assets: {self.unique_assets}",
            f"- Timestamp range: {self.timestamp_start} → {self.timestamp_end}",
            f"- Missing close: {self.missing_close}",
            f"- Missing volume: {self.missing_volume}",
            f"- Non-finite close: {self.nonfinite_close}",
            f"- Non-finite volume: {self.nonfinite_volume}",
            f"- Negative volume: {self.negative_volume}",
            "",
            "## Asset counts through time",
        ]
        lines.extend(
            f"- {day}: {count}" for day, count in sorted(self.asset_counts_by_date.items())
        )
        lines.extend(["", "## Universe turnover"])
        if not self.universe_turnover:
            lines.append("- No consecutive universe snapshots supplied.")
        else:
            for item in self.universe_turnover:
                lines.append(
                    f"- {item.previous_date} → {item.current_date}: "
                    f"turnover={item.one_way_turnover:.6f}, entered={len(item.entered)}, "
                    f"exited={len(item.exited)}"
                )
        lines.extend(["", "## Split timeline"])
        lines.extend(
            f"- {item.fold_id} {item.partition}: {item.start} → {item.end}"
            for item in self.split_timeline
        )
        if self.final_holdout_id is not None:
            lines.append(f"- Protected final holdout: {self.final_holdout_id} (dates not exposed)")
        lines.extend(
            [
                "",
                "## Return sanity",
                f"- {self.return_summary}",
                "",
                "## Volume sanity",
                f"- {self.volume_summary}",
            ]
        )
        return "\n".join(lines) + "\n"


def build_dataset_audit_report(
    observations: Iterable[AuditObservation],
    *,
    universe_snapshots: Iterable[UniverseSnapshot] = (),
    split_manifest: SplitManifest | None = None,
) -> DatasetAuditReport:
    rows = tuple(observations)
    assets = {row.security_id for row in rows}
    timestamps = [row.timestamp for row in rows]
    asset_dates: dict[str, set[str]] = defaultdict(set)
    valid_volumes: list[float] = []
    close_history: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    missing_close = missing_volume = nonfinite_close = nonfinite_volume = negative_volume = 0

    for row in rows:
        asset_dates[row.timestamp.date().isoformat()].add(row.security_id)
        if row.close is None:
            missing_close += 1
        elif not math.isfinite(row.close):
            nonfinite_close += 1
        elif row.close > 0:
            close_history[row.security_id].append((row.timestamp, row.close))
        if row.volume is None:
            missing_volume += 1
        elif not math.isfinite(row.volume):
            nonfinite_volume += 1
        elif row.volume < 0:
            negative_volume += 1
        else:
            valid_volumes.append(row.volume)

    returns: list[float] = []
    for history in close_history.values():
        ordered = sorted(history)
        for (_, previous), (_, current) in zip(ordered, ordered[1:], strict=False):
            returns.append(current / previous - 1.0)

    snapshots = tuple(sorted(universe_snapshots, key=lambda item: item.as_of))
    turnover = tuple(
        _universe_turnover(previous, current)
        for previous, current in zip(snapshots, snapshots[1:], strict=False)
    )
    timeline = _split_timeline(split_manifest)
    return DatasetAuditReport(
        total_rows=len(rows),
        unique_assets=len(assets),
        timestamp_start=min(timestamps) if timestamps else None,
        timestamp_end=max(timestamps) if timestamps else None,
        missing_close=missing_close,
        missing_volume=missing_volume,
        nonfinite_close=nonfinite_close,
        nonfinite_volume=nonfinite_volume,
        negative_volume=negative_volume,
        asset_counts_by_date={day: len(values) for day, values in sorted(asset_dates.items())},
        return_summary=_numeric_summary(returns),
        volume_summary=_numeric_summary(valid_volumes),
        universe_turnover=turnover,
        split_timeline=timeline,
        final_holdout_id=None if split_manifest is None else split_manifest.final_holdout_id,
    )


def _numeric_summary(values: list[float]) -> NumericSummary:
    if not values:
        return NumericSummary(0, None, None, None, None)
    return NumericSummary(
        count=len(values),
        mean=sum(values) / len(values),
        std=statistics.pstdev(values) if len(values) > 1 else 0.0,
        minimum=min(values),
        maximum=max(values),
    )


def _universe_turnover(previous: UniverseSnapshot, current: UniverseSnapshot) -> UniverseTurnover:
    before = set(previous.security_ids)
    after = set(current.security_ids)
    entered = tuple(sorted(after - before))
    exited = tuple(sorted(before - after))
    denominator = max(1, len(before))
    return UniverseTurnover(
        previous_date=previous.as_of,
        current_date=current.as_of,
        previous_count=len(before),
        current_count=len(after),
        entered=entered,
        exited=exited,
        one_way_turnover=(len(entered) + len(exited)) / (2 * denominator),
    )


def _split_timeline(manifest: SplitManifest | None) -> tuple[SplitTimelineEntry, ...]:
    if manifest is None:
        return ()
    entries: list[SplitTimelineEntry] = []
    for fold in manifest.folds:
        entries.append(SplitTimelineEntry(fold.fold_id, "train", fold.train.start, fold.train.end))
        entries.append(
            SplitTimelineEntry(
                fold.fold_id,
                "validation",
                fold.validation.start,
                fold.validation.end,
            )
        )
    return tuple(entries)
