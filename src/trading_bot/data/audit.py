"""Dataset audit reports for missingness, panel coverage, sanity, universe, and splits."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Iterable

from trading_bot.data.splits import RoutineSplitManifest, SplitManifest
from trading_bot.data.universe import UniverseSnapshot


class DatasetAuditError(RuntimeError):
    """Raised when structural audit inputs are internally inconsistent."""


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
    duplicate_observations: int
    missing_close: int
    missing_volume: int
    nonfinite_close: int
    nonfinite_volume: int
    nonpositive_close: int
    negative_volume: int
    nonfinite_return: int
    asset_counts_by_date: dict[str, int]
    return_summary: NumericSummary
    volume_summary: NumericSummary
    universe_turnover: tuple[UniverseTurnover, ...]
    split_timeline: tuple[SplitTimelineEntry, ...]
    final_holdout_id: str | None
    split_sha256: str | None

    def canonical_json(self) -> str:
        return json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        )

    def to_markdown(self) -> str:
        lines = [
            "# Dataset Audit Report",
            "",
            f"- Rows: {self.total_rows}",
            f"- Unique assets: {self.unique_assets}",
            f"- Timestamp range: {self.timestamp_start} → {self.timestamp_end}",
            f"- Duplicate asset/timestamp rows: {self.duplicate_observations}",
            f"- Missing close: {self.missing_close}",
            f"- Missing volume: {self.missing_volume}",
            f"- Non-finite close: {self.nonfinite_close}",
            f"- Non-finite volume: {self.nonfinite_volume}",
            f"- Non-positive close: {self.nonpositive_close}",
            f"- Negative volume: {self.negative_volume}",
            f"- Non-finite derived returns: {self.nonfinite_return}",
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
        if self.split_sha256 is not None:
            lines.append(f"- Full split SHA-256: {self.split_sha256}")
        lines.extend(
            [
                "",
                "## Return sanity",
                _numeric_summary_markdown(self.return_summary),
                "",
                "## Volume sanity",
                _numeric_summary_markdown(self.volume_summary),
            ]
        )
        return "\n".join(lines) + "\n"


def build_dataset_audit_report(
    observations: Iterable[AuditObservation],
    *,
    universe_snapshots: Iterable[UniverseSnapshot] = (),
    split_manifest: SplitManifest | RoutineSplitManifest | None = None,
) -> DatasetAuditReport:
    rows = tuple(observations)
    assets = {row.security_id for row in rows}
    timestamps = [row.timestamp for row in rows]
    identities: set[tuple[str, datetime]] = set()
    duplicate_observations = 0
    asset_dates: dict[str, set[str]] = defaultdict(set)
    valid_volumes: list[float] = []
    close_history: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    missing_close = missing_volume = nonfinite_close = nonfinite_volume = 0
    nonpositive_close = negative_volume = nonfinite_return = 0

    for row in rows:
        identity = (row.security_id, row.timestamp)
        if identity in identities:
            duplicate_observations += 1
        else:
            identities.add(identity)
        asset_dates[row.timestamp.date().isoformat()].add(row.security_id)
        if row.close is None:
            missing_close += 1
        elif not math.isfinite(row.close):
            nonfinite_close += 1
        elif row.close <= 0:
            nonpositive_close += 1
        else:
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
            value = current / previous - 1.0
            if math.isfinite(value):
                returns.append(value)
            else:
                nonfinite_return += 1

    snapshots = _validate_universe_snapshots(tuple(universe_snapshots))
    turnover = tuple(
        _universe_turnover(previous, current)
        for previous, current in zip(snapshots, snapshots[1:], strict=False)
    )
    routine_manifest = _routine_manifest(split_manifest)
    timeline = _split_timeline(routine_manifest)
    return DatasetAuditReport(
        total_rows=len(rows),
        unique_assets=len(assets),
        timestamp_start=min(timestamps) if timestamps else None,
        timestamp_end=max(timestamps) if timestamps else None,
        duplicate_observations=duplicate_observations,
        missing_close=missing_close,
        missing_volume=missing_volume,
        nonfinite_close=nonfinite_close,
        nonfinite_volume=nonfinite_volume,
        nonpositive_close=nonpositive_close,
        negative_volume=negative_volume,
        nonfinite_return=nonfinite_return,
        asset_counts_by_date={day: len(values) for day, values in sorted(asset_dates.items())},
        return_summary=_numeric_summary(returns),
        volume_summary=_numeric_summary(valid_volumes),
        universe_turnover=turnover,
        split_timeline=timeline,
        final_holdout_id=(
            None if routine_manifest is None else routine_manifest.final_holdout_id
        ),
        split_sha256=(
            None if routine_manifest is None else routine_manifest.full_split_sha256
        ),
    )


def _numeric_summary(values: list[float]) -> NumericSummary:
    if not values:
        return NumericSummary(0, None, None, None, None)
    if any(not math.isfinite(value) for value in values):
        raise DatasetAuditError("numeric summary received non-finite values")
    scale = max(abs(value) for value in values)
    if scale == 0:
        return NumericSummary(len(values), 0.0, 0.0, 0.0, 0.0)
    scaled = [value / scale for value in values]
    mean_scaled = math.fsum(scaled) / len(scaled)
    variance_scaled = math.fsum(
        (value - mean_scaled) ** 2 for value in scaled
    ) / len(scaled)
    return NumericSummary(
        count=len(values),
        mean=scale * mean_scaled,
        std=scale * math.sqrt(max(0.0, variance_scaled)),
        minimum=min(values),
        maximum=max(values),
    )


def _validate_universe_snapshots(
    snapshots: tuple[UniverseSnapshot, ...],
) -> tuple[UniverseSnapshot, ...]:
    ordered = tuple(sorted(snapshots, key=lambda item: item.as_of))
    dates: set[date] = set()
    for snapshot in ordered:
        if snapshot.as_of in dates:
            raise DatasetAuditError("universe snapshot dates must be unique")
        dates.add(snapshot.as_of)
        security_ids = snapshot.security_ids
        if len(set(security_ids)) != len(security_ids):
            raise DatasetAuditError("universe snapshot security IDs must be unique")
        ranks = tuple(member.rank for member in snapshot.members)
        if ranks != tuple(range(1, len(snapshot.members) + 1)):
            raise DatasetAuditError("universe snapshot ranks must be contiguous starting at 1")
    return ordered


def _universe_turnover(previous: UniverseSnapshot, current: UniverseSnapshot) -> UniverseTurnover:
    before = set(previous.security_ids)
    after = set(current.security_ids)
    entered = tuple(sorted(after - before))
    exited = tuple(sorted(before - after))
    denominator = len(before) + len(after)
    one_way_turnover = 0.0 if denominator == 0 else (len(entered) + len(exited)) / denominator
    return UniverseTurnover(
        previous_date=previous.as_of,
        current_date=current.as_of,
        previous_count=len(before),
        current_count=len(after),
        entered=entered,
        exited=exited,
        one_way_turnover=one_way_turnover,
    )


def _routine_manifest(
    manifest: SplitManifest | RoutineSplitManifest | None,
) -> RoutineSplitManifest | None:
    if manifest is None:
        return None
    if isinstance(manifest, SplitManifest):
        return manifest.routine_view()
    return manifest


def _split_timeline(
    manifest: RoutineSplitManifest | None,
) -> tuple[SplitTimelineEntry, ...]:
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


def _numeric_summary_markdown(summary: NumericSummary) -> str:
    return (
        f"- count={summary.count}, mean={summary.mean}, std={summary.std}, "
        f"min={summary.minimum}, max={summary.maximum}"
    )


def _json_default(value: object) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"cannot serialize {type(value).__name__} in dataset audit report")
