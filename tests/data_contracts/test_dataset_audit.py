"""Tests for dataset summary, sanity, universe turnover, and split audit reports."""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime, timedelta

import pytest

from trading_bot.data.audit import AuditObservation, build_dataset_audit_report
from trading_bot.data.splits import DateRange, SplitManifest, WalkForwardFold
from trading_bot.data.universe import UniverseMember, UniverseSnapshot


START = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)


def observation(asset: str, minute: int, close: float | None, volume: float | None):
    return AuditObservation(asset, START + timedelta(minutes=minute), close, volume)


def snapshot(day: int, ids: tuple[str, ...]) -> UniverseSnapshot:
    members = tuple(
        UniverseMember(
            security_id=security_id,
            symbol=security_id.upper(),
            average_dollar_volume=1000,
            latest_price=10,
            history_observations=20,
            rank=index,
        )
        for index, security_id in enumerate(ids, 1)
    )
    return UniverseSnapshot(
        as_of=date(2024, 1, day),
        security_master_version="sm-v1",
        policy_version="u-v1",
        policy_sha256="a" * 64,
        members=members,
    )


def split_manifest() -> SplitManifest:
    return SplitManifest(
        split_version="s-v1",
        dataset_version="d-v1",
        folds=(
            WalkForwardFold(
                fold_id="f1",
                train=DateRange(start=date(2020, 1, 1), end=date(2020, 12, 31)),
                validation=DateRange(start=date(2021, 1, 1), end=date(2021, 6, 30)),
            ),
        ),
        final_holdout_id="final-secret",
        final_holdout=DateRange(start=date(2022, 1, 1), end=date(2022, 12, 31)),
    )


def test_report_counts_missingness_assets_and_time_range() -> None:
    rows = [
        observation("a", 0, 100, 10),
        observation("b", 0, None, 20),
        observation("a", 1, 101, None),
    ]
    report = build_dataset_audit_report(rows)
    assert report.total_rows == 3
    assert report.unique_assets == 2
    assert report.missing_close == 1
    assert report.missing_volume == 1
    assert report.timestamp_start == START
    assert report.timestamp_end == START + timedelta(minutes=1)
    assert report.asset_counts_by_date == {"2024-01-02": 2}


def test_report_tracks_nonfinite_and_negative_values_without_polluting_summaries() -> None:
    report = build_dataset_audit_report(
        [
            observation("a", 0, math.nan, math.inf),
            observation("a", 1, 100, -1),
            observation("a", 2, 101, 10),
        ]
    )
    assert report.nonfinite_close == 1
    assert report.nonfinite_volume == 1
    assert report.negative_volume == 1
    assert report.volume_summary.count == 1
    assert report.volume_summary.mean == 10


def test_return_distribution_is_computed_per_asset_chronologically() -> None:
    report = build_dataset_audit_report(
        [
            observation("a", 2, 121, 10),
            observation("a", 0, 100, 10),
            observation("a", 1, 110, 10),
            observation("b", 0, 50, 10),
            observation("b", 1, 45, 10),
        ]
    )
    assert report.return_summary.count == 3
    assert report.return_summary.minimum == pytest.approx(-0.1)
    assert report.return_summary.maximum == pytest.approx(0.1)


def test_universe_turnover_reports_entered_and_exited_names() -> None:
    report = build_dataset_audit_report(
        [], universe_snapshots=[snapshot(2, ("a", "b")), snapshot(9, ("b", "c"))]
    )
    change = report.universe_turnover[0]
    assert change.entered == ("c",)
    assert change.exited == ("a",)
    assert change.one_way_turnover == 0.5


def test_split_timeline_reports_routine_dates_but_not_final_holdout_dates() -> None:
    report = build_dataset_audit_report([], split_manifest=split_manifest())
    assert [(item.partition, item.start, item.end) for item in report.split_timeline] == [
        ("train", date(2020, 1, 1), date(2020, 12, 31)),
        ("validation", date(2021, 1, 1), date(2021, 6, 30)),
    ]
    assert report.final_holdout_id == "final-secret"
    markdown = report.to_markdown()
    assert "final-secret" in markdown
    assert "2022-01-01" not in markdown
    assert "dates not exposed" in markdown


def test_canonical_json_and_markdown_are_deterministic() -> None:
    rows = [observation("a", 0, 100, 10), observation("a", 1, 101, 11)]
    first = build_dataset_audit_report(rows, split_manifest=split_manifest())
    second = build_dataset_audit_report(rows, split_manifest=split_manifest())
    assert first.canonical_json() == second.canonical_json()
    assert json.loads(first.canonical_json())["total_rows"] == 2
    assert first.to_markdown() == second.to_markdown()
