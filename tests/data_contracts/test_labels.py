"""Tests for future-only multi-horizon label generation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trading_bot.data.labels import (
    LabelGenerationError,
    LabelObservation,
    LabelPolicy,
    generate_labels,
)

START = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)


def observation(security_id: str, minute: int, close: float) -> LabelObservation:
    return LabelObservation(security_id, START + timedelta(minutes=minute), close)


def row_for(labels, security_id: str, minute: int):
    timestamp = START + timedelta(minutes=minute)
    return next(
        row for row in labels if row.security_id == security_id and row.timestamp == timestamp
    )


def test_primary_future_returns_and_direction_labels_use_exact_endpoints() -> None:
    rows = [observation("a", minute, 100 + minute) for minute in range(61)]
    labels = generate_labels(rows)
    first = row_for(labels, "a", 0)
    assert first.future_returns[5] == pytest.approx(0.05)
    assert first.future_returns[15] == pytest.approx(0.15)
    assert first.future_returns[30] == pytest.approx(0.30)
    assert first.future_returns[60] == pytest.approx(0.60)
    assert first.directions[5] == 1


def test_missing_future_endpoint_does_not_interpolate_or_cross_a_gap() -> None:
    rows = [observation("a", 0, 100), observation("a", 6, 106)]
    labels = generate_labels(rows, policy=LabelPolicy(horizons_minutes=(5, 6)))
    first = row_for(labels, "a", 0)
    assert 5 not in first.future_returns
    assert first.future_returns[6] == pytest.approx(0.06)


def test_future_excess_return_uses_reference_security_at_same_endpoints() -> None:
    rows = [
        observation("market", 0, 100),
        observation("a", 0, 100),
        observation("market", 5, 101),
        observation("a", 5, 103),
    ]
    labels = generate_labels(
        rows,
        policy=LabelPolicy(horizons_minutes=(5,), reference_security_id="market"),
    )
    first = row_for(labels, "a", 0)
    assert first.future_returns[5] == pytest.approx(0.03)
    assert first.future_excess_returns[5] == pytest.approx(0.02)
    assert all(row.security_id != "market" for row in labels)


def test_cross_sectional_rank_and_quantile_targets_are_computed_per_timestamp_horizon() -> None:
    rows = [
        observation("a", 0, 100),
        observation("b", 0, 100),
        observation("c", 0, 100),
        observation("a", 5, 103),
        observation("b", 5, 101),
        observation("c", 5, 98),
    ]
    labels = generate_labels(rows, policy=LabelPolicy(horizons_minutes=(5,)))
    assert row_for(labels, "a", 0).cross_sectional_ranks[5] == 1.0
    assert row_for(labels, "b", 0).cross_sectional_ranks[5] == 0.5
    assert row_for(labels, "c", 0).cross_sectional_ranks[5] == 0.0
    assert row_for(labels, "a", 0).quantile_ranks[5] == 1.0


def test_future_volatility_uses_only_path_after_decision_timestamp() -> None:
    rows = [
        observation("a", 0, 100),
        observation("a", 1, 101),
        observation("a", 2, 99),
        observation("a", 3, 102),
        observation("a", 5, 103),
    ]
    labels = generate_labels(rows, policy=LabelPolicy(horizons_minutes=(5,)))
    assert row_for(labels, "a", 0).future_volatility[5] > 0


def test_appending_data_after_existing_horizon_does_not_change_existing_label() -> None:
    prefix = [observation("a", minute, 100 + minute) for minute in range(6)]
    baseline = row_for(
        generate_labels(prefix, policy=LabelPolicy(horizons_minutes=(5,))),
        "a",
        0,
    )
    extended = row_for(
        generate_labels(
            [*prefix, observation("a", 6, 1000), observation("a", 7, 1)],
            policy=LabelPolicy(horizons_minutes=(5,)),
        ),
        "a",
        0,
    )
    assert extended == baseline


def test_duplicate_security_timestamp_is_rejected() -> None:
    row = observation("a", 0, 100)
    with pytest.raises(LabelGenerationError, match="duplicate"):
        generate_labels([row, row])
