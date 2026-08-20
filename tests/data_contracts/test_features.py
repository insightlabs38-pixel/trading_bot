"""Tests for the causal reference feature pipeline."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trading_bot.data.features import (
    FeatureObservation,
    FeaturePipelineError,
    FeaturePolicy,
    compute_features,
)


def observation(
    security_id: str,
    minute: int,
    close: float,
    *,
    sector: str = "Technology",
    volume: float = 100.0,
) -> FeatureObservation:
    return FeatureObservation(
        security_id=security_id,
        symbol=security_id.upper(),
        sector=sector,
        timestamp=datetime(2024, 1, 2, 14, 30, tzinfo=UTC) + timedelta(minutes=minute),
        open=close - 0.2,
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        volume=volume,
        vwap=close - 0.1,
    )


def small_policy() -> FeaturePolicy:
    return FeaturePolicy(
        return_horizons=(1, 2),
        relative_volume_window=2,
        volatility_window=2,
        range_window=2,
        momentum_window=2,
    )


def test_trailing_features_use_only_security_history_through_current_bar() -> None:
    rows = [
        observation("a", 0, 100),
        observation("a", 1, 101),
        observation("a", 2, 103),
    ]
    features = compute_features(rows, policy=small_policy())
    assert features[0].values["return_1"] == 0.0
    assert features[1].values["return_1"] == pytest.approx(0.01)
    assert features[2].values["return_2"] == pytest.approx(0.03)
    assert features[2].values["momentum"] == pytest.approx(0.03)


def test_prefix_invariance_proves_future_rows_do_not_change_earlier_features() -> None:
    prefix = [observation("a", minute, 100 + minute) for minute in range(3)]
    baseline = compute_features(prefix, policy=small_policy())
    extended = compute_features(
        [*prefix, observation("a", 3, 500), observation("a", 4, 1)],
        policy=small_policy(),
    )
    assert extended[: len(baseline)] == baseline


def test_volume_volatility_range_momentum_and_liquidity_features_are_present() -> None:
    rows = [
        observation("a", 0, 100, volume=100),
        observation("a", 1, 102, volume=200),
        observation("a", 2, 101, volume=300),
    ]
    values = compute_features(rows, policy=small_policy())[-1].values
    assert values["relative_volume"] == pytest.approx(2.0)
    assert values["realized_volatility"] >= 0
    assert values["true_range_fraction"] > 0
    assert values["average_true_range_fraction"] > 0
    assert values["dollar_volume"] == 30_300
    assert values["log_dollar_volume"] > 0
    assert "trend_slope" in values


def test_market_sector_relative_rank_and_regime_features_use_same_timestamp_panel() -> None:
    rows = [
        observation("a", 0, 100, sector="Technology"),
        observation("b", 0, 100, sector="Technology"),
        observation("c", 0, 100, sector="Energy"),
        observation("a", 1, 102, sector="Technology"),
        observation("b", 1, 101, sector="Technology"),
        observation("c", 1, 98, sector="Energy"),
    ]
    features = compute_features(rows, policy=small_policy())
    latest = {row.security_id: row for row in features if row.timestamp.minute == 31}
    assert latest["a"].values["market_relative_return_1"] > 0
    assert latest["c"].values["market_relative_return_1"] < 0
    assert latest["a"].values["sector_relative_return_1"] > 0
    assert latest["a"].values["cross_sectional_return_rank"] == 1.0
    assert latest["c"].values["cross_sectional_return_rank"] == 0.0
    assert latest["a"].values["market_breadth"] == pytest.approx(2 / 3)
    assert latest["a"].values["market_dispersion"] > 0


def test_raw_normalized_and_session_features_are_present() -> None:
    row = compute_features([observation("a", 0, 100)], policy=small_policy())[0]
    assert row.values["close"] == 100
    assert row.values["close_to_vwap"] > 0
    assert row.values["range_fraction"] > 0
    assert row.values["minute_of_day_sin"] == pytest.approx(0.0)
    assert row.values["minute_of_day_cos"] == pytest.approx(1.0)
    assert row.security_id == "a"
    assert row.symbol == "A"
    assert row.sector == "Technology"


def test_feature_rows_are_deterministic_for_unsorted_input() -> None:
    rows = [
        observation("b", 1, 102),
        observation("a", 0, 100),
        observation("a", 1, 101),
        observation("b", 0, 100),
    ]
    forward = compute_features(rows, policy=small_policy())
    reverse = compute_features(reversed(rows), policy=small_policy())
    assert forward == reverse


def test_duplicate_panel_identity_is_rejected() -> None:
    row = observation("a", 0, 100)
    with pytest.raises(FeaturePipelineError, match="duplicate"):
        compute_features([row, row], policy=small_policy())
