from __future__ import annotations

import math

import pytest

from trading_bot.evaluation import (
    FactorObservation,
    PredictionPoint,
    deflated_sharpe_ratio,
    economic_metrics,
    factor_attribution,
    probability_of_backtest_overfitting,
    rank_ic_metrics,
    robustness_metrics,
)


def test_rank_ic_hand_calculated_and_dimensions() -> None:
    points = []
    for timestamp, fold in ((1, "fold-a"), (2, "fold-b")):
        for index, value in enumerate((1.0, 2.0, 3.0, 4.0)):
            points.append(
                PredictionPoint(
                    asset_id=f"a{index}",
                    timestamp_ns=timestamp,
                    target=value,
                    score=value,
                    fold_id=fold,
                    regime="calm",
                    horizon="15m",
                    sector="tech",
                )
            )
    result = rank_ic_metrics(points)
    assert result.mean_rank_ic == pytest.approx(1.0)
    assert result.median_rank_ic == pytest.approx(1.0)
    assert result.rank_ic_std == pytest.approx(0.0)
    assert result.icir == pytest.approx(0.0)
    assert result.positive_ic_fraction == pytest.approx(1.0)
    assert result.by_fold == {"fold-a": pytest.approx(1.0), "fold-b": pytest.approx(1.0)}
    assert result.by_regime["calm"] == pytest.approx(1.0)
    assert result.by_horizon["15m"] == pytest.approx(1.0)


def test_zero_return_and_drawdown_economic_fixtures() -> None:
    zero = economic_metrics([0.0] * 5)
    assert zero.net_sharpe == pytest.approx(0.0)
    assert zero.cagr == pytest.approx(0.0)
    assert zero.sortino == pytest.approx(0.0)
    assert zero.maximum_drawdown == pytest.approx(0.0)
    assert zero.drawdown_duration_days == 0
    assert zero.calmar == pytest.approx(0.0)
    assert zero.es95 == pytest.approx(0.0)
    assert zero.worst_day == pytest.approx(0.0)
    assert zero.final_nav == pytest.approx(1.0)

    drawdown = economic_metrics([0.10, -0.20, -0.10, 0.40])
    nav1 = 1.10
    nav2 = nav1 * 0.80
    nav3 = nav2 * 0.90
    expected_mdd = nav3 / nav1 - 1.0
    assert drawdown.maximum_drawdown == pytest.approx(expected_mdd)
    assert drawdown.drawdown_duration_days == 2
    assert drawdown.worst_day == pytest.approx(-0.20)


def test_robustness_dsr_pbo_and_factor_attribution() -> None:
    robust = robustness_metrics(
        {"a": [0.01, 0.02, -0.005, 0.01], "b": [0.02, 0.01, 0.0, 0.01]},
        seed_sharpes={1: 1.0, 2: 1.2, 3: 0.8},
        regime_returns={"calm": [0.01, 0.02, 0.01], "volatile": [-0.01, 0.02, 0.03]},
    )
    assert robust.positive_fold_fraction == pytest.approx(1.0)
    assert robust.seed_dispersion == pytest.approx(0.2)

    returns = [0.01, 0.005, -0.002, 0.008, 0.004, 0.006, -0.001, 0.009]
    period_sharpe = (
        sum(returns)
        / len(returns)
        / (
            sum((value - sum(returns) / len(returns)) ** 2 for value in returns)
            / (len(returns) - 1)
        )
        ** 0.5
    )
    dsr = deflated_sharpe_ratio(
        returns,
        trial_period_sharpes=[period_sharpe - 0.2, period_sharpe - 0.1, period_sharpe],
    )
    assert 0.0 <= dsr.probability <= 1.0
    assert dsr.trial_count == 3
    assert dsr.observations == len(returns)

    matrix = [
        [0.01, -0.01, 0.002],
        [0.02, -0.02, 0.001],
        [0.01, 0.01, -0.001],
        [0.015, -0.005, 0.003],
        [0.005, 0.02, 0.001],
        [0.012, -0.01, 0.002],
        [0.009, 0.00, 0.001],
        [0.011, -0.01, 0.002],
    ]
    pbo = probability_of_backtest_overfitting(matrix, split_count=4)
    assert 0.0 <= pbo.probability <= 1.0
    assert pbo.combinations_evaluated == math.comb(4, 2)
    assert pbo.trial_count == 3

    observations = []
    for index in range(10):
        mkt = (index - 4) / 1000.0
        mom = ((index % 3) - 1) / 1000.0
        strategy = 0.001 + 1.5 * mkt - 0.5 * mom
        observations.append(
            FactorObservation(
                date=f"2026-01-{index + 1:02d}",
                strategy_excess_return=strategy,
                factors={"MKT": mkt, "MOM": mom},
            )
        )
    attribution = factor_attribution(observations)
    assert attribution.alpha_daily == pytest.approx(0.001, abs=1e-12)
    assert attribution.betas["MKT"] == pytest.approx(1.5, abs=1e-12)
    assert attribution.betas["MOM"] == pytest.approx(-0.5, abs=1e-12)
    assert attribution.r_squared == pytest.approx(1.0)
