"""Canonical predictive, economic, robustness, and attribution metrics."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from trading_bot.evaluation.contracts import FactorObservation, PredictionPoint


@dataclass(frozen=True, slots=True)
class PredictiveMetrics:
    mean_rank_ic: float
    median_rank_ic: float
    rank_ic_std: float
    icir: float
    positive_ic_fraction: float
    period_count: int
    by_fold: dict[str, float]
    by_regime: dict[str, float]
    by_horizon: dict[str, float]
    by_sector: dict[str, float]


@dataclass(frozen=True, slots=True)
class EconomicMetrics:
    net_sharpe: float | None
    cagr: float
    sortino: float | None
    maximum_drawdown: float
    drawdown_duration_days: int
    calmar: float | None
    es95: float
    worst_day: float
    final_nav: float
    trading_days: int


@dataclass(frozen=True, slots=True)
class RobustnessMetrics:
    fold_sharpes: dict[str, float | None]
    median_fold_sharpe: float | None
    worst_fold_sharpe: float | None
    positive_fold_fraction: float
    seed_dispersion: float | None
    regime_sharpes: dict[str, float | None]


@dataclass(frozen=True, slots=True)
class DeflatedSharpeResult:
    probability: float
    observed_period_sharpe: float
    benchmark_period_sharpe: float
    trial_count: int
    observations: int
    skewness: float
    kurtosis: float


@dataclass(frozen=True, slots=True)
class PBOResult:
    probability: float
    split_count: int
    combinations_evaluated: int
    trial_count: int
    logits: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FactorAttribution:
    alpha_daily: float
    alpha_annualized: float
    betas: dict[str, float]
    r_squared: float
    residual_std: float
    observations: int


def rank_ic_metrics(points: Sequence[PredictionPoint]) -> PredictiveMetrics:
    """Compute cross-sectional Spearman Rank IC without mixing timestamps."""
    if not points:
        raise ValueError("rank IC requires at least one prediction point")
    grouped: dict[int, list[PredictionPoint]] = {}
    for point in points:
        grouped.setdefault(point.timestamp_ns, []).append(point)

    period_ics: list[float] = []
    for timestamp in sorted(grouped):
        rows = grouped[timestamp]
        if len(rows) < 2:
            continue
        ic = _spearman(
            [row.score for row in rows],
            [row.target for row in rows],
        )
        if ic is not None:
            period_ics.append(ic)
    if not period_ics:
        raise ValueError("rank IC requires at least one non-degenerate cross-section")

    mean_ic = statistics.fmean(period_ics)
    median_ic = statistics.median(period_ics)
    std_ic = statistics.stdev(period_ics) if len(period_ics) > 1 else 0.0
    icir = mean_ic / std_ic if std_ic > 0 else 0.0
    return PredictiveMetrics(
        mean_rank_ic=mean_ic,
        median_rank_ic=median_ic,
        rank_ic_std=std_ic,
        icir=icir,
        positive_ic_fraction=sum(value > 0 for value in period_ics) / len(period_ics),
        period_count=len(period_ics),
        by_fold=_dimension_mean_ic(points, "fold_id"),
        by_regime=_dimension_mean_ic(points, "regime"),
        by_horizon=_dimension_mean_ic(points, "horizon"),
        by_sector=_dimension_mean_ic(points, "sector"),
    )


def economic_metrics(
    daily_returns: Sequence[float],
    *,
    annualization_days: int = 252,
    risk_free_rate_annual: float = 0.0,
) -> EconomicMetrics:
    """Compute all headline economic metrics from one net daily return series."""
    returns = _finite_returns(daily_returns)
    if annualization_days < 1:
        raise ValueError("annualization_days must be positive")
    if risk_free_rate_annual <= -1 or not math.isfinite(risk_free_rate_annual):
        raise ValueError("risk_free_rate_annual must be finite and greater than -100%")
    daily_rf = (1.0 + risk_free_rate_annual) ** (1.0 / annualization_days) - 1.0
    excess = [value - daily_rf for value in returns]
    sharpe = _annualized_ratio(excess, annualization_days)

    nav = 1.0
    nav_path = [nav]
    for value in returns:
        if value <= -1.0:
            raise ValueError("daily return cannot be less than or equal to -100%")
        nav *= 1.0 + value
        nav_path.append(nav)
    cagr = nav ** (annualization_days / len(returns)) - 1.0

    target = daily_rf
    downside_variance = statistics.fmean(min(value - target, 0.0) ** 2 for value in returns)
    downside_deviation = math.sqrt(downside_variance)
    mean_excess = statistics.fmean(value - target for value in returns)
    if downside_deviation > 0:
        sortino: float | None = math.sqrt(annualization_days) * mean_excess / downside_deviation
    elif math.isclose(mean_excess, 0.0, abs_tol=1e-15):
        sortino = 0.0
    else:
        sortino = None

    maximum_drawdown, duration = _drawdown_stats(nav_path)
    if maximum_drawdown < 0:
        calmar: float | None = cagr / abs(maximum_drawdown)
    elif math.isclose(cagr, 0.0, abs_tol=1e-15):
        calmar = 0.0
    else:
        calmar = None

    losses = sorted(-value for value in returns)
    var95 = _quantile(losses, 0.95)
    tail = [value for value in losses if value >= var95]
    es95 = statistics.fmean(tail)
    return EconomicMetrics(
        net_sharpe=sharpe,
        cagr=cagr,
        sortino=sortino,
        maximum_drawdown=maximum_drawdown,
        drawdown_duration_days=duration,
        calmar=calmar,
        es95=es95,
        worst_day=min(returns),
        final_nav=nav,
        trading_days=len(returns),
    )


def robustness_metrics(
    fold_returns: Mapping[str, Sequence[float]],
    *,
    seed_sharpes: Mapping[int, float] | None = None,
    regime_returns: Mapping[str, Sequence[float]] | None = None,
    annualization_days: int = 252,
) -> RobustnessMetrics:
    if not fold_returns:
        raise ValueError("robustness metrics require at least one fold")
    fold_sharpes = {
        name: economic_metrics(values, annualization_days=annualization_days).net_sharpe
        for name, values in sorted(fold_returns.items())
    }
    finite_fold_sharpes = [value for value in fold_sharpes.values() if value is not None]
    median_fold = statistics.median(finite_fold_sharpes) if finite_fold_sharpes else None
    worst_fold = min(finite_fold_sharpes) if finite_fold_sharpes else None
    positive_fold_fraction = sum(
        statistics.fmean(_finite_returns(values)) > 0 for values in fold_returns.values()
    ) / len(fold_returns)

    seed_dispersion: float | None = None
    if seed_sharpes:
        values = list(seed_sharpes.values())
        if any(not math.isfinite(value) for value in values):
            raise ValueError("seed sharpes must be finite")
        seed_dispersion = statistics.stdev(values) if len(values) > 1 else 0.0

    regime_sharpes: dict[str, float | None] = {}
    if regime_returns:
        regime_sharpes = {
            name: economic_metrics(values, annualization_days=annualization_days).net_sharpe
            for name, values in sorted(regime_returns.items())
        }

    return RobustnessMetrics(
        fold_sharpes=fold_sharpes,
        median_fold_sharpe=median_fold,
        worst_fold_sharpe=worst_fold,
        positive_fold_fraction=positive_fold_fraction,
        seed_dispersion=seed_dispersion,
        regime_sharpes=regime_sharpes,
    )


def deflated_sharpe_ratio(
    returns: Sequence[float],
    *,
    trial_period_sharpes: Sequence[float],
) -> DeflatedSharpeResult:
    """Bailey/López de Prado-style DSR probability using period Sharpe inputs.

    ``trial_period_sharpes`` must include every comparable attempted trial in the
    multiple-testing family and use the same unannualized period-Sharpe convention.
    """
    values = _finite_returns(returns)
    if len(values) < 3:
        raise ValueError("DSR requires at least three return observations")
    if not trial_period_sharpes:
        raise ValueError("DSR requires at least one attempted trial")
    if any(not math.isfinite(value) for value in trial_period_sharpes):
        raise ValueError("trial sharpes must be finite")

    mean = statistics.fmean(values)
    stdev = statistics.stdev(values)
    if stdev <= 0:
        raise ValueError("DSR requires non-zero return variance")
    observed = mean / stdev
    skewness, kurtosis = _skew_kurtosis(values)
    benchmark = _expected_maximum_sharpe(trial_period_sharpes)
    denominator_term = 1.0 - skewness * observed + ((kurtosis - 1.0) / 4.0) * observed**2
    if denominator_term <= 0:
        raise ValueError("DSR denominator is not positive")
    statistic = (
        (observed - benchmark)
        * math.sqrt(len(values) - 1)
        / math.sqrt(denominator_term)
    )
    probability = statistics.NormalDist().cdf(statistic)
    return DeflatedSharpeResult(
        probability=probability,
        observed_period_sharpe=observed,
        benchmark_period_sharpe=benchmark,
        trial_count=len(trial_period_sharpes),
        observations=len(values),
        skewness=skewness,
        kurtosis=kurtosis,
    )


def probability_of_backtest_overfitting(
    strategy_returns: Sequence[Sequence[float]],
    *,
    split_count: int = 8,
) -> PBOResult:
    """Estimate CSCV-style PBO from a periods x trials return matrix."""
    matrix = np.asarray(strategy_returns, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("strategy_returns must be a periods x trials matrix")
    periods, trials = matrix.shape
    if trials < 2:
        raise ValueError("PBO requires at least two trials")
    if split_count < 2 or split_count % 2:
        raise ValueError("split_count must be an even integer >= 2")
    if periods < split_count:
        raise ValueError("PBO requires at least one observation per split")
    if not bool(np.isfinite(matrix).all()):
        raise ValueError("PBO returns must be finite")

    blocks = [np.asarray(block, dtype=np.int64) for block in np.array_split(np.arange(periods), split_count)]
    half = split_count // 2
    logits: list[float] = []
    for train_blocks in combinations(range(split_count), half):
        train_set = set(train_blocks)
        test_blocks = tuple(index for index in range(split_count) if index not in train_set)
        train_indices = np.concatenate([blocks[index] for index in train_blocks])
        test_indices = np.concatenate([blocks[index] for index in test_blocks])
        train_scores = _column_sharpes(matrix[train_indices, :])
        selected = int(np.argmax(train_scores))
        test_scores = _column_sharpes(matrix[test_indices, :])
        order = np.argsort(test_scores, kind="stable")
        ascending_rank = int(np.where(order == selected)[0][0]) + 1
        omega = ascending_rank / (trials + 1.0)
        logits.append(math.log(omega / (1.0 - omega)))
    probability = sum(value <= 0 for value in logits) / len(logits)
    return PBOResult(
        probability=probability,
        split_count=split_count,
        combinations_evaluated=len(logits),
        trial_count=trials,
        logits=tuple(logits),
    )


def factor_attribution(
    observations: Sequence[FactorObservation],
    *,
    annualization_days: int = 252,
) -> FactorAttribution:
    """OLS alpha/beta attribution for caller-supplied common factor returns."""
    if len(observations) < 3:
        raise ValueError("factor attribution requires at least three observations")
    names = sorted(observations[0].factors)
    if not names:
        raise ValueError("factor attribution requires factor columns")
    if any(sorted(row.factors) != names for row in observations):
        raise ValueError("factor columns must match for every observation")
    if len(observations) <= len(names) + 1:
        raise ValueError("factor attribution requires more observations than coefficients")

    y = np.asarray([row.strategy_excess_return for row in observations], dtype=np.float64)
    x_values = [[1.0, *(row.factors[name] for name in names)] for row in observations]
    x = np.asarray(x_values, dtype=np.float64)
    coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ coefficients
    residual = y - fitted
    ss_res = float(np.dot(residual, residual))
    centered = y - float(np.mean(y))
    ss_total = float(np.dot(centered, centered))
    r_squared = 1.0 - ss_res / ss_total if ss_total > 0 else 0.0
    residual_std = float(np.std(residual, ddof=len(names) + 1))
    alpha_daily = float(coefficients[0])
    return FactorAttribution(
        alpha_daily=alpha_daily,
        alpha_annualized=(1.0 + alpha_daily) ** annualization_days - 1.0,
        betas={name: float(coefficients[index + 1]) for index, name in enumerate(names)},
        r_squared=r_squared,
        residual_std=residual_std,
        observations=len(observations),
    )


def _dimension_mean_ic(points: Sequence[PredictionPoint], attribute: str) -> dict[str, float]:
    grouped: dict[str, list[PredictionPoint]] = {}
    for point in points:
        key = str(getattr(point, attribute))
        grouped.setdefault(key, []).append(point)
    result: dict[str, float] = {}
    for key in sorted(grouped):
        timestamp_groups: dict[int, list[PredictionPoint]] = {}
        for point in grouped[key]:
            timestamp_groups.setdefault(point.timestamp_ns, []).append(point)
        ics: list[float] = []
        for rows in timestamp_groups.values():
            if len(rows) < 2:
                continue
            value = _spearman([row.score for row in rows], [row.target for row in rows])
            if value is not None:
                ics.append(value)
        if ics:
            result[key] = statistics.fmean(ics)
    return result


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman inputs must have equal length >= 2")
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    left_mean = statistics.fmean(left_rank)
    right_mean = statistics.fmean(right_rank)
    left_centered = [value - left_mean for value in left_rank]
    right_centered = [value - right_mean for value in right_rank]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(left_centered, right_centered, strict=True)) / denominator


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average_rank
        cursor = end
    return ranks


def _finite_returns(values: Sequence[float]) -> list[float]:
    if not values:
        raise ValueError("at least one return observation is required")
    result = [float(value) for value in values]
    if any(not math.isfinite(value) for value in result):
        raise ValueError("returns must be finite")
    return result


def _annualized_ratio(values: Sequence[float], annualization_days: int) -> float | None:
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    if stdev > 0:
        return math.sqrt(annualization_days) * mean / stdev
    if math.isclose(mean, 0.0, abs_tol=1e-15):
        return 0.0
    return None


def _drawdown_stats(nav_path: Sequence[float]) -> tuple[float, int]:
    peak = nav_path[0]
    maximum_drawdown = 0.0
    current_duration = 0
    maximum_duration = 0
    for nav in nav_path[1:]:
        if nav >= peak:
            peak = nav
            current_duration = 0
        else:
            current_duration += 1
            maximum_duration = max(maximum_duration, current_duration)
            maximum_drawdown = min(maximum_drawdown, nav / peak - 1.0)
    return maximum_drawdown, maximum_duration


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires values")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _skew_kurtosis(values: Sequence[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    centered = [value - mean for value in values]
    m2 = statistics.fmean(value**2 for value in centered)
    if m2 <= 0:
        raise ValueError("moments require non-zero variance")
    m3 = statistics.fmean(value**3 for value in centered)
    m4 = statistics.fmean(value**4 for value in centered)
    return m3 / m2**1.5, m4 / m2**2


def _expected_maximum_sharpe(trial_sharpes: Sequence[float]) -> float:
    count = len(trial_sharpes)
    if count <= 1:
        return 0.0
    sigma = statistics.stdev(trial_sharpes)
    if sigma <= 0:
        return statistics.fmean(trial_sharpes)
    normal = statistics.NormalDist()
    euler_gamma = 0.5772156649015329
    first_probability = min(max(1.0 - 1.0 / count, 1e-12), 1.0 - 1e-12)
    second_probability = min(max(1.0 - 1.0 / (count * math.e), 1e-12), 1.0 - 1e-12)
    expected_standard_max = (
        (1.0 - euler_gamma) * normal.inv_cdf(first_probability)
        + euler_gamma * normal.inv_cdf(second_probability)
    )
    return statistics.fmean(trial_sharpes) + sigma * expected_standard_max


def _column_sharpes(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    means = np.mean(matrix, axis=0)
    if matrix.shape[0] > 1:
        stdev = np.std(matrix, axis=0, ddof=1)
    else:
        stdev = np.zeros(matrix.shape[1], dtype=np.float64)
    result = np.zeros(matrix.shape[1], dtype=np.float64)
    nonzero = stdev > 0
    result[nonzero] = means[nonzero] / stdev[nonzero]
    zero_positive = (~nonzero) & (means > 0)
    zero_negative = (~nonzero) & (means < 0)
    result[zero_positive] = np.finfo(np.float64).max
    result[zero_negative] = -np.finfo(np.float64).max
    return result
