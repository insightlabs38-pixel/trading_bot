"""Explicit hierarchical leaderboard logic with fail-closed validity gates."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from trading_bot.evaluation.backtest import FrictionMetrics, StressResult
from trading_bot.evaluation.contracts import ValidityEvidence
from trading_bot.evaluation.metrics import (
    DeflatedSharpeResult,
    EconomicMetrics,
    FactorAttribution,
    PBOResult,
    PredictiveMetrics,
    RobustnessMetrics,
)


@dataclass(frozen=True, slots=True)
class ValidityStatus:
    eligible: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrialEvaluation:
    trial_id: str
    predictive: PredictiveMetrics
    economic: EconomicMetrics
    friction: FrictionMetrics
    robustness: RobustnessMetrics
    validity: ValidityStatus
    cost_stress: tuple[StressResult, ...] = ()
    spread_stress: tuple[StressResult, ...] = ()
    latency_stress: tuple[StressResult, ...] = ()
    deflated_sharpe: DeflatedSharpeResult | None = None
    pbo: PBOResult | None = None
    factor_attribution: FactorAttribution | None = None
    attempted_trial_count: int = 1

    def __post_init__(self) -> None:
        if not self.trial_id.strip():
            raise ValueError("trial_id must not be blank")
        if self.attempted_trial_count < 1:
            raise ValueError("attempted_trial_count must be positive")


@dataclass(frozen=True, slots=True)
class LeaderboardRow:
    rank: int
    trial_id: str
    eligible: bool
    disqualification_reasons: tuple[str, ...]
    mean_rank_ic: float
    icir: float
    net_sharpe: float | None
    calmar: float | None
    maximum_drawdown: float
    sortino: float | None
    es95: float
    average_turnover: float
    total_modeled_cost: float
    positive_fold_fraction: float
    dsr_probability: float | None
    pbo_probability: float | None
    attempted_trial_count: int


def validity_status(
    evidence: ValidityEvidence,
    *,
    robustness: RobustnessMetrics,
    minimum_positive_fold_fraction: float,
    economic: EconomicMetrics,
) -> ValidityStatus:
    """Apply only explicit hard validity/promotion gates; never a hidden score."""
    reasons: list[str] = []
    if not evidence.data_leakage_free:
        reasons.append("data_leakage_or_timestamp_causality_failure")
    if not evidence.cost_accounting_complete:
        reasons.append("invalid_or_missing_cost_accounting")
    if not evidence.final_holdout_clean:
        reasons.append("final_holdout_contamination")
    if not evidence.evaluation_data_complete:
        reasons.append("corrupted_or_incomplete_evaluation_data")
    if not evidence.exposure_valid:
        reasons.append("invalid_leverage_or_exposure")
    if evidence.coverage_fraction < evidence.minimum_coverage_fraction:
        reasons.append("insufficient_required_evaluation_coverage")
    if economic.net_sharpe is None:
        reasons.append("undefined_primary_net_sharpe")
    if robustness.positive_fold_fraction < minimum_positive_fold_fraction:
        reasons.append("positive_fold_fraction_below_frozen_minimum")
    return ValidityStatus(eligible=not reasons, reasons=tuple(reasons))


def build_leaderboard(evaluations: Sequence[TrialEvaluation]) -> tuple[LeaderboardRow, ...]:
    """Sort by the documented hierarchy instead of collapsing metrics into one score."""
    if not evaluations:
        raise ValueError("leaderboard requires at least one trial")
    trial_ids = [evaluation.trial_id for evaluation in evaluations]
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError("leaderboard trial IDs must be unique")

    ordered = sorted(evaluations, key=_ranking_key)
    return tuple(
        LeaderboardRow(
            rank=index + 1,
            trial_id=evaluation.trial_id,
            eligible=evaluation.validity.eligible,
            disqualification_reasons=evaluation.validity.reasons,
            mean_rank_ic=evaluation.predictive.mean_rank_ic,
            icir=evaluation.predictive.icir,
            net_sharpe=evaluation.economic.net_sharpe,
            calmar=evaluation.economic.calmar,
            maximum_drawdown=evaluation.economic.maximum_drawdown,
            sortino=evaluation.economic.sortino,
            es95=evaluation.economic.es95,
            average_turnover=evaluation.friction.average_one_way_turnover,
            total_modeled_cost=evaluation.friction.total_modeled_cost,
            positive_fold_fraction=evaluation.robustness.positive_fold_fraction,
            dsr_probability=(
                evaluation.deflated_sharpe.probability
                if evaluation.deflated_sharpe is not None
                else None
            ),
            pbo_probability=evaluation.pbo.probability if evaluation.pbo is not None else None,
            attempted_trial_count=evaluation.attempted_trial_count,
        )
        for index, evaluation in enumerate(ordered)
    )


def _ranking_key(evaluation: TrialEvaluation) -> tuple[object, ...]:
    # Ascending tuple with negated "higher is better" fields.
    return (
        0 if evaluation.validity.eligible else 1,
        -evaluation.predictive.mean_rank_ic,
        -evaluation.predictive.positive_ic_fraction,
        -evaluation.predictive.icir,
        -_finite_or_floor(evaluation.economic.net_sharpe),
        -_finite_or_floor(evaluation.economic.calmar),
        abs(evaluation.economic.maximum_drawdown),
        -_finite_or_floor(evaluation.economic.sortino),
        evaluation.economic.es95,
        evaluation.friction.total_modeled_cost,
        -evaluation.robustness.positive_fold_fraction,
        evaluation.trial_id,
    )


def _finite_or_floor(value: float | None) -> float:
    if value is None or not math.isfinite(value):
        return -1e300
    return value
