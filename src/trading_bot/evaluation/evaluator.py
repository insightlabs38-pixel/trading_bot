"""Top-level CPU evaluator orchestration over saved predictions and explicit strategy weights."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from trading_bot.evaluation.artifacts import PredictionDataset, ScoreField
from trading_bot.evaluation.backtest import (
    BacktestResult,
    cost_stress,
    latency_stress,
    run_backtest,
    spread_stress,
)
from trading_bot.evaluation.contracts import (
    EvaluationAssumptions,
    FactorObservation,
    LatencyReturn,
    PositionPoint,
    ValidityEvidence,
)
from trading_bot.evaluation.leaderboard import TrialEvaluation, validity_status
from trading_bot.evaluation.metrics import (
    DeflatedSharpeResult,
    FactorAttribution,
    PBOResult,
    deflated_sharpe_ratio,
    factor_attribution,
    probability_of_backtest_overfitting,
    rank_ic_metrics,
    robustness_metrics,
)


@dataclass(frozen=True, slots=True)
class TrialEvaluationInputs:
    """All evaluator-side inputs required to reproduce one trial result."""

    trial_id: str
    predictions: PredictionDataset
    positions: tuple[PositionPoint, ...]
    fold_returns: Mapping[str, Sequence[float]]
    validity_evidence: ValidityEvidence = field(default_factory=ValidityEvidence)
    seed_sharpes: Mapping[int, float] | None = None
    regime_returns: Mapping[str, Sequence[float]] | None = None
    trial_period_sharpes: Sequence[float] | None = None
    pbo_family_returns: Sequence[Sequence[float]] | None = None
    factor_observations: Sequence[FactorObservation] | None = None
    latency_returns: tuple[LatencyReturn, ...] = ()
    score_field: ScoreField = "rank_score"
    seed: int | None = None
    regime_by_identity: Mapping[tuple[str, int], str] | None = None
    sector_by_asset: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.trial_id.strip():
            raise ValueError("trial_id must not be blank")
        if not self.positions:
            raise ValueError("trial evaluation requires explicit portfolio positions")
        if not self.fold_returns:
            raise ValueError("trial evaluation requires fold returns")


def positions_from_explicit_weights(
    predictions: PredictionDataset,
    weights: Mapping[tuple[str, int], float],
) -> tuple[PositionPoint, ...]:
    """Join evaluator-owned frozen weights to saved predictions without training imports.

    The saved prediction target is treated as the subsequent realized return for the
    chosen target horizon. Production portfolio construction remains an explicit
    upstream/frozen input rather than being invented inside the evaluator.
    """
    expected_identities = {(row.asset_id, row.timestamp_ns) for row in predictions.records}
    if set(weights) != expected_identities:
        missing = sorted(expected_identities - set(weights))
        extra = sorted(set(weights) - expected_identities)
        raise ValueError(
            f"weight identities do not match predictions; missing={missing} extra={extra}"
        )
    return tuple(
        PositionPoint(
            asset_id=row.asset_id,
            timestamp_ns=row.timestamp_ns,
            weight=float(weights[(row.asset_id, row.timestamp_ns)]),
            realized_return=row.target,
        )
        for row in predictions.records
    )


def evaluate_trial(
    inputs: TrialEvaluationInputs,
    assumptions: EvaluationAssumptions,
) -> TrialEvaluation:
    """Reproduce all CPU-verifiable Phase 6 metrics from evaluator-side artifacts."""
    prediction_points = inputs.predictions.prediction_points(
        score_field=inputs.score_field,
        regime_by_identity=inputs.regime_by_identity,
        sector_by_asset=inputs.sector_by_asset,
        seed=inputs.seed,
    )
    predictive = rank_ic_metrics(prediction_points)
    backtest = run_backtest(inputs.positions, assumptions)
    robustness = robustness_metrics(
        inputs.fold_returns,
        seed_sharpes=inputs.seed_sharpes,
        regime_returns=inputs.regime_returns,
        annualization_days=assumptions.annualization_days,
    )

    dsr: DeflatedSharpeResult | None = None
    if inputs.trial_period_sharpes is not None:
        daily_net = [row.net_return for row in backtest.daily_returns]
        dsr = deflated_sharpe_ratio(
            daily_net,
            trial_period_sharpes=inputs.trial_period_sharpes,
        )

    pbo: PBOResult | None = None
    if inputs.pbo_family_returns is not None:
        pbo = probability_of_backtest_overfitting(inputs.pbo_family_returns)

    attribution: FactorAttribution | None = None
    if inputs.factor_observations is not None:
        attribution = factor_attribution(
            inputs.factor_observations,
            annualization_days=assumptions.annualization_days,
        )

    validity = validity_status(
        inputs.validity_evidence,
        robustness=robustness,
        minimum_positive_fold_fraction=assumptions.minimum_positive_fold_fraction,
        economic=backtest.economic,
    )

    latency = (
        latency_stress(inputs.positions, inputs.latency_returns, assumptions)
        if inputs.latency_returns
        else ()
    )
    return TrialEvaluation(
        trial_id=inputs.trial_id,
        predictive=predictive,
        economic=backtest.economic,
        friction=backtest.friction,
        robustness=robustness,
        validity=validity,
        cost_stress=cost_stress(inputs.positions, assumptions),
        spread_stress=spread_stress(inputs.positions, assumptions),
        latency_stress=latency,
        deflated_sharpe=dsr,
        pbo=pbo,
        factor_attribution=attribution,
        attempted_trial_count=(
            len(inputs.trial_period_sharpes) if inputs.trial_period_sharpes is not None else 1
        ),
    )


def evaluate_backtest_only(
    positions: Sequence[PositionPoint],
    assumptions: EvaluationAssumptions,
) -> BacktestResult:
    """Small public helper for hand-calculated return-accounting fixtures."""
    return run_backtest(positions, assumptions)
