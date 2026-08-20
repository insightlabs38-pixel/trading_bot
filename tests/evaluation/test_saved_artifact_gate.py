from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("torch")

from trading_bot.evaluation import (
    EvaluationAssumptions,
    FactorObservation,
    LatencyReturn,
    TrialEvaluationInputs,
    build_leaderboard,
    evaluate_trial,
    positions_from_explicit_weights,
    read_prediction_artifact,
    verify_evaluation_report,
    write_evaluation_report,
)
from trading_bot.training.predictions import PredictionRecord, write_prediction_artifact


def _timestamp(day: int) -> int:
    return 1_767_225_600_000_000_000 + day * 86_400_000_000_000


def test_saved_predictions_reproduce_complete_cpu_leaderboard_without_training_import(
    tmp_path: Path,
) -> None:
    records = []
    for day in range(8):
        timestamp = _timestamp(day)
        targets = (-0.012 + day * 0.0002, -0.003, 0.006, 0.015 - day * 0.0001)
        for index, target in enumerate(targets):
            records.append(
                PredictionRecord(
                    asset_id=f"A{index}",
                    timestamp_ns=timestamp,
                    target=target,
                    expected_return=target * 0.9,
                    rank_score=float(index),
                )
            )
    artifact = write_prediction_artifact(
        records,
        tmp_path / "predictions",
        dataset_id="dataset-v1",
        split_id="fold-a",
        model_config_hash="model-hash",
        checkpoint_id="step-8",
        target_name="return-15m",
    )

    code = (
        "import sys;"
        "from trading_bot.evaluation.artifacts import read_prediction_artifact;"
        f"d=read_prediction_artifact({str(artifact.path)!r});"
        "assert len(d.records)==32;"
        "assert 'trading_bot.training' not in sys.modules;"
        "assert 'torch' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert completed.returncode == 0, completed.stderr

    dataset = read_prediction_artifact(artifact.path)
    weights = {}
    for record in dataset.records:
        if record.asset_id == "A3":
            weight = 0.5
        elif record.asset_id == "A0":
            weight = -0.5
        else:
            weight = 0.0
        weights[(record.asset_id, record.timestamp_ns)] = weight
    positions = positions_from_explicit_weights(dataset, weights)

    fold_returns = {
        "f1": [0.010, 0.012, 0.008, 0.011],
        "f2": [0.009, 0.007, 0.013, 0.010],
    }
    family_returns = [
        [0.010 + row * 0.0001, 0.003 - row * 0.0001, -0.002 + row * 0.00005]
        for row in range(8)
    ]
    factors = [
        FactorObservation(
            date=f"2026-01-{index + 1:02d}",
            strategy_excess_return=0.001 + 1.2 * (index - 3) / 10_000,
            factors={"MKT": (index - 3) / 10_000},
        )
        for index in range(8)
    ]
    latency_returns = tuple(
        LatencyReturn(
            record.asset_id,
            record.timestamp_ns,
            1.0,
            record.target
            - (0.001 if weights[(record.asset_id, record.timestamp_ns)] > 0 else 0.0),
        )
        for record in dataset.records
    )
    assumptions = EvaluationAssumptions(
        fee_bps=1.0,
        spread_bps=2.0,
        slippage_bps=1.0,
        impact_bps=1.0,
        latency_stress_seconds=(0.0, 1.0),
    )
    evaluation = evaluate_trial(
        TrialEvaluationInputs(
            trial_id="trial-a",
            predictions=dataset,
            positions=positions,
            fold_returns=fold_returns,
            seed_sharpes={1: 1.1, 2: 1.0, 3: 1.2},
            regime_returns={"calm": [0.01, 0.012, 0.009, 0.011]},
            trial_period_sharpes=(0.2, 0.3, 0.4),
            pbo_family_returns=family_returns,
            factor_observations=factors,
            latency_returns=latency_returns,
        ),
        assumptions,
    )
    rows = build_leaderboard([evaluation])
    assert rows[0].trial_id == "trial-a"
    assert evaluation.predictive.mean_rank_ic > 0.9
    assert evaluation.deflated_sharpe is not None
    assert evaluation.pbo is not None
    assert evaluation.factor_attribution is not None
    assert len(evaluation.cost_stress) == 4
    assert len(evaluation.spread_stress) == 4
    assert len(evaluation.latency_stress) == 2

    report = write_evaluation_report(
        [evaluation],
        rows,
        tmp_path / "evaluation-report",
        metadata={"dataset_id": dataset.dataset_id, "split_id": dataset.split_id},
    )
    verified = verify_evaluation_report(report.path)
    assert verified["leaderboard"][0]["trial_id"] == "trial-a"
