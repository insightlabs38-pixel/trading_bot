from __future__ import annotations

import json
from pathlib import Path

import pytest

from trading_bot.evaluation import (
    EvaluationAssumptions,
    PositionPoint,
    PredictionPoint,
    TrialEvaluation,
    ValidityEvidence,
    build_leaderboard,
    rank_ic_metrics,
    robustness_metrics,
    run_backtest,
    validity_status,
    verify_evaluation_report,
    write_evaluation_report,
)


def _trial(trial_id: str, *, score_flip: bool = False, valid: bool = True) -> TrialEvaluation:
    predictions = []
    for timestamp in (1, 2, 3):
        for index, target in enumerate((0.01, 0.02, 0.03, 0.04)):
            score = -target if score_flip else target
            predictions.append(
                PredictionPoint(
                    asset_id=f"a{index}",
                    timestamp_ns=timestamp,
                    target=target,
                    score=score,
                    fold_id="f1",
                    regime="calm",
                    horizon="15m",
                    sector="mixed",
                )
            )
    predictive = rank_ic_metrics(predictions)
    positions = [
        PositionPoint("A", 1_700_000_000_000_000_000, 1.0, 0.01),
        PositionPoint("A", 1_700_086_400_000_000_000, 1.0, 0.02),
        PositionPoint("A", 1_700_172_800_000_000_000, 1.0, -0.005),
    ]
    backtest = run_backtest(positions, EvaluationAssumptions())
    robustness = robustness_metrics({"f1": [0.01, 0.02, -0.005]})
    evidence = ValidityEvidence(data_leakage_free=valid)
    validity = validity_status(
        evidence,
        robustness=robustness,
        minimum_positive_fold_fraction=0.7,
        economic=backtest.economic,
    )
    return TrialEvaluation(
        trial_id=trial_id,
        predictive=predictive,
        economic=backtest.economic,
        friction=backtest.friction,
        robustness=robustness,
        validity=validity,
        attempted_trial_count=3,
    )


def test_leaderboard_uses_explicit_hierarchy_and_disqualifies_invalid_trial(tmp_path: Path) -> None:
    strong = _trial("strong")
    weak = _trial("weak", score_flip=True)
    invalid = _trial("invalid", valid=False)
    rows = build_leaderboard([weak, invalid, strong])
    assert [row.trial_id for row in rows] == ["strong", "weak", "invalid"]
    assert rows[0].eligible is True
    assert rows[-1].eligible is False
    assert "data_leakage_or_timestamp_causality_failure" in rows[-1].disqualification_reasons

    result = write_evaluation_report(
        [strong, weak, invalid],
        rows,
        tmp_path / "report",
        metadata={"dataset": "synthetic", "split": "walk-forward"},
    )
    payload = verify_evaluation_report(result.path)
    assert payload["leaderboard"][0]["trial_id"] == "strong"
    markdown = (result.path / "report.md").read_text()
    assert "Canonical Evaluation Leaderboard" in markdown
    assert "data_leakage_or_timestamp_causality_failure" in markdown

    second = write_evaluation_report(
        [strong, weak, invalid],
        rows,
        tmp_path / "report-2",
        metadata={"split": "walk-forward", "dataset": "synthetic"},
    )
    assert (result.path / "report.json").read_bytes() == (second.path / "report.json").read_bytes()
    assert result.json_sha256 == second.json_sha256


def test_report_detects_tampering(tmp_path: Path) -> None:
    trial = _trial("trial")
    rows = build_leaderboard([trial])
    result = write_evaluation_report([trial], rows, tmp_path / "report")
    path = result.path / "report.json"
    payload = json.loads(path.read_text())
    payload["schema_version"] = 99
    path.write_text(json.dumps(payload))
    with pytest.raises(Exception, match="checksum mismatch"):
        verify_evaluation_report(result.path)
