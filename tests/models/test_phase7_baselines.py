"""CPU acceptance gate for all Phase 7 baseline model families."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from trading_bot.config import ObjectiveConfig
from trading_bot.evaluation import (
    EvaluationAssumptions,
    TrialEvaluationInputs,
    build_leaderboard,
    evaluate_trial,
    positions_from_explicit_weights,
    read_prediction_artifact,
    verify_evaluation_report,
    write_evaluation_report,
)
from trading_bot.evaluation.artifacts import PredictionDataset
from trading_bot.models import (
    BaselineMLPModel,
    BaselineSplit,
    BaselineTargetNames,
    CausalTransformerReturnModel,
    ClassicalBaseline,
    ClassicalCheckpointError,
    ClassicalCheckpointIdentity,
    ElasticNetBaseline,
    GRUReturnModel,
    LSTMReturnModel,
    LightGBMBaseline,
    LogisticDirectionBaseline,
    RidgeBaseline,
    TCNReturnModel,
    XGBoostBaseline,
    build_baseline_loss,
    restore_classical_checkpoint,
    save_classical_checkpoint,
)
from trading_bot.training import (
    CheckpointIdentity,
    CheckpointManager,
    Trainer,
    TrainingBatch,
    benchmark_inference,
    count_parameters,
    predict_records,
    write_prediction_artifact,
)
from trading_bot.training.contracts import TradingModel

_TARGETS = BaselineTargetNames(return_target="return_15m", direction_target="direction_15m")


@dataclass(frozen=True, slots=True)
class FamilyResult:
    trial_id: str
    prediction_path: Path
    parameter_count: int
    samples_per_second: float


@dataclass(frozen=True, slots=True)
class Phase7GateResult:
    families: tuple[FamilyResult, ...]
    leaderboard_trial_ids: tuple[str, ...]
    report_path: Path


def _return_objective() -> ObjectiveConfig:
    return ObjectiveConfig(kind="excess_return", horizons_minutes=(15,), loss="mse")


def _direction_objective() -> ObjectiveConfig:
    return ObjectiveConfig(kind="direction", horizons_minutes=(15,), loss="bce")


def _make_split() -> BaselineSplit:
    batches: list[TrainingBatch] = []
    assets = torch.linspace(-1.0, 1.0, 8)
    sequence_index = torch.arange(4, dtype=torch.float32)
    base_timestamp = 1_767_225_600_000_000_000
    for batch_index in range(10):
        feature_0 = (
            assets[:, None] + batch_index * 0.03 + sequence_index[None, :] * 0.01
        )
        feature_1 = (
            -0.45 * assets[:, None]
            + 0.02 * batch_index
            - 0.005 * sequence_index[None, :]
        )
        feature_2 = torch.sin(
            assets[:, None] * 1.5 + sequence_index[None, :] * 0.2 + batch_index * 0.1
        )
        features = torch.stack((feature_0, feature_1, feature_2), dim=-1)
        future_return = (
            0.02 * feature_0[:, -1]
            - 0.015 * feature_1[:, -1]
            + 0.004 * feature_2.mean(dim=1)
        )
        timestamp = base_timestamp + batch_index * 86_400_000_000_000
        batches.append(
            TrainingBatch(
                features=features,
                targets={
                    _TARGETS.return_target: future_return,
                    _TARGETS.direction_target: (future_return > 0).float(),
                },
                asset_ids=tuple(f"asset-{index}" for index in range(8)),
                timestamps_ns=torch.full((8,), timestamp, dtype=torch.int64),
            )
        )
    return BaselineSplit(
        train_batches=tuple(batches[:6]),
        validation_batches=tuple(batches[6:]),
        dataset_id="phase7-rehearsal-v1",
        split_id="phase7-walk-forward-v1",
    )


def _classical_factories() -> tuple[
    tuple[str, ObjectiveConfig, Callable[[ObjectiveConfig], ClassicalBaseline]], ...
]:
    return (
        ("ridge", _return_objective(), lambda objective: RidgeBaseline(objective)),
        (
            "elastic_net",
            _return_objective(),
            lambda objective: ElasticNetBaseline(objective, alpha=0.001, l1_ratio=0.5),
        ),
        (
            "logistic_direction",
            _direction_objective(),
            lambda objective: LogisticDirectionBaseline(objective),
        ),
        (
            "lightgbm",
            _return_objective(),
            lambda objective: LightGBMBaseline(objective, n_estimators=16, max_depth=3),
        ),
        (
            "xgboost",
            _return_objective(),
            lambda objective: XGBoostBaseline(objective, n_estimators=16, max_depth=3),
        ),
    )


def _neural_factories() -> tuple[tuple[str, Callable[[], TradingModel]], ...]:
    return (
        ("mlp", lambda: BaselineMLPModel(3, hidden_features=16)),
        ("gru", lambda: GRUReturnModel(3, hidden_features=12)),
        ("lstm", lambda: LSTMReturnModel(3, hidden_features=12)),
        ("tcn", lambda: TCNReturnModel(3, hidden_features=12, kernel_size=3)),
        (
            "causal_transformer",
            lambda: CausalTransformerReturnModel(
                3,
                model_features=16,
                num_heads=4,
                num_layers=1,
                feedforward_features=32,
                max_sequence_length=8,
            ),
        ),
    )


def _rank_weights(dataset: PredictionDataset) -> dict[tuple[str, int], float]:
    by_timestamp: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for record in dataset.records:
        if record.rank_score is None:
            raise AssertionError("Phase 7 gate prediction must provide rank_score")
        by_timestamp[record.timestamp_ns].append((record.asset_id, record.rank_score))
    weights: dict[tuple[str, int], float] = {}
    for timestamp, values in by_timestamp.items():
        ordered = sorted(values, key=lambda item: (item[1], item[0]))
        if len(ordered) < 4:
            raise AssertionError("Phase 7 rehearsal timestamp requires at least four assets")
        for asset_id, _ in ordered:
            weights[(asset_id, timestamp)] = 0.0
        for asset_id, _ in ordered[:2]:
            weights[(asset_id, timestamp)] = -0.25
        for asset_id, _ in ordered[-2:]:
            weights[(asset_id, timestamp)] = 0.25
    return weights


@pytest.fixture(scope="module")
def phase7_gate(tmp_path_factory: pytest.TempPathFactory) -> Phase7GateResult:
    root = tmp_path_factory.mktemp("phase7-baselines")
    split = _make_split()
    family_results: list[FamilyResult] = []

    for family, objective, factory in _classical_factories():
        model = factory(objective)
        model.fit(split.train_batches, targets=_TARGETS)
        complexity = model.complexity()
        benchmark = model.benchmark_inference(split.validation_batches, warmup=1, iterations=2)
        assert complexity.learned_scalar_count > 0
        assert complexity.serialized_bytes > 0
        assert benchmark.samples_per_second > 0

        identity = ClassicalCheckpointIdentity(
            model_config_hash=f"{family}-config",
            dataset_id=split.dataset_id,
            split_id=split.split_id,
        )
        checkpoint = save_classical_checkpoint(
            model,
            root / family / "checkpoint",
            identity=identity,
        )
        resumed = factory(objective)
        restore_classical_checkpoint(resumed, checkpoint, expected_identity=identity)
        before = model.predict_records(split.validation_batches, targets=_TARGETS)
        after = resumed.predict_records(split.validation_batches, targets=_TARGETS)
        assert [record.rank_score for record in after] == pytest.approx(
            [record.rank_score for record in before]
        )

        prediction = write_prediction_artifact(
            after,
            root / family / "predictions",
            dataset_id=split.dataset_id,
            split_id=split.split_id,
            model_config_hash=identity.model_config_hash,
            checkpoint_id="fitted-state",
            target_name=_TARGETS.return_target,
        )
        family_results.append(
            FamilyResult(
                trial_id=family,
                prediction_path=prediction.path,
                parameter_count=complexity.learned_scalar_count,
                samples_per_second=benchmark.samples_per_second,
            )
        )

    objective = _return_objective()
    loss_fn = build_baseline_loss(objective, _TARGETS)
    for family, factory in _neural_factories():
        gradient_model = factory()
        gradient_loss = loss_fn(gradient_model(split.train_batches[0]), split.train_batches[0])
        gradient_loss.backward()
        gradients = [parameter.grad for parameter in gradient_model.parameters()]
        assert gradients and all(
            gradient is not None and bool(torch.isfinite(gradient).all().item())
            for gradient in gradients
        )

        model = factory()
        parameter_count = count_parameters(model, trainable_only=True)
        benchmark = benchmark_inference(
            model,
            split.validation_batches[0],
            warmup=1,
            iterations=2,
        )
        assert parameter_count > 0
        assert benchmark.samples_per_second > 0

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            max_steps=4,
            precision="fp32",
            seed=31,
        )
        state = trainer.fit(split.train_batches[:2])
        identity = CheckpointIdentity(
            model_config_hash=f"{family}-config",
            training_config_hash="phase7-training-config",
            dataset_id=split.dataset_id,
            split_id=split.split_id,
        )
        manager = CheckpointManager(root / family / "checkpoints")
        manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            training_state=state,
            identity=identity,
            precision="fp32",
        )

        resumed_model = factory()
        resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=0.01)
        resumed_trainer = Trainer(
            model=resumed_model,
            optimizer=resumed_optimizer,
            loss_fn=loss_fn,
            max_steps=4,
            precision="fp32",
            seed=31,
        )
        restored = manager.restore(
            "latest",
            model=resumed_model,
            optimizer=resumed_optimizer,
            scheduler=None,
            expected_identity=identity,
        )
        final_state = resumed_trainer.fit(
            split.train_batches[2:4],
            initial_state=restored.training_state,
        )
        assert final_state.optimizer_step == 4
        records = predict_records(
            resumed_model,
            split.validation_batches,
            target_name=_TARGETS.return_target,
        )
        prediction = write_prediction_artifact(
            records,
            root / family / "predictions",
            dataset_id=split.dataset_id,
            split_id=split.split_id,
            model_config_hash=identity.model_config_hash,
            checkpoint_id=f"step-{final_state.optimizer_step:08d}",
            target_name=_TARGETS.return_target,
        )
        family_results.append(
            FamilyResult(
                trial_id=family,
                prediction_path=prediction.path,
                parameter_count=parameter_count,
                samples_per_second=benchmark.samples_per_second,
            )
        )

    assumptions = EvaluationAssumptions(
        fee_bps=0.5,
        spread_bps=1.0,
        slippage_bps=0.5,
        impact_bps=0.25,
    )
    evaluations = []
    for result in family_results:
        dataset = read_prediction_artifact(result.prediction_path)
        positions = positions_from_explicit_weights(dataset, _rank_weights(dataset))
        evaluations.append(
            evaluate_trial(
                TrialEvaluationInputs(
                    trial_id=result.trial_id,
                    predictions=dataset,
                    positions=positions,
                    fold_returns={
                        "fold-a": (0.010, 0.012, 0.009, 0.011),
                        "fold-b": (0.008, 0.010, 0.013, 0.009),
                    },
                ),
                assumptions,
            )
        )
    leaderboard = build_leaderboard(evaluations)
    report = write_evaluation_report(
        evaluations,
        leaderboard,
        root / "baseline-leaderboard",
        metadata={"dataset_id": split.dataset_id, "split_id": split.split_id},
    )
    verified = verify_evaluation_report(report.path)
    assert len(verified["leaderboard"]) == len(family_results)
    return Phase7GateResult(
        families=tuple(family_results),
        leaderboard_trial_ids=tuple(row.trial_id for row in leaderboard),
        report_path=report.path,
    )


def test_all_phase7_baseline_families_complete_common_cpu_gate(
    phase7_gate: Phase7GateResult,
) -> None:
    expected = {
        "ridge",
        "elastic_net",
        "logistic_direction",
        "lightgbm",
        "xgboost",
        "mlp",
        "gru",
        "lstm",
        "tcn",
        "causal_transformer",
    }
    assert {result.trial_id for result in phase7_gate.families} == expected
    assert all(result.parameter_count > 0 for result in phase7_gate.families)
    assert all(result.samples_per_second > 0 for result in phase7_gate.families)
    assert all(result.prediction_path.is_dir() for result in phase7_gate.families)


def test_phase7_rehearsal_baseline_leaderboard_contains_every_family(
    phase7_gate: Phase7GateResult,
) -> None:
    assert len(phase7_gate.leaderboard_trial_ids) == 10
    assert len(set(phase7_gate.leaderboard_trial_ids)) == 10
    assert phase7_gate.report_path.is_dir()


def test_shared_neural_objective_adapter_covers_direction_ranking_and_multitask() -> None:
    batch = _make_split().train_batches[0]
    model = BaselineMLPModel(3, hidden_features=8)
    output = model(batch)
    objectives = (
        _direction_objective(),
        ObjectiveConfig(kind="ranking", horizons_minutes=(15,), loss="pairwise_rank"),
        ObjectiveConfig(
            kind="multitask",
            horizons_minutes=(15,),
            loss="composite",
            task_weights={
                "expected_return": 1.0,
                "rank_score": 0.5,
                "direction_probability": 0.25,
            },
        ),
    )
    for objective in objectives:
        loss = build_baseline_loss(objective, _TARGETS)(output, batch)
        assert loss.ndim == 0
        assert bool(torch.isfinite(loss).item())


def test_classical_checkpoint_rejects_tampering(tmp_path: Path) -> None:
    split = _make_split()
    objective = _return_objective()
    model = RidgeBaseline(objective)
    model.fit(split.train_batches, targets=_TARGETS)
    identity = ClassicalCheckpointIdentity(
        model_config_hash="ridge-config",
        dataset_id=split.dataset_id,
        split_id=split.split_id,
    )
    checkpoint = save_classical_checkpoint(model, tmp_path / "checkpoint", identity=identity)
    state_path = checkpoint / "state.pkl"
    payload = bytearray(state_path.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    state_path.write_bytes(payload)
    with pytest.raises(ClassicalCheckpointError, match="checksum/size mismatch"):
        restore_classical_checkpoint(
            RidgeBaseline(objective),
            checkpoint,
            expected_identity=identity,
        )


def test_classical_baselines_reject_incompatible_objectives() -> None:
    with pytest.raises(ValueError, match="regression baseline"):
        RidgeBaseline(_direction_objective())
    with pytest.raises(ValueError, match="logistic baseline"):
        LogisticDirectionBaseline(_return_objective())
