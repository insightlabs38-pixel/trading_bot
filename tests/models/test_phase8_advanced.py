"""CPU/reference acceptance gate for Phase 8 advanced model families."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

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
    CORE_ADVANCED_ARCHITECTURES,
    AdvancedArchitecture,
    AdvancedScale,
    BaselineSplit,
    BaselineTargetNames,
    FoundationBackbone,
    FoundationModelIdentity,
    FrozenFoundationAdapter,
    advanced_model_spec,
    build_advanced_model,
    build_baseline_loss,
    profile_advanced_model,
)
from trading_bot.training import (
    CheckpointIdentity,
    CheckpointManager,
    Trainer,
    TrainingBatch,
    benchmark_inference,
    predict_records,
    write_prediction_artifact,
)

_TARGETS = BaselineTargetNames(return_target="return_15m", direction_target="direction_15m")


@dataclass(frozen=True, slots=True)
class AdvancedFamilyResult:
    trial_id: str
    prediction_path: Path
    parameter_count: int
    state_bytes: int
    samples_per_second: float


@dataclass(frozen=True, slots=True)
class Phase8GateResult:
    families: tuple[AdvancedFamilyResult, ...]
    leaderboard_trial_ids: tuple[str, ...]
    report_path: Path


def _return_objective() -> ObjectiveConfig:
    return ObjectiveConfig(kind="excess_return", horizons_minutes=(15,), loss="mse")


def _multitask_objective() -> ObjectiveConfig:
    return ObjectiveConfig(
        kind="multitask",
        horizons_minutes=(15,),
        loss="composite",
        task_weights={
            "expected_return": 1.0,
            "rank_score": 0.5,
            "direction_probability": 0.25,
        },
    )


def _make_split() -> BaselineSplit:
    batches: list[TrainingBatch] = []
    assets = torch.linspace(-1.0, 1.0, 8)
    sequence_index = torch.arange(8, dtype=torch.float32)
    base_timestamp = 1_767_225_600_000_000_000
    for batch_index in range(10):
        feature_0 = assets[:, None] + batch_index * 0.025 + sequence_index[None, :] * 0.008
        feature_1 = (
            -0.40 * assets[:, None]
            + batch_index * 0.015
            - sequence_index[None, :] * 0.004
        )
        feature_2 = torch.sin(
            assets[:, None] * 1.4 + sequence_index[None, :] * 0.18 + batch_index * 0.09
        )
        features = torch.stack((feature_0, feature_1, feature_2), dim=-1)
        future_return = (
            0.018 * feature_0[:, -1]
            - 0.013 * feature_1[:, -1]
            + 0.005 * feature_2.mean(dim=1)
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
        dataset_id="phase8-rehearsal-v1",
        split_id="phase8-walk-forward-v1",
    )


def _rank_weights(dataset: PredictionDataset) -> dict[tuple[str, int], float]:
    by_timestamp: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for record in dataset.records:
        if record.rank_score is None:
            raise AssertionError("Phase 8 gate prediction must provide rank_score")
        by_timestamp[record.timestamp_ns].append((record.asset_id, record.rank_score))
    weights: dict[tuple[str, int], float] = {}
    for timestamp, values in by_timestamp.items():
        ordered = sorted(values, key=lambda item: (item[1], item[0]))
        if len(ordered) < 4:
            raise AssertionError("Phase 8 rehearsal timestamp requires at least four assets")
        for asset_id, _ in ordered:
            weights[(asset_id, timestamp)] = 0.0
        for asset_id, _ in ordered[:2]:
            weights[(asset_id, timestamp)] = -0.25
        for asset_id, _ in ordered[-2:]:
            weights[(asset_id, timestamp)] = 0.25
    return weights


def _small_spec(architecture: AdvancedArchitecture):
    return advanced_model_spec(
        architecture,
        "small",
        input_features=3,
        max_sequence_length=8,
    )


def _assert_all_trainable_gradients_are_finite(model: nn.Module) -> None:
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(
        gradient is not None and bool(torch.isfinite(gradient).all().item())
        for gradient in gradients
    )


@pytest.fixture(scope="module")
def phase8_gate(tmp_path_factory: pytest.TempPathFactory) -> Phase8GateResult:
    root = tmp_path_factory.mktemp("phase8-advanced")
    split = _make_split()
    family_results: list[AdvancedFamilyResult] = []
    return_loss = build_baseline_loss(_return_objective(), _TARGETS)
    gradient_loss = build_baseline_loss(_multitask_objective(), _TARGETS)

    for architecture in CORE_ADVANCED_ARCHITECTURES:
        spec = _small_spec(architecture)
        gradient_model = build_advanced_model(spec)
        output = gradient_model(split.train_batches[0])
        output.validate(split.train_batches[0].batch_size)
        loss = gradient_loss(output, split.train_batches[0])
        loss = loss + 0.01 * output.require("volatility").mean()
        loss = loss + 0.01 * output.require("uncertainty").mean()
        loss.backward()
        _assert_all_trainable_gradients_are_finite(gradient_model)

        model = build_advanced_model(spec)
        profile = profile_advanced_model(model)
        benchmark = benchmark_inference(
            model,
            split.validation_batches[0],
            warmup=1,
            iterations=2,
        )
        assert profile.parameter_count > 0
        assert profile.trainable_parameter_count == profile.parameter_count
        assert profile.total_state_bytes > 0
        assert benchmark.samples_per_second > 0

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=return_loss,
            max_steps=4,
            precision="fp32",
            seed=41,
        )
        state = trainer.fit(split.train_batches[:2])
        identity = CheckpointIdentity(
            model_config_hash=f"phase8-{architecture}-small",
            training_config_hash="phase8-training-config",
            dataset_id=split.dataset_id,
            split_id=split.split_id,
        )
        manager = CheckpointManager(root / architecture / "checkpoints")
        manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            training_state=state,
            identity=identity,
            precision="fp32",
        )

        resumed_model = build_advanced_model(spec)
        resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=0.005)
        resumed_trainer = Trainer(
            model=resumed_model,
            optimizer=resumed_optimizer,
            loss_fn=return_loss,
            max_steps=4,
            precision="fp32",
            seed=41,
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
            root / architecture / "predictions",
            dataset_id=split.dataset_id,
            split_id=split.split_id,
            model_config_hash=identity.model_config_hash,
            checkpoint_id=f"step-{final_state.optimizer_step:08d}",
            target_name=_TARGETS.return_target,
        )
        family_results.append(
            AdvancedFamilyResult(
                trial_id=architecture,
                prediction_path=prediction.path,
                parameter_count=profile.parameter_count,
                state_bytes=profile.total_state_bytes,
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
        root / "advanced-leaderboard",
        metadata={"dataset_id": split.dataset_id, "split_id": split.split_id},
    )
    verified = verify_evaluation_report(report.path)
    assert len(verified["leaderboard"]) == len(family_results)
    return Phase8GateResult(
        families=tuple(family_results),
        leaderboard_trial_ids=tuple(row.trial_id for row in leaderboard),
        report_path=report.path,
    )


def test_all_cpu_reference_advanced_families_complete_common_gate(
    phase8_gate: Phase8GateResult,
) -> None:
    assert {result.trial_id for result in phase8_gate.families} == set(
        CORE_ADVANCED_ARCHITECTURES
    )
    assert all(result.parameter_count > 0 for result in phase8_gate.families)
    assert all(result.state_bytes > 0 for result in phase8_gate.families)
    assert all(result.samples_per_second > 0 for result in phase8_gate.families)
    assert all(result.prediction_path.is_dir() for result in phase8_gate.families)


def test_phase8_rehearsal_leaderboard_contains_every_core_family(
    phase8_gate: Phase8GateResult,
) -> None:
    assert len(phase8_gate.leaderboard_trial_ids) == len(CORE_ADVANCED_ARCHITECTURES)
    assert len(set(phase8_gate.leaderboard_trial_ids)) == len(CORE_ADVANCED_ARCHITECTURES)
    assert phase8_gate.report_path.is_dir()


@pytest.mark.parametrize("architecture", CORE_ADVANCED_ARCHITECTURES)
def test_small_medium_large_specs_have_consistent_heads_and_scaling(
    architecture: AdvancedArchitecture,
) -> None:
    batch = _make_split().validation_batches[0]
    parameter_counts: list[int] = []
    for scale in ("small", "medium", "large"):
        typed_scale: AdvancedScale = scale
        spec = advanced_model_spec(
            architecture,
            typed_scale,
            input_features=3,
            max_sequence_length=8,
        )
        model = build_advanced_model(spec)
        output = model(batch)
        output.validate(batch.batch_size)
        assert set(output.tensors()) == {
            "expected_return",
            "rank_score",
            "direction_probability",
            "volatility",
            "uncertainty",
        }
        parameter_counts.append(profile_advanced_model(model).parameter_count)
    assert parameter_counts[0] < parameter_counts[1] < parameter_counts[2]


def test_cross_sectional_families_fail_closed_on_mixed_timestamps() -> None:
    source = _make_split().validation_batches[0]
    mixed = TrainingBatch(
        features=source.features,
        targets=source.targets,
        asset_ids=source.asset_ids,
        timestamps_ns=source.timestamps_ns + torch.arange(source.batch_size, dtype=torch.int64),
    )
    for architecture in (
        "temporal_cross_sectional_transformer",
        "temporal_graph",
    ):
        model = build_advanced_model(_small_spec(architecture))
        with pytest.raises(ValueError, match="one decision timestamp"):
            model(mixed)


class _TinyFoundationBackbone(FoundationBackbone):
    output_features = 6

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(3, self.output_features)

    def forward(self, sequence: Tensor) -> Tensor:
        return torch.tanh(self.encoder(sequence))


def test_frozen_foundation_adapter_is_offline_typed_and_head_trainable() -> None:
    identity = FoundationModelIdentity(
        provider="synthetic-test",
        model_id="local-foundation-fixture",
        revision="fixture-v1",
        checkpoint_sha256="0" * 64,
    )
    backbone = _TinyFoundationBackbone()
    adapter = FrozenFoundationAdapter(
        backbone,
        identity,
        input_features=3,
        model_features=12,
    )
    batch = _make_split().train_batches[0]
    output = adapter(batch)
    output.validate(batch.batch_size)
    loss = build_baseline_loss(_multitask_objective(), _TARGETS)(output, batch)
    loss = loss + 0.01 * output.require("volatility").mean()
    loss = loss + 0.01 * output.require("uncertainty").mean()
    loss.backward()

    assert all(not parameter.requires_grad for parameter in backbone.parameters())
    assert all(parameter.grad is None for parameter in backbone.parameters())
    adapter_gradients = [
        parameter.grad
        for name, parameter in adapter.named_parameters()
        if not name.startswith("backbone.") and parameter.requires_grad
    ]
    assert adapter_gradients
    assert all(
        gradient is not None and bool(torch.isfinite(gradient).all().item())
        for gradient in adapter_gradients
    )
    profile = profile_advanced_model(adapter)
    assert 0 < profile.trainable_parameter_count < profile.parameter_count


def test_foundation_identity_rejects_unverified_checkpoint_digest() -> None:
    with pytest.raises(ValueError, match="64-character hex digest"):
        FoundationModelIdentity(
            provider="provider",
            model_id="model",
            revision="revision",
            checkpoint_sha256="not-a-checksum",
        )
