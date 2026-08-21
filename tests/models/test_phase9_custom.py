"""CPU/reference acceptance gate for Phase 9 custom architectures and temporal math."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import nn

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
    CORE_CUSTOM_ARCHITECTURES,
    BaselineSplit,
    BaselineTargetNames,
    CustomArchitecture,
    CustomScale,
    HeterogeneousMoEReturnModel,
    MarketMixerAblations,
    MultiDecayTemporalOperator,
    TritonUnavailableError,
    benchmark_temporal_operator,
    build_baseline_loss,
    build_custom_model,
    custom_model_spec,
    market_mixer_ablation_suite,
    multi_decay_dispatch,
    multi_decay_reference,
    profile_custom_model,
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
class CustomFamilyResult:
    trial_id: str
    prediction_path: Path
    parameter_count: int
    active_parameter_upper_bound: int
    state_bytes: int
    samples_per_second: float


@dataclass(frozen=True, slots=True)
class Phase9GateResult:
    families: tuple[CustomFamilyResult, ...]
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
        feature_0 = assets[:, None] + batch_index * 0.021 + sequence_index[None, :] * 0.009
        feature_1 = -0.35 * assets[:, None] + batch_index * 0.013
        feature_1 = feature_1 - sequence_index[None, :] * 0.003
        feature_2 = torch.sin(
            assets[:, None] * 1.2 + sequence_index[None, :] * 0.20 + batch_index * 0.07
        )
        features = torch.stack((feature_0, feature_1, feature_2), dim=-1)
        future_return = (
            0.019 * feature_0[:, -1] - 0.012 * feature_1[:, -1] + 0.006 * feature_2.mean(dim=1)
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
        dataset_id="phase9-rehearsal-v1",
        split_id="phase9-walk-forward-v1",
    )


def _rank_weights(dataset: PredictionDataset) -> dict[tuple[str, int], float]:
    by_timestamp: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for record in dataset.records:
        if record.rank_score is None:
            raise AssertionError("Phase 9 gate prediction must provide rank_score")
        by_timestamp[record.timestamp_ns].append((record.asset_id, record.rank_score))
    weights: dict[tuple[str, int], float] = {}
    for timestamp, values in by_timestamp.items():
        ordered = sorted(values, key=lambda item: (item[1], item[0]))
        if len(ordered) < 4:
            raise AssertionError("Phase 9 rehearsal timestamp requires at least four assets")
        for asset_id, _ in ordered:
            weights[(asset_id, timestamp)] = 0.0
        for asset_id, _ in ordered[:2]:
            weights[(asset_id, timestamp)] = -0.25
        for asset_id, _ in ordered[-2:]:
            weights[(asset_id, timestamp)] = 0.25
    return weights


def _small_spec(architecture: CustomArchitecture):
    return custom_model_spec(
        architecture,
        "small",
        input_features=3,
        max_sequence_length=8,
    )


def _assert_present_gradients_are_finite(model: nn.Module) -> None:
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)


@pytest.fixture(scope="module")
def phase9_gate(tmp_path_factory: pytest.TempPathFactory) -> Phase9GateResult:
    root = tmp_path_factory.mktemp("phase9-custom")
    split = _make_split()
    family_results: list[CustomFamilyResult] = []
    return_loss = build_baseline_loss(_return_objective(), _TARGETS)
    gradient_loss = build_baseline_loss(_multitask_objective(), _TARGETS)

    for architecture in CORE_CUSTOM_ARCHITECTURES:
        torch.manual_seed(91)
        spec = _small_spec(architecture)
        gradient_model = build_custom_model(spec)
        output = gradient_model(split.train_batches[0])
        output.validate(split.train_batches[0].batch_size)
        loss = gradient_loss(output, split.train_batches[0])
        loss = loss + 0.01 * output.require("volatility").mean()
        loss = loss + 0.01 * output.require("uncertainty").mean()
        loss.backward()
        _assert_present_gradients_are_finite(gradient_model)

        torch.manual_seed(91)
        model = build_custom_model(spec)
        profile = profile_custom_model(model)
        benchmark = benchmark_inference(
            model,
            split.validation_batches[0],
            warmup=1,
            iterations=2,
        )
        assert profile.parameter_count > 0
        assert profile.trainable_parameter_count == profile.parameter_count
        assert 0 < profile.active_parameter_upper_bound <= profile.parameter_count
        assert profile.total_state_bytes > 0
        assert benchmark.samples_per_second > 0

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=return_loss,
            max_steps=4,
            precision="fp32",
            seed=53,
        )
        state = trainer.fit(split.train_batches[:2])
        identity = CheckpointIdentity(
            model_config_hash=f"phase9-{architecture}-small",
            training_config_hash="phase9-training-config",
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

        torch.manual_seed(91)
        resumed_model = build_custom_model(spec)
        resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=0.005)
        resumed_trainer = Trainer(
            model=resumed_model,
            optimizer=resumed_optimizer,
            loss_fn=return_loss,
            max_steps=4,
            precision="fp32",
            seed=53,
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
            CustomFamilyResult(
                trial_id=architecture,
                prediction_path=prediction.path,
                parameter_count=profile.parameter_count,
                active_parameter_upper_bound=profile.active_parameter_upper_bound,
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
        root / "custom-leaderboard",
        metadata={"dataset_id": split.dataset_id, "split_id": split.split_id},
    )
    verified = verify_evaluation_report(report.path)
    assert len(verified["leaderboard"]) == len(family_results)
    return Phase9GateResult(
        families=tuple(family_results),
        leaderboard_trial_ids=tuple(row.trial_id for row in leaderboard),
        report_path=report.path,
    )


def test_multi_decay_reference_matches_hand_calculation() -> None:
    sequence = torch.tensor([[[2.0], [4.0]]], dtype=torch.float64)
    decay_probabilities = torch.tensor([[0.25], [0.75]], dtype=torch.float64)
    decay_logits = torch.logit(decay_probabilities)
    mix_logits = torch.log(torch.tensor([[0.25], [0.75]], dtype=torch.float64))
    actual = multi_decay_reference(sequence, decay_logits, mix_logits)
    expected = torch.tensor([[[0.75], [1.875]]], dtype=torch.float64)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_multi_decay_reference_is_differentiable() -> None:
    sequence = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
    decay_logits = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    mix_logits = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        multi_decay_reference,
        (sequence, decay_logits, mix_logits),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )


def test_multi_decay_operator_is_causal_and_accepts_strided_input() -> None:
    torch.manual_seed(7)
    source = torch.randn(2, 8, 6)
    strided = source[:, :, ::2]
    assert not strided.is_contiguous()
    operator = MultiDecayTemporalOperator(3, num_decays=3, backend="reference")

    baseline = operator(strided)
    perturbed = strided.clone()
    perturbed[:, 5:] = perturbed[:, 5:] + 100.0
    changed = operator(perturbed)
    torch.testing.assert_close(baseline[:, :5], changed[:, :5], rtol=1e-5, atol=1e-6)


def test_multi_decay_dispatch_fails_closed_for_unvalidated_triton() -> None:
    sequence = torch.randn(2, 4, 3)
    decay_logits = torch.zeros(2, 3)
    mix_logits = torch.zeros(2, 3)
    reference = multi_decay_reference(sequence, decay_logits, mix_logits)
    automatic = multi_decay_dispatch(
        sequence,
        decay_logits,
        mix_logits,
        backend="auto",
    )
    torch.testing.assert_close(automatic, reference)
    with pytest.raises(TritonUnavailableError, match="intentionally unavailable"):
        multi_decay_dispatch(
            sequence,
            decay_logits,
            mix_logits,
            backend="triton",
        )


def test_temporal_operator_cpu_profile_reports_reference_state() -> None:
    operator = MultiDecayTemporalOperator(4, num_decays=3)
    result = benchmark_temporal_operator(
        operator,
        torch.randn(4, 8, 4),
        warmup=1,
        iterations=2,
    )
    assert result.backend == "reference"
    assert result.samples_per_second > 0
    assert result.state_bytes > 0


def test_market_mixer_ablation_matrix_disables_each_major_component() -> None:
    batch = _make_split().validation_batches[0]
    cases = market_mixer_ablation_suite()
    assert tuple(case.name for case in cases) == (
        "full",
        "no_short",
        "no_long",
        "no_gated_fusion",
        "no_cross_sectional",
        "no_market_context",
    )
    parameter_counts = []
    for case in cases:
        model = build_custom_model(
            custom_model_spec(
                "market_mixer",
                "small",
                input_features=3,
                max_sequence_length=8,
                market_mixer_ablations=case.ablations,
            )
        )
        output = model(batch)
        output.validate(batch.batch_size)
        parameter_counts.append(profile_custom_model(model).parameter_count)
    assert len(set(parameter_counts)) >= 4


def test_market_mixer_cross_sectional_components_fail_closed_on_mixed_time() -> None:
    source = _make_split().validation_batches[0]
    mixed = TrainingBatch(
        features=source.features,
        targets=source.targets,
        asset_ids=source.asset_ids,
        timestamps_ns=source.timestamps_ns + torch.arange(source.batch_size, dtype=torch.int64),
    )
    full_model = build_custom_model(_small_spec("market_mixer"))
    with pytest.raises(ValueError, match="one decision timestamp"):
        full_model(mixed)

    temporal_only = build_custom_model(
        custom_model_spec(
            "market_mixer",
            "small",
            input_features=3,
            max_sequence_length=8,
            market_mixer_ablations=MarketMixerAblations(
                cross_sectional=False,
                market_context=False,
            ),
        )
    )
    temporal_only(mixed).validate(mixed.batch_size)


def test_heterogeneous_moe_reports_sparse_router_utilization() -> None:
    batch = _make_split().validation_batches[0]
    model = build_custom_model(_small_spec("heterogeneous_moe"))
    assert isinstance(model, HeterogeneousMoEReturnModel)
    with pytest.raises(RuntimeError, match="before the first forward"):
        model.router_diagnostics()
    model(batch).validate(batch.batch_size)
    diagnostics = model.router_diagnostics()
    assert diagnostics.expert_names == (
        "local_tcn",
        "long_memory",
        "temporal_attention",
        "frequency",
    )
    assert sum(diagnostics.assignment_counts) == batch.batch_size * diagnostics.top_k
    assert sum(diagnostics.mean_weights) == pytest.approx(1.0, abs=1e-6)
    profile = profile_custom_model(model)
    assert profile.active_parameter_upper_bound < profile.parameter_count


def test_heterogeneous_moe_frequency_expert_is_optional() -> None:
    batch = _make_split().validation_batches[0]
    model = build_custom_model(
        custom_model_spec(
            "heterogeneous_moe",
            "small",
            input_features=3,
            max_sequence_length=8,
            include_frequency_expert=False,
        )
    )
    assert isinstance(model, HeterogeneousMoEReturnModel)
    model(batch).validate(batch.batch_size)
    assert model.router_diagnostics().expert_names == (
        "local_tcn",
        "long_memory",
        "temporal_attention",
    )


@pytest.mark.parametrize("architecture", CORE_CUSTOM_ARCHITECTURES)
def test_small_medium_large_custom_specs_scale_and_keep_common_heads(
    architecture: CustomArchitecture,
) -> None:
    batch = _make_split().validation_batches[0]
    parameter_counts: list[int] = []
    for scale in ("small", "medium", "large"):
        typed_scale: CustomScale = scale
        model = build_custom_model(
            custom_model_spec(
                architecture,
                typed_scale,
                input_features=3,
                max_sequence_length=8,
            )
        )
        output = model(batch)
        output.validate(batch.batch_size)
        assert set(output.tensors()) == {
            "expected_return",
            "rank_score",
            "direction_probability",
            "volatility",
            "uncertainty",
        }
        parameter_counts.append(profile_custom_model(model).parameter_count)
    assert parameter_counts[0] < parameter_counts[1] < parameter_counts[2]


def test_all_cpu_reference_custom_families_complete_common_gate(
    phase9_gate: Phase9GateResult,
) -> None:
    assert {result.trial_id for result in phase9_gate.families} == set(CORE_CUSTOM_ARCHITECTURES)
    assert all(result.parameter_count > 0 for result in phase9_gate.families)
    assert all(result.state_bytes > 0 for result in phase9_gate.families)
    assert all(result.samples_per_second > 0 for result in phase9_gate.families)
    assert all(result.prediction_path.is_dir() for result in phase9_gate.families)


def test_phase9_rehearsal_leaderboard_contains_every_custom_family(
    phase9_gate: Phase9GateResult,
) -> None:
    assert len(phase9_gate.leaderboard_trial_ids) == len(CORE_CUSTOM_ARCHITECTURES)
    assert len(set(phase9_gate.leaderboard_trial_ids)) == len(CORE_CUSTOM_ARCHITECTURES)
    assert phase9_gate.report_path.is_dir()
