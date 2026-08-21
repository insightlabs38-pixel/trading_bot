"""CPU acceptance gate for the Phase 10 experiment registry and search manifest."""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
import torch
import yaml  # type: ignore[import-untyped]

from trading_bot.campaign import (
    CampaignSearchManifest,
    architecture_by_id,
    campaign_manifest_canonical_json,
    campaign_manifest_sha256,
    enumerate_campaign,
    load_campaign_search_manifest,
    objective_by_id,
    scale_parameters,
)
from trading_bot.campaign.search_space import ScalePreset
from trading_bot.models import (
    AdvancedArchitecture,
    AdvancedScale,
    BaselineMLPModel,
    CausalTransformerReturnModel,
    CustomArchitecture,
    CustomScale,
    GRUReturnModel,
    LSTMReturnModel,
    TCNReturnModel,
    advanced_model_spec,
    custom_model_spec,
)
from trading_bot.training import TrainingBatch, count_parameters

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_PATH = _REPO_ROOT / "configs" / "campaigns" / "h200_tournament_v1.yaml"
_ADVANCED = (
    "patchtst",
    "itransformer",
    "mamba_reference",
    "vsn_lstm",
    "temporal_cross_sectional_transformer",
    "temporal_graph",
)
_CUSTOM = ("market_mixer", "heterogeneous_moe")
_SCALES = ("small", "medium", "large")


def _manifest() -> CampaignSearchManifest:
    return load_campaign_search_manifest(_MANIFEST_PATH)


def test_frozen_manifest_enumerates_documented_campaign() -> None:
    manifest = _manifest()
    enumeration = enumerate_campaign(manifest)

    assert manifest.frozen is True
    assert manifest.campaign_id == "h200-tournament-v1"
    assert manifest.screening_objective_id == "excess_huber_15m"
    assert len(manifest.architectures) == 19
    assert len(enumeration.mandatory_families) == 17
    assert set(enumeration.optional_families) == {
        "logistic_direction",
        "foundation_adapter",
    }
    assert len(enumeration.searchable_families) == 13
    assert enumeration.canonical_scale_count == 45
    assert enumeration.screening_candidate_points == 3159
    assert enumeration.planned_fit_count == 123
    assert enumeration.enabled_objectives == (
        "excess_mse_15m",
        "excess_huber_15m",
        "ranking_15m",
        "direction_bce_15m",
        "multitask_return_rank_direction_15m",
    )
    assert enumeration.planned_objectives == (
        "multitask_return_rank_vol_direction_15m",
        "multi_horizon_huber_15_30m",
        "distributional_quantile_15m",
    )


def test_manifest_hash_and_canonical_json_are_stable() -> None:
    first = _manifest()
    second = _manifest()

    assert campaign_manifest_canonical_json(first) == campaign_manifest_canonical_json(second)
    digest = campaign_manifest_sha256(first)
    assert digest == campaign_manifest_sha256(second)
    assert len(digest) == 64
    int(digest, 16)


def test_campaign_loader_does_not_import_torch() -> None:
    command = (
        "import sys; import trading_bot.campaign; "
        "loaded=sorted(name for name in sys.modules if name.startswith('torch')); "
        "assert not loaded, loaded"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("architecture", _ADVANCED)
@pytest.mark.parametrize("scale", _SCALES)
def test_advanced_yaml_presets_match_phase8_reference(
    architecture: str,
    scale: str,
) -> None:
    manifest = _manifest()
    reference = advanced_model_spec(
        cast(AdvancedArchitecture, architecture),
        cast(AdvancedScale, scale),
        input_features=3,
        max_sequence_length=64,
    )
    parameters = scale_parameters(
        manifest,
        architecture,
        cast(ScalePreset, scale),
    )
    for name, expected in parameters.items():
        assert getattr(reference, name) == expected


@pytest.mark.parametrize("architecture", _CUSTOM)
@pytest.mark.parametrize("scale", _SCALES)
def test_custom_yaml_presets_match_phase9_reference(
    architecture: str,
    scale: str,
) -> None:
    manifest = _manifest()
    reference = custom_model_spec(
        cast(CustomArchitecture, architecture),
        cast(CustomScale, scale),
        input_features=3,
        max_sequence_length=64,
    )
    parameters = scale_parameters(
        manifest,
        architecture,
        cast(ScalePreset, scale),
    )
    for name, expected in parameters.items():
        assert getattr(reference, name) == expected


def _baseline_model(family: str, parameters: dict[str, object]) -> torch.nn.Module:
    if family == "mlp":
        return BaselineMLPModel(3, hidden_features=int(parameters["hidden_features"]))
    if family == "gru":
        return GRUReturnModel(3, hidden_features=int(parameters["hidden_features"]))
    if family == "lstm":
        return LSTMReturnModel(3, hidden_features=int(parameters["hidden_features"]))
    if family == "tcn":
        return TCNReturnModel(
            3,
            hidden_features=int(parameters["hidden_features"]),
            kernel_size=int(parameters["kernel_size"]),
        )
    if family == "causal_transformer":
        return CausalTransformerReturnModel(
            3,
            model_features=int(parameters["model_features"]),
            num_heads=int(parameters["num_heads"]),
            num_layers=int(parameters["num_layers"]),
            feedforward_features=int(parameters["feedforward_features"]),
            max_sequence_length=32,
        )
    raise AssertionError(f"unexpected baseline family {family!r}")


@pytest.mark.parametrize(
    "family",
    ("mlp", "gru", "lstm", "tcn", "causal_transformer"),
)
@pytest.mark.parametrize("scale", _SCALES)
def test_neural_baseline_canonical_scales_construct_and_forward(
    family: str,
    scale: str,
) -> None:
    manifest = _manifest()
    parameters = scale_parameters(
        manifest,
        family,
        cast(ScalePreset, scale),
    )
    model = _baseline_model(family, dict(parameters))
    batch = TrainingBatch(
        features=torch.zeros((4, 8, 3)),
        targets={"return_15m": torch.zeros(4)},
        asset_ids=("a", "b", "c", "d"),
        timestamps_ns=torch.full(
            (4,),
            1_700_000_000_000_000_000,
            dtype=torch.int64,
        ),
    )
    output = model(batch)
    assert output.expected_return is not None
    assert output.expected_return.shape == (4,)
    assert count_parameters(model) > 0


def test_planned_objectives_are_defined_but_not_launchable() -> None:
    manifest = _manifest()
    planned = {
        "multitask_return_rank_vol_direction_15m",
        "multi_horizon_huber_15_30m",
        "distributional_quantile_15m",
    }
    actual_planned = {
        objective.objective_id
        for objective in manifest.objectives
        if objective.selection != "enabled"
    }
    assert actual_planned == planned
    registered_for_launch = {
        objective_id
        for architecture in manifest.architectures
        for objective_id in architecture.objective_ids
    }
    assert planned.isdisjoint(registered_for_launch)
    full_multitask = objective_by_id(
        manifest,
        "multitask_return_rank_vol_direction_15m",
    )
    assert full_multitask.objective.task_weights["volatility"] == 0.25
    multi_horizon = objective_by_id(manifest, "multi_horizon_huber_15_30m")
    assert multi_horizon.objective.horizons_minutes == (15, 30)
    distributional = objective_by_id(manifest, "distributional_quantile_15m")
    assert distributional.required_heads == ("quantiles",)


def test_search_ranges_and_budget_rungs_are_preregistered() -> None:
    manifest = _manifest()
    assert manifest.search.learning_rates == (0.0001, 0.0003, 0.001)
    assert manifest.search.weight_decays == (0.0, 0.0001, 0.001)
    assert manifest.search.dropouts == (0.0, 0.1, 0.2)
    assert manifest.search.context_lengths == (32, 64, 128)
    assert manifest.seeds.screening_seed == 17
    assert manifest.seeds.finalist_seeds == (17, 29, 43)

    budgets = {budget.stage: budget for budget in manifest.budgets}
    assert budgets["screening"].fraction_of_family_full_budget == 0.15
    assert budgets["screening"].target_configurations == 66
    assert budgets["screening"].promote_configurations == 18
    assert budgets["promotion"].fraction_of_family_full_budget == 0.50
    assert budgets["promotion"].target_configurations == 18
    assert budgets["finalists"].fraction_of_family_full_budget == 1.0
    assert budgets["finalists"].target_configurations == 4
    assert budgets["finalists"].use_all_finalist_seeds is True


def test_invalid_batch_budget_and_objective_references_fail_closed() -> None:
    manifest = _manifest()

    bad_batch = copy.deepcopy(manifest.model_dump(mode="python"))
    bad_batch["search"]["batch_constraints"][0]["effective_batch_size"] = 255
    with pytest.raises(ValueError, match="effective batch"):
        CampaignSearchManifest.model_validate(bad_batch)

    bad_budget = copy.deepcopy(manifest.model_dump(mode="python"))
    bad_budget["budgets"][1]["target_configurations"] = 59
    with pytest.raises(ValueError, match="60-70"):
        CampaignSearchManifest.model_validate(bad_budget)

    bad_objective = copy.deepcopy(manifest.model_dump(mode="python"))
    bad_objective["architectures"][5]["objective_ids"] = ["does_not_exist"]
    with pytest.raises(ValueError, match="unknown objectives"):
        CampaignSearchManifest.model_validate(bad_objective)


def test_yaml_change_alters_search_without_python_edit(tmp_path: Path) -> None:
    raw = yaml.safe_load(_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    original = _manifest()
    raw["search"]["learning_rates"].append(0.0007)
    changed_path = tmp_path / "changed.yaml"
    changed_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    changed = load_campaign_search_manifest(changed_path)

    assert changed.search.learning_rates == (0.0001, 0.0003, 0.001, 0.0007)
    assert campaign_manifest_sha256(changed) != campaign_manifest_sha256(original)
    assert enumerate_campaign(changed).screening_candidate_points > enumerate_campaign(
        original
    ).screening_candidate_points


def test_registry_resolvers_fail_closed() -> None:
    manifest = _manifest()
    assert architecture_by_id(manifest, "market_mixer").kind == "custom"
    assert objective_by_id(manifest, "ranking_15m").objective.kind == "ranking"
    with pytest.raises(KeyError, match="not registered"):
        architecture_by_id(manifest, "unknown")
    with pytest.raises(KeyError, match="not registered"):
        objective_by_id(manifest, "unknown")
    with pytest.raises(KeyError, match="does not define scale"):
        scale_parameters(manifest, "ridge", "small")
