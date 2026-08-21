"""Validated, version-controlled experiment registry and search-space manifest.

Phase 10 deliberately keeps this module free of model/training imports beyond the
frozen configuration schemas. The future campaign scheduler can therefore load
and inspect the experiment plan without importing PyTorch or CUDA model code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, cast

import yaml  # type: ignore[import-untyped]
from pydantic import (
    Field,
    JsonValue,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from trading_bot.config.base import FrozenConfigModel
from trading_bot.config.schemas import ModelHead, ObjectiveConfig

ArchitectureKind = Literal[
    "classical",
    "neural_baseline",
    "advanced",
    "custom",
    "foundation",
]
ExperimentPool = Literal["mandatory", "optional"]
ScalePreset = Literal["reference", "small", "medium", "large"]
CampaignStage = Literal[
    "calibration",
    "screening",
    "promotion",
    "objective_search",
    "finalists",
]
ObjectiveSelection = Literal["enabled", "planned_not_selected"]
SearchAxis = Literal[
    "learning_rate",
    "weight_decay",
    "dropout",
    "context_length",
    "batch",
]

_REQUIRED_STAGES: tuple[CampaignStage, ...] = (
    "calibration",
    "screening",
    "promotion",
    "objective_search",
    "finalists",
)
_OBJECTIVE_LOSSES: dict[str, frozenset[str]] = {
    "excess_return": frozenset({"huber", "mse"}),
    "direction": frozenset({"bce"}),
    "ranking": frozenset({"pairwise_rank"}),
    "multitask": frozenset({"composite"}),
    "distributional": frozenset({"quantile"}),
}
_HEAD_TASK_KEYS: dict[ModelHead, str] = {
    "return": "expected_return",
    "rank": "rank_score",
    "direction": "direction_probability",
    "volatility": "volatility",
    "uncertainty": "uncertainty",
    "quantiles": "quantiles",
}


class CampaignManifestError(ValueError):
    """Raised when a campaign search manifest cannot be loaded safely."""


class CanonicalScaleConfig(FrozenConfigModel):
    """One named model-size preset whose parameters are stored in YAML."""

    name: ScalePreset
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class ArchitectureFamilySpec(FrozenConfigModel):
    """One selectable architecture family in the tournament registry."""

    family: str = Field(min_length=1)
    kind: ArchitectureKind
    pool: ExperimentPool
    searchable: bool = True
    external_artifact_required: bool = False
    scales: tuple[CanonicalScaleConfig, ...]
    objective_ids: tuple[str, ...]
    search_axes: tuple[SearchAxis, ...] = (
        "learning_rate",
        "weight_decay",
        "context_length",
        "batch",
    )

    @field_validator("scales")
    @classmethod
    def scales_must_be_unique(
        cls, value: tuple[CanonicalScaleConfig, ...]
    ) -> tuple[CanonicalScaleConfig, ...]:
        if not value:
            raise ValueError("architecture family requires at least one canonical scale")
        names = [scale.name for scale in value]
        if len(set(names)) != len(names):
            raise ValueError("architecture scale names must be unique")
        return value

    @field_validator("objective_ids", "search_axes")
    @classmethod
    def string_tuple_must_be_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("architecture tuple values must be unique")
        return value

    @model_validator(mode="after")
    def validate_architecture_boundary(self) -> ArchitectureFamilySpec:
        scale_names = {scale.name for scale in self.scales}
        if self.kind in {"advanced", "custom"} and scale_names != {
            "small",
            "medium",
            "large",
        }:
            raise ValueError("advanced/custom families require small/medium/large presets")
        if self.kind == "classical" and scale_names != {"reference"}:
            raise ValueError("classical references use exactly the reference scale")
        if self.external_artifact_required and self.pool != "optional":
            raise ValueError("external-artifact families must remain optional until selected")
        if self.searchable and not self.objective_ids:
            raise ValueError("searchable architecture must declare supported objectives")
        return self


class ObjectiveVariant(FrozenConfigModel):
    """One frozen objective candidate plus its selection state and required heads."""

    objective_id: str = Field(min_length=1)
    objective: ObjectiveConfig
    required_heads: tuple[ModelHead, ...]
    selection: ObjectiveSelection = "enabled"

    @field_validator("required_heads")
    @classmethod
    def heads_must_be_unique(cls, value: tuple[ModelHead, ...]) -> tuple[ModelHead, ...]:
        if not value:
            raise ValueError("objective variant requires at least one model head")
        if len(set(value)) != len(value):
            raise ValueError("objective required heads must be unique")
        return value

    @model_validator(mode="after")
    def validate_objective_loss_pair(self) -> ObjectiveVariant:
        allowed = _OBJECTIVE_LOSSES[self.objective.kind]
        if self.objective.loss not in allowed:
            raise ValueError(
                f"objective kind {self.objective.kind!r} does not support "
                f"loss {self.objective.loss!r} in the campaign manifest"
            )
        if self.objective.kind == "multitask":
            required = {_HEAD_TASK_KEYS[head] for head in self.required_heads}
            missing = sorted(required - set(self.objective.task_weights))
            if missing:
                raise ValueError(f"multitask objective is missing task weights: {missing}")
        return self


class BatchConstraint(FrozenConfigModel):
    """Microbatch/effective-batch combination with explicit accumulation."""

    microbatch_size: PositiveInt
    gradient_accumulation_steps: PositiveInt
    effective_batch_size: PositiveInt

    @model_validator(mode="after")
    def effective_batch_must_match(self) -> BatchConstraint:
        if self.microbatch_size * self.gradient_accumulation_steps != self.effective_batch_size:
            raise ValueError("effective batch must equal microbatch * gradient accumulation")
        return self


class HyperparameterSearchSpace(FrozenConfigModel):
    """Global bounded search axes; families explicitly opt into relevant axes."""

    learning_rates: tuple[PositiveFloat, ...]
    weight_decays: tuple[float, ...]
    dropouts: tuple[float, ...]
    context_lengths: tuple[PositiveInt, ...]
    batch_constraints: tuple[BatchConstraint, ...]

    @field_validator("learning_rates", "context_lengths")
    @classmethod
    def positive_axes_must_be_unique(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value:
            raise ValueError("search axis must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("search axis values must be unique")
        return value

    @field_validator("weight_decays")
    @classmethod
    def weight_decays_must_be_valid(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value or any(item < 0 for item in value):
            raise ValueError("weight-decay search values must be non-negative")
        if len(set(value)) != len(value):
            raise ValueError("weight-decay search values must be unique")
        return value

    @field_validator("dropouts")
    @classmethod
    def dropouts_must_be_valid(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value or any(item < 0 or item >= 1 for item in value):
            raise ValueError("dropout search values must lie in [0, 1)")
        if len(set(value)) != len(value):
            raise ValueError("dropout search values must be unique")
        return value

    @field_validator("batch_constraints")
    @classmethod
    def batch_constraints_must_be_unique(
        cls, value: tuple[BatchConstraint, ...]
    ) -> tuple[BatchConstraint, ...]:
        if not value:
            raise ValueError("at least one batch constraint is required")
        identities = [
            (
                item.microbatch_size,
                item.gradient_accumulation_steps,
                item.effective_batch_size,
            )
            for item in value
        ]
        if len(set(identities)) != len(identities):
            raise ValueError("batch constraints must be unique")
        return value


class SeedPolicy(FrozenConfigModel):
    """Screening uses one preregistered seed; finalists use the full seed set."""

    screening_seed: int = Field(ge=0)
    finalist_seeds: tuple[int, ...]

    @field_validator("finalist_seeds")
    @classmethod
    def finalist_seeds_must_be_valid(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(seed < 0 for seed in value):
            raise ValueError("finalist seeds must be non-negative and non-empty")
        if len(set(value)) != len(value):
            raise ValueError("finalist seeds must be unique")
        return value

    @model_validator(mode="after")
    def screening_seed_must_be_reused(self) -> SeedPolicy:
        if self.screening_seed not in self.finalist_seeds:
            raise ValueError("screening seed must be included in finalist seeds")
        return self


class StageBudget(FrozenConfigModel):
    """Relative training budget and planned breadth for one campaign rung."""

    stage: CampaignStage
    fraction_of_family_full_budget: float = Field(gt=0.0, le=1.0)
    target_configurations: PositiveInt | None = None
    promote_configurations: PositiveInt | None = None
    use_all_finalist_seeds: bool = False

    @model_validator(mode="after")
    def promotion_target_must_fit(self) -> StageBudget:
        if (
            self.target_configurations is not None
            and self.promote_configurations is not None
            and self.promote_configurations > self.target_configurations
        ):
            raise ValueError("promotion target cannot exceed stage configuration count")
        return self


class CampaignSearchManifest(FrozenConfigModel):
    """Complete Phase 10 experiment registry/search-space contract."""

    schema_version: Literal[1] = 1
    campaign_id: str = Field(min_length=1)
    frozen: bool
    screening_objective_id: str = Field(min_length=1)
    architectures: tuple[ArchitectureFamilySpec, ...]
    objectives: tuple[ObjectiveVariant, ...]
    search: HyperparameterSearchSpace
    seeds: SeedPolicy
    budgets: tuple[StageBudget, ...]

    @model_validator(mode="after")
    def validate_manifest(self) -> CampaignSearchManifest:
        architecture_ids = [item.family for item in self.architectures]
        if not architecture_ids or len(set(architecture_ids)) != len(architecture_ids):
            raise ValueError("architecture family IDs must be non-empty and unique")
        objective_ids = [item.objective_id for item in self.objectives]
        if not objective_ids or len(set(objective_ids)) != len(objective_ids):
            raise ValueError("objective IDs must be non-empty and unique")
        objective_map = {item.objective_id: item for item in self.objectives}
        objective_id_set = set(objective_ids)
        enabled_objective_ids = {
            item.objective_id for item in self.objectives if item.selection == "enabled"
        }
        for architecture in self.architectures:
            unknown = set(architecture.objective_ids) - objective_id_set
            if unknown:
                raise ValueError(
                    f"architecture {architecture.family!r} references unknown objectives: "
                    f"{sorted(unknown)}"
                )
            not_launchable = set(architecture.objective_ids) - enabled_objective_ids
            if not_launchable:
                raise ValueError(
                    f"architecture {architecture.family!r} references objectives that are "
                    f"not selected for launch: {sorted(not_launchable)}"
                )
        screening = objective_map.get(self.screening_objective_id)
        if screening is None or screening.selection != "enabled":
            raise ValueError("screening objective must exist and be enabled")
        if not any(item.pool == "mandatory" for item in self.architectures):
            raise ValueError("campaign requires at least one mandatory architecture family")

        budget_map = {budget.stage: budget for budget in self.budgets}
        if len(budget_map) != len(self.budgets) or tuple(budget_map) != _REQUIRED_STAGES:
            raise ValueError(
                "campaign budgets must define calibration/screening/promotion/"
                "objective_search/finalists exactly once and in order"
            )
        screening_budget = budget_map["screening"]
        promotion_budget = budget_map["promotion"]
        finalist_budget = budget_map["finalists"]
        if not 0.10 <= screening_budget.fraction_of_family_full_budget <= 0.15:
            raise ValueError("screening budget must remain within the frozen 10-15% rung")
        if not 0.35 <= promotion_budget.fraction_of_family_full_budget <= 0.50:
            raise ValueError("promotion budget must remain within the frozen 35-50% rung")
        if finalist_budget.fraction_of_family_full_budget != 1.0:
            raise ValueError("finalist budget must use the full family training budget")
        if screening_budget.target_configurations is None or not (
            60 <= screening_budget.target_configurations <= 70
        ):
            raise ValueError("screening target must remain within the planned 60-70 configs")
        if screening_budget.promote_configurations is None or not (
            16 <= screening_budget.promote_configurations <= 20
        ):
            raise ValueError("screening promotion target must remain within 16-20 configs")
        if promotion_budget.target_configurations != screening_budget.promote_configurations:
            raise ValueError("promotion breadth must equal the screening promotion target")
        if finalist_budget.target_configurations is None or not (
            3 <= finalist_budget.target_configurations <= 4
        ):
            raise ValueError("finalist target must remain within the planned 3-4 systems")
        if not finalist_budget.use_all_finalist_seeds:
            raise ValueError("finalists must use all preregistered finalist seeds")
        return self


class CampaignEnumeration(FrozenConfigModel):
    """Deterministic manifest-only summary for schedulers and CI inspection."""

    campaign_id: str
    manifest_sha256: str
    mandatory_families: tuple[str, ...]
    optional_families: tuple[str, ...]
    searchable_families: tuple[str, ...]
    enabled_objectives: tuple[str, ...]
    planned_objectives: tuple[str, ...]
    canonical_scale_count: int
    screening_candidate_points: int
    planned_fit_count: int


def load_campaign_search_manifest(path: str | Path) -> CampaignSearchManifest:
    """Load one strict YAML search manifest without importing model/training code."""
    manifest_path = Path(path)
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CampaignManifestError(
            f"unable to read campaign manifest {manifest_path}: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise CampaignManifestError(f"invalid campaign YAML in {manifest_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise CampaignManifestError("campaign manifest root must be a mapping")
    try:
        return CampaignSearchManifest.model_validate(raw)
    except ValueError as exc:
        raise CampaignManifestError(f"invalid campaign search manifest: {exc}") from exc


def campaign_manifest_canonical_json(manifest: CampaignSearchManifest) -> str:
    """Return deterministic JSON suitable for immutable identity and audit logs."""
    return json.dumps(
        manifest.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def campaign_manifest_sha256(manifest: CampaignSearchManifest) -> str:
    """Hash the complete validated manifest rather than the source YAML formatting."""
    payload = campaign_manifest_canonical_json(manifest).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def enumerate_campaign(manifest: CampaignSearchManifest) -> CampaignEnumeration:
    """Summarize the entire intended search space without expanding every trial."""
    mandatory = tuple(item.family for item in manifest.architectures if item.pool == "mandatory")
    optional = tuple(item.family for item in manifest.architectures if item.pool == "optional")
    searchable = tuple(item.family for item in manifest.architectures if item.searchable)
    enabled_objectives = tuple(
        item.objective_id for item in manifest.objectives if item.selection == "enabled"
    )
    planned_objectives = tuple(
        item.objective_id
        for item in manifest.objectives
        if item.selection == "planned_not_selected"
    )
    scale_count = sum(len(item.scales) for item in manifest.architectures)

    search = manifest.search
    screening_points = 0
    for architecture in manifest.architectures:
        if not architecture.searchable:
            continue
        if manifest.screening_objective_id not in architecture.objective_ids:
            continue
        family_points = len(architecture.scales)
        if "learning_rate" in architecture.search_axes:
            family_points *= len(search.learning_rates)
        if "weight_decay" in architecture.search_axes:
            family_points *= len(search.weight_decays)
        if "dropout" in architecture.search_axes:
            family_points *= len(search.dropouts)
        if "context_length" in architecture.search_axes:
            family_points *= len(search.context_lengths)
        if "batch" in architecture.search_axes:
            family_points *= len(search.batch_constraints)
        screening_points += family_points

    budget_map = {budget.stage: budget for budget in manifest.budgets}
    calibration_count = int(budget_map["calibration"].target_configurations or 0)
    screening_count = int(budget_map["screening"].target_configurations or 0)
    promotion_count = int(budget_map["promotion"].target_configurations or 0)
    objective_count = int(budget_map["objective_search"].target_configurations or 0)
    finalist_systems = int(budget_map["finalists"].target_configurations or 0)
    finalist_fits = finalist_systems * len(manifest.seeds.finalist_seeds)
    planned_fit_count = (
        calibration_count + screening_count + promotion_count + objective_count + finalist_fits
    )
    return CampaignEnumeration(
        campaign_id=manifest.campaign_id,
        manifest_sha256=campaign_manifest_sha256(manifest),
        mandatory_families=mandatory,
        optional_families=optional,
        searchable_families=searchable,
        enabled_objectives=enabled_objectives,
        planned_objectives=planned_objectives,
        canonical_scale_count=scale_count,
        screening_candidate_points=screening_points,
        planned_fit_count=planned_fit_count,
    )


def objective_by_id(manifest: CampaignSearchManifest, objective_id: str) -> ObjectiveVariant:
    """Resolve an objective by immutable manifest ID and fail closed if absent."""
    for objective in manifest.objectives:
        if objective.objective_id == objective_id:
            return objective
    raise KeyError(f"campaign objective {objective_id!r} is not registered")


def architecture_by_id(manifest: CampaignSearchManifest, family: str) -> ArchitectureFamilySpec:
    """Resolve an architecture by immutable manifest ID and fail closed if absent."""
    for architecture in manifest.architectures:
        if architecture.family == family:
            return architecture
    raise KeyError(f"campaign architecture {family!r} is not registered")


def scale_parameters(
    manifest: CampaignSearchManifest,
    family: str,
    scale: ScalePreset,
) -> dict[str, JsonValue]:
    """Return a copy of canonical YAML parameters for one architecture/scale."""
    architecture = architecture_by_id(manifest, family)
    for scale_config in architecture.scales:
        if scale_config.name == scale:
            return cast(dict[str, JsonValue], dict(scale_config.parameters))
    raise KeyError(f"campaign architecture {family!r} does not define scale {scale!r}")
