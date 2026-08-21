"""Deterministic Phase 10 manifest -> Phase 11 screening trial planning."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterable

from pydantic import JsonValue

from trading_bot.campaign.search_space import CampaignSearchManifest, ExperimentPool
from trading_bot.scheduler.types import LaunchPriority, TrialSpec


def build_screening_trial_specs(
    manifest: CampaignSearchManifest,
    *,
    fallback_runtime_seconds: float,
) -> tuple[TrialSpec, ...]:
    """Build the frozen screening breadth by stratified round-robin over family grids."""
    if fallback_runtime_seconds <= 0:
        raise ValueError("fallback runtime must be positive")
    budget = next(item for item in manifest.budgets if item.stage == "screening")
    if budget.target_configurations is None:
        raise ValueError("screening budget must define target_configurations")

    family_candidates: list[list[dict[str, JsonValue]]] = []
    family_meta: list[tuple[str, ExperimentPool]] = []
    for architecture in manifest.architectures:
        if not architecture.searchable:
            continue
        if manifest.screening_objective_id not in architecture.objective_ids:
            continue
        candidates = list(_family_candidates(manifest, architecture.family))
        if not candidates:
            raise ValueError(f"searchable family {architecture.family!r} produced no candidates")
        family_candidates.append(candidates)
        family_meta.append((architecture.family, architecture.pool))

    if not family_candidates:
        raise ValueError("campaign contains no searchable screening families")

    selected: list[tuple[str, ExperimentPool, dict[str, JsonValue]]] = []
    cursor = 0
    while len(selected) < budget.target_configurations:
        made_progress = False
        for (family, pool), candidates in zip(family_meta, family_candidates, strict=True):
            if cursor < len(candidates):
                selected.append((family, pool, candidates[cursor]))
                made_progress = True
                if len(selected) >= budget.target_configurations:
                    break
        if not made_progress:
            raise ValueError("screening search space is smaller than target breadth")
        cursor += 1

    specs: list[TrialSpec] = []
    for index, (family, pool, config) in enumerate(selected, start=1):
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
        suffix = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]
        scale = str(config["scale"])
        specs.append(
            TrialSpec(
                trial_id=f"{manifest.campaign_id}-screen-{index:03d}-{suffix}",
                family=family,
                scale=scale,
                stage="screening",
                budget_fraction=budget.fraction_of_family_full_budget,
                priority=(
                    LaunchPriority.MANDATORY if pool == "mandatory" else LaunchPriority.OPTIONAL
                ),
                config=config,
                fallback_runtime_seconds=fallback_runtime_seconds,
            )
        )
    return tuple(specs)


def _family_candidates(
    manifest: CampaignSearchManifest,
    family: str,
) -> Iterable[dict[str, JsonValue]]:
    architecture = next(item for item in manifest.architectures if item.family == family)
    search = manifest.search

    def values(axis: str, candidates: tuple[object, ...]) -> tuple[object | None, ...]:
        return candidates if axis in architecture.search_axes else (None,)

    learning_rates = values("learning_rate", tuple(search.learning_rates))
    weight_decays = values("weight_decay", tuple(search.weight_decays))
    dropouts = values("dropout", tuple(search.dropouts))
    context_lengths = values("context_length", tuple(search.context_lengths))
    batch_constraints = values("batch", tuple(search.batch_constraints))

    for scale in architecture.scales:
        for learning_rate, weight_decay, dropout, context_length, batch in itertools.product(
            learning_rates,
            weight_decays,
            dropouts,
            context_lengths,
            batch_constraints,
        ):
            config: dict[str, JsonValue] = {
                "family": architecture.family,
                "scale": scale.name,
                "model_parameters": scale.parameters,
                "objective_id": manifest.screening_objective_id,
            }
            if learning_rate is not None:
                config["learning_rate"] = float(learning_rate)
            if weight_decay is not None:
                config["weight_decay"] = float(weight_decay)
            if dropout is not None:
                config["dropout"] = float(dropout)
            if context_length is not None:
                config["context_length"] = int(context_length)
            if batch is not None:
                config["batch"] = batch.model_dump(mode="json")
            yield config
