"""Fast compressed campaign simulation for the Phase 11 acceptance gate."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from trading_bot.campaign.search_space import CampaignSearchManifest
from trading_bot.config.base import FrozenConfigModel
from trading_bot.evaluation.leaderboard import LeaderboardRow
from trading_bot.scheduler.controller import CampaignController
from trading_bot.scheduler.db import CampaignDB
from trading_bot.scheduler.policy import SchedulerRuntimePolicy
from trading_bot.scheduler.runtime import DeadlineMode
from trading_bot.scheduler.types import CampaignState, DrainInputs, LaunchPriority, TrialSpec
from trading_bot.storage.base import StorageBackend


class SimulationSummary(FrozenConfigModel):
    campaign_id: str
    final_state: CampaignState
    screening_registered: int = Field(gt=0)
    evaluated_screening_trials: int = Field(gt=0)
    promotion_rows: int = Field(gt=0)
    retry_failures: int = Field(gt=0)
    promotion_lineage_rows: int = Field(gt=0)
    interrupted_at_drain: int = Field(ge=0)
    late_mode: DeadlineMode
    optional_late_launch_allowed: bool
    finalist_late_launch_allowed: bool
    final_snapshot_key: str
    final_snapshot_size: int = Field(gt=0)


def run_compressed_campaign_simulation(
    manifest: CampaignSearchManifest,
    policy: SchedulerRuntimePolicy,
    *,
    db_path: str | Path,
    durable_backend: StorageBackend,
    snapshot_key: str,
    started_at: float = 2_000_000_000.0,
    duration_hours: float = 48.0,
) -> SimulationSummary:
    """Exercise state, restart, retry, promotion, deadline, drain, and durable snapshot paths."""
    deadline_at = started_at + duration_hours * 3600.0
    db = CampaignDB(db_path)
    controller = CampaignController(db, manifest, policy)
    controller.bootstrap(started_at=started_at, deadline_at=deadline_at)
    controller.transition_campaign(CampaignState.CALIBRATION)

    searchable = [item for item in manifest.architectures if item.searchable]
    for index, architecture in enumerate(searchable[:3], start=1):
        scale = architecture.scales[0]
        spec = TrialSpec(
            trial_id=f"sim-calibration-{index}",
            family=architecture.family,
            scale=scale.name,
            stage="calibration",
            budget_fraction=0.05,
            config={
                "family": architecture.family,
                "scale": scale.name,
                "context_length": 32,
            },
            fallback_runtime_seconds=600.0,
        )
        controller.register_trial(spec)
        controller.mark_running(spec.trial_id)
        controller.mark_evaluating(
            spec.trial_id,
            runtime_seconds=300.0 + index * 30.0,
            samples_per_second=1000.0 / index,
            context_length=32,
            gpu_utilization_percent=80.0 + index,
            peak_vram_bytes=1_000_000_000 * index,
        )
        controller.mark_synced_complete(spec.trial_id)

    controller.transition_campaign(CampaignState.SCREENING)
    screening = controller.register_screening_queue(fallback_runtime_seconds=900.0)
    evaluated_ids: list[str] = []
    for index, spec in enumerate(screening[:8]):
        decision = controller.launch_decision(
            spec,
            now=started_at + 3600.0 + index * 120.0,
            drain_inputs=DrainInputs(),
        )
        if not decision.allowed:
            raise RuntimeError("compressed screening trial unexpectedly rejected")
        controller.mark_running(spec.trial_id)
        active = spec
        if index == 0:
            retry = controller.retry_process_failure(
                spec.trial_id,
                message="synthetic transient worker exit",
            )
            if retry is None:
                raise RuntimeError("compressed simulation expected one retry child")
            controller.mark_running(retry.trial_id)
            active = retry
        controller.mark_evaluating(
            active.trial_id,
            runtime_seconds=100.0 + index * 10.0,
            samples_per_second=500.0 - index * 10.0,
            context_length=_context_length(active),
        )
        evaluated_ids.append(active.trial_id)

    # Simulate a controller restart while preserving the exact fixed deadline and manifest.
    db.close()
    db = CampaignDB(db_path)
    controller = CampaignController(db, manifest, policy)
    controller.bootstrap(started_at=started_at, deadline_at=deadline_at)

    screening_rows = _leaderboard_rows(evaluated_ids)
    promotion_budget = next(item for item in manifest.budgets if item.stage == "promotion")
    promotion_children = controller.promote_from_leaderboard(
        screening_rows,
        count=4,
        to_stage="promotion",
        to_budget_fraction=promotion_budget.fraction_of_family_full_budget,
    )
    controller.transition_campaign(CampaignState.PROMOTION)
    for index, spec in enumerate(promotion_children):
        controller.mark_running(spec.trial_id)
        controller.mark_evaluating(
            spec.trial_id,
            runtime_seconds=350.0 + index * 20.0,
            samples_per_second=350.0 - index * 10.0,
            context_length=_context_length(spec),
        )

    promotion_rows = _leaderboard_rows([spec.trial_id for spec in promotion_children])
    objective_budget = next(item for item in manifest.budgets if item.stage == "objective_search")
    objective_children = controller.promote_from_leaderboard(
        promotion_rows,
        count=2,
        to_stage="objective_search",
        to_budget_fraction=objective_budget.fraction_of_family_full_budget,
    )
    controller.transition_campaign(CampaignState.OBJECTIVE_SEARCH)
    for index, spec in enumerate(objective_children):
        controller.mark_running(spec.trial_id)
        controller.mark_evaluating(
            spec.trial_id,
            runtime_seconds=420.0 + index * 30.0,
            samples_per_second=300.0 - index * 10.0,
            context_length=_context_length(spec),
        )

    objective_rows = _leaderboard_rows([spec.trial_id for spec in objective_children])
    finalist_budget = next(item for item in manifest.budgets if item.stage == "finalists")
    finalist_children = controller.promote_from_leaderboard(
        objective_rows,
        count=2,
        to_stage="finalists",
        to_budget_fraction=finalist_budget.fraction_of_family_full_budget,
    )
    controller.transition_campaign(CampaignState.FINALISTS)

    reserve_seconds = float(policy.initial_drain_reserve_minutes * 60)
    late_now = deadline_at - reserve_seconds - 4.0 * 3600.0
    pending_optional = _first_pending_optional(controller, screening)
    optional_decision = controller.launch_decision(
        pending_optional,
        now=late_now,
        drain_inputs=DrainInputs(),
    )
    finalist_decision = controller.launch_decision(
        finalist_children[0],
        now=late_now,
        drain_inputs=DrainInputs(),
    )

    for index, spec in enumerate(finalist_children):
        if not controller.launch_decision(
            spec,
            now=late_now + index * 60.0,
            drain_inputs=DrainInputs(),
        ).allowed:
            raise RuntimeError("finalist unexpectedly rejected in finalist-only mode")
        controller.mark_running(spec.trial_id)
        controller.mark_evaluating(
            spec.trial_id,
            runtime_seconds=700.0 + index * 50.0,
            samples_per_second=250.0 - index * 10.0,
            context_length=_context_length(spec),
        )
        controller.mark_synced_complete(spec.trial_id)

    # Growing evaluator/sync work participates in the reserve, then the clock crosses it.
    drain_inputs = DrainInputs(
        outstanding_evaluator_seconds=900.0,
        unsynced_bytes=500_000_000,
        storage_bytes_per_second=25_000_000.0,
    )
    drain_now = deadline_at - reserve_seconds + 1.0
    drain_decision = controller.launch_decision(
        pending_optional,
        now=drain_now,
        drain_inputs=drain_inputs,
    )
    if drain_decision.mode != DeadlineMode.DRAIN or drain_decision.allowed:
        raise RuntimeError("compressed simulation failed to enter deadline drain mode")
    controller.enter_drain()
    interrupted = sum(1 for row in db.rows("trials") if str(row["state"]) == "INTERRUPTED")
    controller.complete_campaign(unsynced_critical_bytes=0)
    snapshot = controller.snapshot(durable_backend, snapshot_key)

    summary = SimulationSummary(
        campaign_id=manifest.campaign_id,
        final_state=db.campaign_state(manifest.campaign_id),
        screening_registered=len(screening),
        evaluated_screening_trials=len(evaluated_ids),
        promotion_rows=len(promotion_children),
        retry_failures=len(db.rows("failures")),
        promotion_lineage_rows=len(db.rows("promotions")),
        interrupted_at_drain=interrupted,
        late_mode=optional_decision.mode,
        optional_late_launch_allowed=optional_decision.allowed,
        finalist_late_launch_allowed=finalist_decision.allowed,
        final_snapshot_key=snapshot.key,
        final_snapshot_size=snapshot.size,
    )
    db.close()
    return summary


def _context_length(spec: TrialSpec) -> int | None:
    value = spec.config.get("context_length")
    return int(value) if isinstance(value, int) else None


def _first_pending_optional(
    controller: CampaignController,
    screening: tuple[TrialSpec, ...],
) -> TrialSpec:
    for spec in screening:
        if (
            spec.priority == LaunchPriority.OPTIONAL
            and controller.db.trial_state(spec.trial_id).value == "PENDING"
        ):
            return spec
    # The v1 foundation reference is not searchable, so screening may be all mandatory.
    for spec in screening:
        if controller.db.trial_state(spec.trial_id).value == "PENDING":
            return spec.model_copy(update={"priority": LaunchPriority.OPTIONAL})
    raise RuntimeError("compressed simulation requires one pending screening trial")


def _leaderboard_rows(trial_ids: list[str]) -> tuple[LeaderboardRow, ...]:
    return tuple(
        LeaderboardRow(
            rank=index,
            trial_id=trial_id,
            eligible=True,
            disqualification_reasons=(),
            mean_rank_ic=0.10 / index,
            icir=1.0 / index,
            net_sharpe=1.5 / index,
            calmar=1.0 / index,
            maximum_drawdown=-0.05 * index,
            sortino=1.8 / index,
            es95=0.01 * index,
            average_turnover=0.20,
            total_modeled_cost=0.001 * index,
            positive_fold_fraction=1.0,
            dsr_probability=0.90,
            pbo_probability=0.10,
            attempted_trial_count=len(trial_ids),
        )
        for index, trial_id in enumerate(trial_ids, start=1)
    )
