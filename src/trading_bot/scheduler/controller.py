"""Deadline-aware campaign controller over durable Phase 11 primitives."""

from __future__ import annotations

import math
from collections.abc import Sequence

from trading_bot.campaign.search_space import (
    CampaignSearchManifest,
    CampaignStage,
    campaign_manifest_canonical_json,
    campaign_manifest_sha256,
)
from trading_bot.evaluation.leaderboard import LeaderboardRow
from trading_bot.scheduler.db import CampaignDB
from trading_bot.scheduler.planner import build_screening_trial_specs
from trading_bot.scheduler.policy import SchedulerRuntimePolicy
from trading_bot.scheduler.promotion import select_promotions
from trading_bot.scheduler.runtime import (
    LaunchDecision,
    RuntimeEstimator,
    dynamic_drain_reserve_seconds,
    launch_decision,
)
from trading_bot.scheduler.types import (
    CampaignState,
    DrainInputs,
    LaunchPriority,
    RuntimeObservation,
    TrialSpec,
    TrialState,
)
from trading_bot.storage.base import StorageBackend, StorageObjectMetadata


class CampaignController:
    """Model-free controller; worker execution is delegated to subprocess runners."""

    def __init__(
        self,
        db: CampaignDB,
        manifest: CampaignSearchManifest,
        policy: SchedulerRuntimePolicy,
    ) -> None:
        self.db = db
        self.manifest = manifest
        self.policy = policy
        self.estimator = RuntimeEstimator(db, policy)

    @property
    def campaign_id(self) -> str:
        return self.manifest.campaign_id

    def bootstrap(self, *, started_at: float, deadline_at: float) -> None:
        manifest_hash = campaign_manifest_sha256(self.manifest)
        manifest_json = campaign_manifest_canonical_json(self.manifest)
        try:
            row = self.db.campaign_row(self.campaign_id)
        except KeyError:
            self.db.create_campaign(
                campaign_id=self.campaign_id,
                manifest_sha256=manifest_hash,
                manifest_json=manifest_json,
                started_at=started_at,
                deadline_at=deadline_at,
                drain_reserve_seconds=float(self.policy.initial_drain_reserve_minutes * 60),
            )
            return
        stored_manifest_hash = str(row["manifest_sha256"])
        stored_manifest_json = str(row["manifest_json"])
        if stored_manifest_hash != manifest_hash or stored_manifest_json != manifest_json:
            raise RuntimeError("existing campaign DB does not match the frozen Phase 10 manifest")
        if float(row["deadline_at"]) != deadline_at:
            raise RuntimeError("campaign restart cannot silently change the fixed deadline")

    def transition_campaign(self, target: CampaignState) -> None:
        self.db.transition_campaign(self.campaign_id, target)

    def register_screening_queue(
        self, *, fallback_runtime_seconds: float
    ) -> tuple[TrialSpec, ...]:
        specs = build_screening_trial_specs(
            self.manifest,
            fallback_runtime_seconds=fallback_runtime_seconds,
        )
        for spec in specs:
            self.db.insert_trial(self.campaign_id, spec)
        return specs

    def register_trial(self, spec: TrialSpec) -> None:
        self.db.insert_trial(self.campaign_id, spec)

    def launch_decision(
        self,
        spec: TrialSpec,
        *,
        now: float,
        drain_inputs: DrainInputs,
    ) -> LaunchDecision:
        row = self.db.campaign_row(self.campaign_id)
        reserve = dynamic_drain_reserve_seconds(self.policy, drain_inputs)
        if math.isfinite(reserve):
            self.db.update_drain_reserve(self.campaign_id, reserve)
        estimate = self.estimator.estimate_seconds(spec)
        decision = launch_decision(
            self.policy,
            spec,
            estimated_runtime_seconds=estimate,
            now=now,
            deadline_at=float(row["deadline_at"]),
            drain_reserve_seconds=reserve,
        )
        self.db.record_event(
            self.campaign_id,
            "launch_decision",
            decision.model_dump(mode="json"),
            spec.trial_id,
        )
        return decision

    def mark_running(self, trial_id: str) -> None:
        self.db.transition_trial(trial_id, TrialState.STARTING)
        self.db.transition_trial(trial_id, TrialState.RUNNING)

    def mark_evaluating(
        self,
        trial_id: str,
        *,
        runtime_seconds: float,
        samples_per_second: float | None = None,
        context_length: int | None = None,
        precision: str = "bf16",
        gpu_utilization_percent: float | None = None,
        peak_vram_bytes: int | None = None,
    ) -> None:
        spec = self.db.trial_spec(trial_id)
        self.db.record_runtime(
            RuntimeObservation(
                trial_id=trial_id,
                family=spec.family,
                scale=spec.scale,
                context_length=context_length,
                precision=precision,
                budget_fraction=spec.budget_fraction,
                runtime_seconds=runtime_seconds,
                samples_per_second=samples_per_second,
                gpu_utilization_percent=gpu_utilization_percent,
                peak_vram_bytes=peak_vram_bytes,
            )
        )
        self.db.transition_trial(trial_id, TrialState.EVALUATING)

    def mark_synced_complete(self, trial_id: str) -> None:
        state = self.db.trial_state(trial_id)
        if state == TrialState.EVALUATING:
            self.db.transition_trial(trial_id, TrialState.SYNCING)
        self.db.transition_trial(trial_id, TrialState.COMPLETE)

    def retry_process_failure(self, trial_id: str, *, message: str) -> TrialSpec | None:
        """Create an immutable same-config retry child; classification policy remains Phase 12."""
        spec = self.db.trial_spec(trial_id)
        if spec.attempt >= self.policy.max_trial_retries:
            self.db.record_failure(
                trial_id,
                failure_class="process_failure",
                retryable=False,
                message=message,
            )
            self.db.transition_trial(trial_id, TrialState.TERMINAL_FAILURE)
            return None
        self.db.record_failure(
            trial_id,
            failure_class="process_failure",
            retryable=True,
            message=message,
        )
        self.db.transition_trial(trial_id, TrialState.RETRYABLE_FAILURE)
        child = TrialSpec(
            trial_id=f"{trial_id}-r{spec.attempt + 1}",
            family=spec.family,
            scale=spec.scale,
            stage=spec.stage,
            budget_fraction=spec.budget_fraction,
            priority=spec.priority,
            config=spec.config,
            parent_trial_id=trial_id,
            root_trial_id=spec.effective_root_trial_id,
            attempt=spec.attempt + 1,
            fallback_runtime_seconds=spec.fallback_runtime_seconds,
        )
        self.db.insert_trial(self.campaign_id, child)
        return child

    def promote_from_leaderboard(
        self,
        leaderboard: Sequence[LeaderboardRow],
        *,
        count: int,
        to_stage: CampaignStage,
        to_budget_fraction: float,
    ) -> tuple[TrialSpec, ...]:
        selected_rows = select_promotions(leaderboard, count=count)
        selected = {row.trial_id: row for row in selected_rows}
        children: list[TrialSpec] = []
        for row in sorted(leaderboard, key=lambda item: item.rank):
            state = self.db.trial_state(row.trial_id)
            if state != TrialState.EVALUATING:
                raise RuntimeError(
                    f"promotion candidate {row.trial_id!r} must be in "
                    f"EVALUATING, got {state.value}"
                )
            parent = self.db.trial_spec(row.trial_id)
            promoted = selected.get(row.trial_id)
            if promoted is None:
                self.db.transition_trial(row.trial_id, TrialState.PRUNED)
                continue
            self.db.transition_trial(row.trial_id, TrialState.COMPLETE)
            child_id = f"{row.trial_id}-{to_stage}"
            child = TrialSpec(
                trial_id=child_id,
                family=parent.family,
                scale=parent.scale,
                stage=to_stage,
                budget_fraction=to_budget_fraction,
                priority=(
                    LaunchPriority.FINALIST if to_stage == "finalists" else parent.priority
                ),
                config=parent.config,
                parent_trial_id=parent.trial_id,
                root_trial_id=parent.effective_root_trial_id,
                attempt=0,
                fallback_runtime_seconds=(
                    parent.fallback_runtime_seconds
                    * to_budget_fraction
                    / parent.budget_fraction
                ),
            )
            self.db.insert_trial(self.campaign_id, child)
            self.db.record_promotion(
                parent_trial_id=parent.trial_id,
                child_trial_id=child.trial_id,
                from_stage=parent.stage,
                to_stage=to_stage,
                leaderboard_rank=promoted.rank,
                reason="canonical_phase6_leaderboard_rank",
            )
            children.append(child)
        return tuple(children)

    def interrupt_pending_for_drain(self) -> tuple[str, ...]:
        interrupted: list[str] = []
        for row in self.db.rows("trials"):
            if TrialState(str(row["state"])) == TrialState.PENDING:
                trial_id = str(row["trial_id"])
                self.db.transition_trial(trial_id, TrialState.INTERRUPTED)
                interrupted.append(trial_id)
        return tuple(interrupted)

    def enter_drain(self) -> None:
        state = self.db.campaign_state(self.campaign_id)
        if state != CampaignState.DRAIN:
            self.db.transition_campaign(self.campaign_id, CampaignState.DRAIN)
        self.interrupt_pending_for_drain()

    def complete_campaign(self, *, unsynced_critical_bytes: int) -> None:
        if unsynced_critical_bytes != 0:
            raise RuntimeError("campaign cannot complete with unsynced critical artifacts")
        active = {
            TrialState.PENDING,
            TrialState.STARTING,
            TrialState.RUNNING,
            TrialState.EVALUATING,
            TrialState.SYNCING,
        }
        unfinished = [
            str(row["trial_id"])
            for row in self.db.rows("trials")
            if TrialState(str(row["state"])) in active
        ]
        if unfinished:
            raise RuntimeError(f"campaign cannot complete with active trials: {unfinished}")
        if self.db.campaign_state(self.campaign_id) != CampaignState.DRAIN:
            raise RuntimeError("campaign must enter DRAIN before COMPLETE")
        self.db.transition_campaign(self.campaign_id, CampaignState.COMPLETE)

    def snapshot(self, backend: StorageBackend, key: str) -> StorageObjectMetadata:
        metadata = self.db.snapshot_to_storage(backend, key)
        self.db.record_event(
            self.campaign_id,
            "campaign_snapshot_verified",
            {"key": key, "size": metadata.size, "sha256": metadata.checksum_sha256},
        )
        return metadata
