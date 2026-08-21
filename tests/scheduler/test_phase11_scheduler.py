"""CPU acceptance gate for the Phase 11 deadline-aware campaign scheduler."""

from __future__ import annotations

import math
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from trading_bot.campaign import load_campaign_search_manifest
from trading_bot.config import SchedulerConfig
from trading_bot.evaluation.leaderboard import LeaderboardRow
from trading_bot.scheduler import (
    CampaignController,
    CampaignDB,
    CampaignState,
    DeadlineMode,
    DrainInputs,
    LaunchPriority,
    ResourceAllocator,
    RuntimeObservation,
    SchedulerResourcePolicy,
    SubprocessTrialRunner,
    TrialSpec,
    TrialState,
    build_screening_trial_specs,
    can_performance_prune,
    dynamic_drain_reserve_seconds,
    load_scheduler_runtime_policy,
    run_compressed_campaign_simulation,
    select_promotions,
)
from trading_bot.storage.local import LocalStorageBackend

ROOT = Path(__file__).parents[2]
MANIFEST_PATH = ROOT / "configs/campaigns/h200_tournament_v1.yaml"
POLICY_PATH = ROOT / "configs/campaigns/h200_scheduler_v1.yaml"


def _manifest():
    return load_campaign_search_manifest(MANIFEST_PATH)


def _policy():
    return load_scheduler_runtime_policy(POLICY_PATH)


def _controller(tmp_path: Path) -> tuple[CampaignDB, CampaignController]:
    manifest = _manifest()
    db = CampaignDB(tmp_path / "campaign.sqlite")
    controller = CampaignController(db, manifest, _policy())
    controller.bootstrap(
        started_at=2_000_000_000.0,
        deadline_at=2_000_000_000.0 + 48 * 3600,
    )
    return db, controller


def _row(trial_id: str, rank: int, *, eligible: bool = True) -> LeaderboardRow:
    return LeaderboardRow(
        rank=rank,
        trial_id=trial_id,
        eligible=eligible,
        disqualification_reasons=() if eligible else ("synthetic_invalid",),
        mean_rank_ic=0.10 / rank,
        icir=1.0 / rank,
        net_sharpe=1.5 / rank,
        calmar=1.0 / rank,
        maximum_drawdown=-0.05 * rank,
        sortino=1.8 / rank,
        es95=0.01 * rank,
        average_turnover=0.20,
        total_modeled_cost=0.001 * rank,
        positive_fold_fraction=1.0,
        dsr_probability=0.9,
        pbo_probability=0.1,
        attempted_trial_count=4,
    )


def test_scheduler_import_boundary_does_not_import_torch() -> None:
    command = (
        "import sys; import trading_bot.scheduler; "
        "assert 'torch' not in sys.modules, "
        "sorted(name for name in sys.modules if name.startswith('torch'))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_frozen_scheduler_policy_matches_shared_safety_defaults() -> None:
    policy = _policy()
    shared = SchedulerConfig()
    assert policy.initial_drain_reserve_minutes == shared.initial_drain_reserve_minutes == 90
    assert policy.max_trial_retries == shared.max_trial_retries == 2
    assert policy.worker_kill_grace_seconds == shared.kill_grace_seconds == 45
    thresholds = policy.deadline_thresholds
    assert (
        thresholds.broad_exploration_min_usable_hours,
        thresholds.normal_promotion_min_usable_hours,
        thresholds.restricted_expansion_min_usable_hours,
        thresholds.finalists_only_min_usable_hours,
        thresholds.avoid_expensive_min_usable_hours,
    ) == (24.0, 12.0, 6.0, 3.0, 1.5)
    assert policy.runtime_quantile == 0.9
    assert policy.pruning_grace_fraction == 0.5
    assert policy.resources.exclusive_gpu_trials
    assert policy.resources.max_gpu_trials == 1


def test_screening_plan_materializes_frozen_66_trial_breadth() -> None:
    manifest = _manifest()
    specs = build_screening_trial_specs(manifest, fallback_runtime_seconds=900.0)
    searchable = {
        item.family
        for item in manifest.architectures
        if item.searchable and manifest.screening_objective_id in item.objective_ids
    }
    assert len(specs) == 66
    assert len({spec.trial_id for spec in specs}) == 66
    assert {spec.family for spec in specs[: len(searchable)]} == searchable
    assert all(spec.stage == "screening" for spec in specs)
    assert all(spec.budget_fraction == 0.15 for spec in specs)
    assert all(len(spec.config_sha256) == 64 for spec in specs)


def test_sqlite_state_tables_transitions_and_immutable_trial(tmp_path: Path) -> None:
    db, controller = _controller(tmp_path)
    expected_tables = {
        "campaign",
        "trials",
        "metrics",
        "checkpoints",
        "runtime_stats",
        "events",
        "failures",
        "promotions",
        "resources",
    }
    assert {
        row[0]
        for row in db._connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        if not str(row[0]).startswith("sqlite_")
    } == expected_tables

    controller.transition_campaign(CampaignState.CALIBRATION)
    spec = TrialSpec(
        trial_id="immutable-1",
        family="mlp",
        scale="small",
        stage="calibration",
        budget_fraction=0.05,
        config={"hidden_features": 16},
        fallback_runtime_seconds=60.0,
    )
    controller.register_trial(spec)
    stored_hash = db.trial_spec(spec.trial_id).config_sha256
    controller.mark_running(spec.trial_id)
    controller.mark_evaluating(spec.trial_id, runtime_seconds=10.0)
    controller.mark_synced_complete(spec.trial_id)
    assert db.trial_state(spec.trial_id) == TrialState.COMPLETE
    assert db.trial_spec(spec.trial_id).config_sha256 == stored_hash
    with pytest.raises(ValueError, match="invalid trial transition"):
        db.transition_trial(spec.trial_id, TrialState.RUNNING)
    db.close()


def test_sqlite_snapshot_is_checksum_verified_and_restorable(tmp_path: Path) -> None:
    db, controller = _controller(tmp_path)
    backend = LocalStorageBackend(tmp_path / "durable")
    snapshot = controller.snapshot(backend, "campaign/scheduler.sqlite")
    assert snapshot.size > 0
    assert snapshot.checksum_sha256 is not None
    restored = backend.download(
        "campaign/scheduler.sqlite",
        tmp_path / "restored.sqlite",
        expected_sha256=snapshot.checksum_sha256,
    )
    with sqlite3.connect(restored) as connection:
        state = connection.execute("SELECT state FROM campaign").fetchone()[0]
    assert state == CampaignState.BOOTSTRAP.value
    db.close()


def test_runtime_estimator_uses_conservative_observed_full_budget_quantile(
    tmp_path: Path,
) -> None:
    db, controller = _controller(tmp_path)
    for index, runtime in enumerate((100.0, 300.0), start=1):
        spec = TrialSpec(
            trial_id=f"runtime-{index}",
            family="mlp",
            scale="small",
            stage="screening",
            budget_fraction=0.10,
            config={"context_length": 32},
            fallback_runtime_seconds=100.0,
        )
        controller.register_trial(spec)
        db.record_runtime(
            RuntimeObservation(
                trial_id=spec.trial_id,
                family=spec.family,
                scale=spec.scale,
                context_length=32,
                budget_fraction=0.10,
                runtime_seconds=runtime,
            )
        )
    target = TrialSpec(
        trial_id="runtime-target",
        family="mlp",
        scale="small",
        stage="promotion",
        budget_fraction=0.50,
        config={"context_length": 32},
        fallback_runtime_seconds=100.0,
    )
    assert controller.estimator.estimate_seconds(target) == pytest.approx(1725.0)
    db.close()


def test_dynamic_drain_reserve_accounts_for_eval_sync_and_unknown_throughput() -> None:
    policy = _policy()
    reserve = dynamic_drain_reserve_seconds(
        policy,
        DrainInputs(
            outstanding_evaluator_seconds=4000.0,
            unsynced_bytes=1_000_000_000,
            storage_bytes_per_second=1_000_000.0,
        ),
    )
    assert reserve == pytest.approx(6200.0)
    assert math.isinf(
        dynamic_drain_reserve_seconds(
            policy,
            DrainInputs(unsynced_bytes=1, storage_bytes_per_second=None),
        )
    )


def test_deadline_policy_refuses_optional_then_all_new_work(tmp_path: Path) -> None:
    db, controller = _controller(tmp_path)
    spec = TrialSpec(
        trial_id="optional-late",
        family="mlp",
        scale="small",
        stage="screening",
        budget_fraction=0.15,
        priority=LaunchPriority.OPTIONAL,
        config={},
        fallback_runtime_seconds=60.0,
    )
    controller.register_trial(spec)
    row = db.campaign_row(controller.campaign_id)
    deadline = float(row["deadline_at"])
    reserve = float(_policy().initial_drain_reserve_minutes * 60)
    restricted = controller.launch_decision(
        spec,
        now=deadline - reserve - 8 * 3600,
        drain_inputs=DrainInputs(),
    )
    assert restricted.mode == DeadlineMode.RESTRICTED_EXPANSION
    assert not restricted.allowed
    draining = controller.launch_decision(
        spec,
        now=deadline - reserve + 1,
        drain_inputs=DrainInputs(),
    )
    assert draining.mode == DeadlineMode.DRAIN
    assert not draining.allowed
    db.close()


def test_promotion_consumes_canonical_leaderboard_and_respects_grace() -> None:
    rows = (
        _row("a", 1),
        _row("b", 2, eligible=False),
        _row("c", 3),
        _row("d", 4),
    )
    promoted = select_promotions(rows, count=2)
    assert [row.trial_id for row in promoted] == ["a", "c"]
    assert not can_performance_prune(
        completed_full_budget_fraction=0.05,
        rung_full_budget_fraction=0.15,
        pruning_grace_fraction=0.5,
    )
    assert can_performance_prune(
        completed_full_budget_fraction=0.075,
        rung_full_budget_fraction=0.15,
        pruning_grace_fraction=0.5,
    )


def test_subprocess_runner_captures_logs_in_fresh_process(tmp_path: Path) -> None:
    runner = SubprocessTrialRunner(tmp_path / "logs")
    worker = runner.start(
        "worker-log",
        [
            sys.executable,
            "-c",
            "import sys; print('stdout-ok'); print('stderr-ok', file=sys.stderr)",
        ],
    )
    assert runner.wait(worker, timeout_seconds=5.0) == 0
    assert worker.stdout_path.read_text().strip() == "stdout-ok"
    assert worker.stderr_path.read_text().strip() == "stderr-ok"


def test_subprocess_runner_escalates_term_to_kill_for_hung_group(tmp_path: Path) -> None:
    runner = SubprocessTrialRunner(tmp_path / "logs")
    worker = runner.start(
        "worker-hang",
        [
            sys.executable,
            "-c",
            (
                "import signal,time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "print('ready', flush=True); time.sleep(30)"
            ),
        ],
    )
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if worker.stdout_path.exists() and "ready" in worker.stdout_path.read_text():
            break
        time.sleep(0.01)
    else:
        runner.terminate(worker, grace_seconds=0.0)
        raise AssertionError("hung worker did not initialize signal handler")
    return_code = runner.terminate(worker, grace_seconds=0.05)
    assert return_code == -signal.SIGKILL


def test_resource_allocator_keeps_gpu_exclusive_and_cpu_evaluator_independent() -> None:
    allocator = ResourceAllocator(_policy().resources)
    allocator.start_gpu_trial("gpu-a", is_tiny=True)
    assert not allocator.can_start_gpu_trial(is_tiny=True)
    allocator.start_cpu_evaluator("eval-a")
    assert allocator.active_gpu_trials == ("gpu-a",)
    assert allocator.active_cpu_evaluators == ("eval-a",)
    allocator.finish_cpu_evaluator("eval-a")
    allocator.finish_gpu_trial("gpu-a")


def test_tiny_gpu_concurrency_requires_positive_calibration_evidence() -> None:
    policy = SchedulerResourcePolicy(
        exclusive_gpu_trials=False,
        max_gpu_trials=2,
        max_cpu_evaluators=2,
        allow_concurrent_tiny_trials_after_calibration=True,
        minimum_tiny_trial_throughput_gain=1.10,
    )
    allocator = ResourceAllocator(policy)
    allocator.start_gpu_trial("tiny-a", is_tiny=True)
    assert not allocator.can_start_gpu_trial(is_tiny=True)
    allocator.record_tiny_concurrency_calibration(1.05)
    assert not allocator.can_start_gpu_trial(is_tiny=True)
    allocator.record_tiny_concurrency_calibration(1.20)
    assert allocator.can_start_gpu_trial(is_tiny=True)
    allocator.start_gpu_trial("tiny-b", is_tiny=True)
    assert not allocator.can_start_gpu_trial(is_tiny=True)
    allocator.finish_gpu_trial("tiny-b")
    allocator.finish_gpu_trial("tiny-a")


def test_compressed_campaign_reaches_durable_complete_after_restart_retry_and_drain(
    tmp_path: Path,
) -> None:
    backend = LocalStorageBackend(tmp_path / "durable")
    summary = run_compressed_campaign_simulation(
        _manifest(),
        _policy(),
        db_path=tmp_path / "simulation.sqlite",
        durable_backend=backend,
        snapshot_key="campaign/final.sqlite",
    )
    assert summary.final_state == CampaignState.COMPLETE
    assert summary.screening_registered == 66
    assert summary.evaluated_screening_trials == 8
    assert summary.promotion_rows == 4
    assert summary.retry_failures == 1
    assert summary.promotion_lineage_rows == 8
    assert summary.interrupted_at_drain > 0
    assert summary.late_mode == DeadlineMode.FINALISTS_ONLY
    assert not summary.optional_late_launch_allowed
    assert summary.finalist_late_launch_allowed
    assert backend.exists(summary.final_snapshot_key)

    restored = backend.download(summary.final_snapshot_key, tmp_path / "final-restored.sqlite")
    with sqlite3.connect(restored) as connection:
        state = connection.execute("SELECT state FROM campaign").fetchone()[0]
        failures = connection.execute("SELECT COUNT(*) FROM failures").fetchone()[0]
        promotions = connection.execute("SELECT COUNT(*) FROM promotions").fetchone()[0]
        bad_lineage = connection.execute(
            """
            SELECT COUNT(*)
            FROM promotions p
            JOIN trials parent ON parent.trial_id = p.parent_trial_id
            JOIN trials child ON child.trial_id = p.child_trial_id
            WHERE parent.root_trial_id != child.root_trial_id
            """
        ).fetchone()[0]
    assert state == CampaignState.COMPLETE.value
    assert failures == 1
    assert promotions == 8
    assert bad_lineage == 0
