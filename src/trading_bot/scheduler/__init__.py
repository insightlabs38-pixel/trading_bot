"""CPU-safe H200 campaign scheduler primitives and compressed simulation."""

from trading_bot.scheduler.controller import CampaignController
from trading_bot.scheduler.db import CampaignDatabaseError, CampaignDB
from trading_bot.scheduler.planner import build_screening_trial_specs
from trading_bot.scheduler.policy import (
    DeadlineThresholds,
    SchedulerPolicyError,
    SchedulerResourcePolicy,
    SchedulerRuntimePolicy,
    load_scheduler_runtime_policy,
)
from trading_bot.scheduler.process import SubprocessTrialRunner, WorkerProcess
from trading_bot.scheduler.promotion import can_performance_prune, select_promotions
from trading_bot.scheduler.resources import ResourceAllocator
from trading_bot.scheduler.runtime import (
    DeadlineMode,
    LaunchDecision,
    RuntimeEstimator,
    deadline_mode,
    dynamic_drain_reserve_seconds,
    launch_decision,
    usable_seconds,
)
from trading_bot.scheduler.simulation import (
    SimulationSummary,
    run_compressed_campaign_simulation,
)
from trading_bot.scheduler.types import (
    CampaignState,
    DrainInputs,
    LaunchPriority,
    ResourceSample,
    RuntimeObservation,
    TrialSpec,
    TrialState,
)

__all__ = [
    "CampaignController",
    "CampaignDB",
    "CampaignDatabaseError",
    "CampaignState",
    "DeadlineMode",
    "DeadlineThresholds",
    "DrainInputs",
    "LaunchDecision",
    "LaunchPriority",
    "ResourceAllocator",
    "ResourceSample",
    "RuntimeEstimator",
    "RuntimeObservation",
    "SchedulerPolicyError",
    "SchedulerResourcePolicy",
    "SchedulerRuntimePolicy",
    "SimulationSummary",
    "SubprocessTrialRunner",
    "TrialSpec",
    "TrialState",
    "WorkerProcess",
    "build_screening_trial_specs",
    "can_performance_prune",
    "deadline_mode",
    "dynamic_drain_reserve_seconds",
    "launch_decision",
    "load_scheduler_runtime_policy",
    "run_compressed_campaign_simulation",
    "select_promotions",
    "usable_seconds",
]
