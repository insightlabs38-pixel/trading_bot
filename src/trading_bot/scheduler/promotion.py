"""Successive-halving helpers that consume the canonical Phase 6 leaderboard."""

from __future__ import annotations

from collections.abc import Sequence

from trading_bot.evaluation.leaderboard import LeaderboardRow


def can_performance_prune(
    *,
    completed_full_budget_fraction: float,
    rung_full_budget_fraction: float,
    pruning_grace_fraction: float,
) -> bool:
    if not 0.0 <= completed_full_budget_fraction <= 1.0:
        raise ValueError("completed budget fraction must lie in [0, 1]")
    if not 0.0 < rung_full_budget_fraction <= 1.0:
        raise ValueError("rung budget fraction must lie in (0, 1]")
    if not 0.0 < pruning_grace_fraction <= 1.0:
        raise ValueError("pruning grace fraction must lie in (0, 1]")
    return completed_full_budget_fraction >= rung_full_budget_fraction * pruning_grace_fraction


def select_promotions(
    leaderboard: Sequence[LeaderboardRow],
    *,
    count: int,
) -> tuple[LeaderboardRow, ...]:
    """Select eligible rows only, preserving the frozen Phase 6 rank hierarchy."""
    if count < 1:
        raise ValueError("promotion count must be positive")
    if not leaderboard:
        raise ValueError("promotion requires a non-empty canonical leaderboard")
    ranks = [row.rank for row in leaderboard]
    trial_ids = [row.trial_id for row in leaderboard]
    if len(set(ranks)) != len(ranks) or len(set(trial_ids)) != len(trial_ids):
        raise ValueError("leaderboard ranks and trial IDs must be unique")
    eligible = sorted((row for row in leaderboard if row.eligible), key=lambda row: row.rank)
    return tuple(eligible[:count])
