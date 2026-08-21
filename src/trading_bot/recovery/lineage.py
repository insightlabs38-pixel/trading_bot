"""Immutable child-trial lineage for validated repair proposals."""

from __future__ import annotations

import copy
from typing import cast

from pydantic import JsonValue

from trading_bot.scheduler.types import TrialSpec


def derive_repaired_child(parent: TrialSpec, *, proposal_sha256: str) -> TrialSpec:
    valid_digest = len(proposal_sha256) == 64 and all(
        character in "0123456789abcdef" for character in proposal_sha256
    )
    if not valid_digest:
        raise ValueError("proposal_sha256 must be a lowercase SHA-256 digest")
    child_config = cast(dict[str, JsonValue], copy.deepcopy(parent.config))
    child_config["repair_provenance"] = {"proposal_sha256": proposal_sha256}
    return TrialSpec(
        trial_id=f"{parent.trial_id}-repair-{proposal_sha256[:10]}",
        family=parent.family,
        scale=parent.scale,
        stage=parent.stage,
        budget_fraction=parent.budget_fraction,
        priority=parent.priority,
        config=child_config,
        parent_trial_id=parent.trial_id,
        root_trial_id=parent.effective_root_trial_id,
        attempt=parent.attempt + 1,
        fallback_runtime_seconds=parent.fallback_runtime_seconds,
    )
