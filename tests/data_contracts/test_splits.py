"""Tests for immutable chronological split manifests and final-holdout protection."""

from __future__ import annotations

import json
from datetime import date

import pytest
from pydantic import ValidationError

from trading_bot.data.splits import (
    DateRange,
    FinalHoldoutAccess,
    FinalHoldoutAccessError,
    SplitManifest,
    WalkForwardFold,
)


def manifest() -> SplitManifest:
    return SplitManifest(
        split_version="splits-v1",
        dataset_version="dataset-v7",
        folds=(
            WalkForwardFold(
                fold_id="fold-1",
                train=DateRange(start=date(2018, 1, 1), end=date(2019, 12, 31)),
                validation=DateRange(start=date(2020, 1, 1), end=date(2020, 6, 30)),
            ),
            WalkForwardFold(
                fold_id="fold-2",
                train=DateRange(start=date(2018, 1, 1), end=date(2020, 6, 30)),
                validation=DateRange(start=date(2020, 7, 1), end=date(2020, 12, 31)),
            ),
        ),
        final_holdout_id="final-v1",
        final_holdout=DateRange(start=date(2021, 1, 1), end=date(2021, 12, 31)),
    )


def test_manifest_supports_expanding_walk_forward_training() -> None:
    value = manifest()
    assert value.fold("fold-1").train.start == date(2018, 1, 1)
    assert value.fold("fold-2").train.start == date(2018, 1, 1)
    assert value.fold("fold-2").train.end == date(2020, 6, 30)


def test_train_must_precede_validation() -> None:
    with pytest.raises(ValidationError, match="training period must end"):
        WalkForwardFold(
            fold_id="bad",
            train=DateRange(start=date(2020, 1, 1), end=date(2020, 6, 1)),
            validation=DateRange(start=date(2020, 6, 1), end=date(2020, 12, 1)),
        )


def test_validation_periods_must_move_forward_without_overlap() -> None:
    payload = manifest().model_dump(mode="python")
    payload["folds"] = tuple(reversed(payload["folds"]))
    with pytest.raises(ValidationError, match="chronological validation order"):
        SplitManifest.model_validate(payload)


def test_final_holdout_must_be_after_all_routine_periods() -> None:
    payload = manifest().model_dump(mode="python")
    payload["final_holdout"] = DateRange(start=date(2020, 12, 1), end=date(2021, 12, 31))
    with pytest.raises(ValidationError, match="final holdout must begin after"):
        SplitManifest.model_validate(payload)


def test_final_holdout_is_inaccessible_in_routine_mode() -> None:
    value = manifest()
    with pytest.raises(FinalHoldoutAccessError, match="inaccessible"):
        value.final_holdout_range()
    assert value.final_holdout_range(
        access=FinalHoldoutAccess.FINAL_EVALUATION
    ) == DateRange(start=date(2021, 1, 1), end=date(2021, 12, 31))


def test_routine_partition_does_not_reveal_final_holdout() -> None:
    value = manifest()
    assert value.routine_partition(date(2020, 8, 1)) == ("fold-2", "validation")
    assert value.routine_partition(date(2021, 6, 1)) is None


def test_split_manifest_hash_and_serialization_are_stable() -> None:
    value = manifest()
    reconstructed = SplitManifest.model_validate(json.loads(value.canonical_json()))
    assert reconstructed == value
    assert reconstructed.split_sha256() == value.split_sha256()
    assert reconstructed.dataset_version == "dataset-v7"
    assert reconstructed.final_holdout_id == "final-v1"


def test_unknown_fields_and_duplicate_fold_ids_are_rejected() -> None:
    payload = manifest().model_dump(mode="python")
    payload["mystery"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SplitManifest.model_validate(payload)
    payload = manifest().model_dump(mode="python")
    payload["folds"] = (payload["folds"][0], payload["folds"][0])
    with pytest.raises(ValidationError, match="fold IDs must be unique"):
        SplitManifest.model_validate(payload)
