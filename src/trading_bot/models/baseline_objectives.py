"""Shared objective-config adapter for all neural Phase 7 baselines."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from trading_bot.config import ObjectiveConfig
from trading_bot.models.baseline_common import BaselineTargetNames
from trading_bot.training.contracts import ModelOutput, TrainingBatch
from trading_bot.training.trainer import LossFunction

_COMPONENTS = frozenset({"expected_return", "rank_score", "direction_probability"})


def build_baseline_loss(
    objective: ObjectiveConfig,
    targets: BaselineTargetNames,
) -> LossFunction:
    """Build one loss callable from the same validated objective schema for every neural family."""
    if objective.kind == "excess_return":
        if objective.loss not in {"mse", "huber"}:
            raise ValueError("excess-return neural baselines require mse or huber loss")
        return lambda output, batch: _return_loss(output, batch, objective.loss, targets)
    if objective.kind == "direction":
        if objective.loss != "bce":
            raise ValueError("direction neural baselines require bce loss")
        return lambda output, batch: _direction_loss(output, batch, targets)
    if objective.kind == "ranking":
        if objective.loss != "pairwise_rank":
            raise ValueError("ranking neural baselines require pairwise_rank loss")
        return lambda output, batch: _ranking_loss(output, batch, targets)
    if objective.kind == "multitask":
        if objective.loss != "composite":
            raise ValueError("multitask neural baselines require composite loss")
        unknown = set(objective.task_weights) - _COMPONENTS
        if unknown:
            raise ValueError(f"unsupported multitask baseline components: {sorted(unknown)}")
        if not any(weight > 0 for weight in objective.task_weights.values()):
            raise ValueError("multitask baseline requires at least one positive task weight")
        return lambda output, batch: _multitask_loss(output, batch, objective, targets)
    raise ValueError(
        "distributional objective is not implemented for Phase 7 reference baselines"
    )


def _return_loss(
    output: ModelOutput,
    batch: TrainingBatch,
    loss: str,
    targets: BaselineTargetNames,
) -> Tensor:
    prediction = _scalar_head(output.require("expected_return"), batch.batch_size, "expected_return")
    target = _scalar_target(batch, targets.return_target)
    if loss == "huber":
        return F.smooth_l1_loss(prediction, target)
    return F.mse_loss(prediction, target)


def _direction_loss(
    output: ModelOutput,
    batch: TrainingBatch,
    targets: BaselineTargetNames,
) -> Tensor:
    probability = _scalar_head(
        output.require("direction_probability"),
        batch.batch_size,
        "direction_probability",
    )
    target = _scalar_target(batch, targets.direction_target)
    if not bool(((target == 0) | (target == 1)).all().item()):
        raise ValueError("direction targets must be encoded as 0/1")
    if not bool(((probability >= 0) & (probability <= 1)).all().item()):
        raise ValueError("direction probabilities must lie in [0, 1]")
    return F.binary_cross_entropy(probability, target)


def _ranking_loss(
    output: ModelOutput,
    batch: TrainingBatch,
    targets: BaselineTargetNames,
) -> Tensor:
    score = _scalar_head(output.require("rank_score"), batch.batch_size, "rank_score")
    target = _scalar_target(batch, targets.return_target)
    score_delta = score[:, None] - score[None, :]
    target_delta = target[:, None] - target[None, :]
    same_timestamp = batch.timestamps_ns[:, None] == batch.timestamps_ns[None, :]
    upper_triangle = torch.triu(
        torch.ones(
            (batch.batch_size, batch.batch_size),
            dtype=torch.bool,
            device=score.device,
        ),
        diagonal=1,
    )
    valid = same_timestamp & upper_triangle & (target_delta != 0)
    if not bool(valid.any().item()):
        return score.sum() * 0.0
    desired_sign = torch.sign(target_delta[valid])
    return F.softplus(-desired_sign * score_delta[valid]).mean()


def _multitask_loss(
    output: ModelOutput,
    batch: TrainingBatch,
    objective: ObjectiveConfig,
    targets: BaselineTargetNames,
) -> Tensor:
    components: list[Tensor] = []
    weights: list[float] = []
    for name, weight in objective.task_weights.items():
        if weight <= 0:
            continue
        if name == "expected_return":
            components.append(_return_loss(output, batch, "huber", targets))
        elif name == "rank_score":
            components.append(_ranking_loss(output, batch, targets))
        elif name == "direction_probability":
            components.append(_direction_loss(output, batch, targets))
        else:
            raise ValueError(f"unsupported multitask baseline component {name!r}")
        weights.append(float(weight))
    if not components:
        raise ValueError("multitask baseline produced no active loss components")
    total = components[0] * weights[0]
    for component, weight in zip(components[1:], weights[1:], strict=True):
        total = total + component * weight
    return total / sum(weights)


def _scalar_target(batch: TrainingBatch, name: str) -> Tensor:
    target = batch.targets.get(name)
    if target is None:
        raise KeyError(f"batch does not provide target {name!r}")
    return _scalar_head(target, batch.batch_size, name)


def _scalar_head(value: Tensor, batch_size: int, name: str) -> Tensor:
    if value.shape == (batch_size, 1):
        value = value.squeeze(-1)
    if value.shape != (batch_size,):
        raise ValueError(f"{name} must contain one scalar per sample")
    return value.float()
