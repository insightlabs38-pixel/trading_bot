"""Future-only label generation kept separate from the causal feature pipeline."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable


class LabelGenerationError(RuntimeError):
    """Raised when label inputs violate uniqueness or chronology contracts."""


@dataclass(frozen=True, slots=True)
class LabelObservation:
    security_id: str
    timestamp: datetime
    close: float

    def __post_init__(self) -> None:
        if not self.security_id.strip():
            raise ValueError("security_id must not be blank")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        if not math.isfinite(self.close) or self.close <= 0:
            raise ValueError("close must be finite and positive")


@dataclass(frozen=True, slots=True)
class LabelPolicy:
    horizons_minutes: tuple[int, ...] = (5, 15, 30, 60)
    reference_security_id: str | None = None
    include_reference_security: bool = False

    def __post_init__(self) -> None:
        if not self.horizons_minutes or any(value <= 0 for value in self.horizons_minutes):
            raise ValueError("label horizons must be non-empty and positive")
        if len(set(self.horizons_minutes)) != len(self.horizons_minutes):
            raise ValueError("label horizons must be unique")
        if self.reference_security_id is not None and not self.reference_security_id.strip():
            raise ValueError("reference_security_id must not be blank")


@dataclass(frozen=True, slots=True)
class LabelRow:
    security_id: str
    timestamp: datetime
    future_returns: dict[int, float]
    future_excess_returns: dict[int, float]
    directions: dict[int, int]
    cross_sectional_ranks: dict[int, float]
    future_volatility: dict[int, float]
    quantile_ranks: dict[int, float]


def generate_labels(
    observations: Iterable[LabelObservation],
    *,
    policy: LabelPolicy | None = None,
) -> tuple[LabelRow, ...]:
    """Generate future targets using exact timestamp endpoints and future-only paths."""
    policy = policy or LabelPolicy()
    rows = tuple(observations)
    lookup: dict[tuple[str, datetime], LabelObservation] = {}
    times_by_security: dict[str, list[datetime]] = defaultdict(list)
    for row in rows:
        identity = (row.security_id, row.timestamp)
        if identity in lookup:
            raise LabelGenerationError("duplicate security/timestamp label observation")
        lookup[identity] = row
        times_by_security[row.security_id].append(row.timestamp)
    for values in times_by_security.values():
        values.sort()

    reference_id = policy.reference_security_id
    prelim: dict[tuple[str, datetime], dict[str, dict[int, float] | dict[int, int]]] = {}
    for row in sorted(rows, key=lambda value: (value.timestamp, value.security_id)):
        if reference_id == row.security_id and not policy.include_reference_security:
            continue
        future_returns: dict[int, float] = {}
        excess_returns: dict[int, float] = {}
        directions: dict[int, int] = {}
        future_volatility: dict[int, float] = {}
        for horizon in policy.horizons_minutes:
            endpoint_time = row.timestamp + timedelta(minutes=horizon)
            endpoint = lookup.get((row.security_id, endpoint_time))
            if endpoint is None:
                continue
            future_return = endpoint.close / row.close - 1.0
            _require_finite_target(future_return, row, horizon, "future_return")
            future_returns[horizon] = future_return
            directions[horizon] = int(future_return > 0)
            volatility = _future_realized_volatility(
                row,
                endpoint_time,
                lookup,
                times_by_security[row.security_id],
            )
            _require_finite_target(volatility, row, horizon, "future_volatility")
            future_volatility[horizon] = volatility
            if reference_id is not None:
                reference_now = lookup.get((reference_id, row.timestamp))
                reference_future = lookup.get((reference_id, endpoint_time))
                if reference_now is not None and reference_future is not None:
                    reference_return = reference_future.close / reference_now.close - 1.0
                    _require_finite_target(
                        reference_return,
                        row,
                        horizon,
                        "reference_return",
                    )
                    excess = future_return - reference_return
                    _require_finite_target(excess, row, horizon, "future_excess_return")
                    excess_returns[horizon] = excess
        prelim[(row.security_id, row.timestamp)] = {
            "future_returns": future_returns,
            "future_excess_returns": excess_returns,
            "directions": directions,
            "future_volatility": future_volatility,
        }

    ranks: dict[tuple[str, datetime], dict[int, float]] = defaultdict(dict)
    by_time_horizon: dict[tuple[datetime, int], list[tuple[str, float]]] = defaultdict(list)
    for (security_id, timestamp), target in prelim.items():
        future_returns = target["future_returns"]
        assert isinstance(future_returns, dict)
        for horizon, value in future_returns.items():
            by_time_horizon[(timestamp, horizon)].append((security_id, float(value)))
    for (timestamp, horizon), values in by_time_horizon.items():
        calculated = _percentile_rank_pairs(values)
        for security_id, rank in calculated.items():
            ranks[(security_id, timestamp)][horizon] = rank

    output: list[LabelRow] = []
    for identity in sorted(prelim, key=lambda item: (item[1], item[0])):
        target = prelim[identity]
        future_returns = target["future_returns"]
        excess_returns = target["future_excess_returns"]
        directions = target["directions"]
        future_volatility = target["future_volatility"]
        assert isinstance(future_returns, dict)
        assert isinstance(excess_returns, dict)
        assert isinstance(directions, dict)
        assert isinstance(future_volatility, dict)
        rank_values = ranks.get(identity, {})
        output.append(
            LabelRow(
                security_id=identity[0],
                timestamp=identity[1],
                future_returns={int(k): float(v) for k, v in future_returns.items()},
                future_excess_returns={int(k): float(v) for k, v in excess_returns.items()},
                directions={int(k): int(v) for k, v in directions.items()},
                cross_sectional_ranks=dict(rank_values),
                future_volatility={int(k): float(v) for k, v in future_volatility.items()},
                quantile_ranks=dict(rank_values),
            )
        )
    return tuple(output)


def _require_finite_target(
    value: float,
    row: LabelObservation,
    horizon: int,
    name: str,
) -> None:
    if not math.isfinite(value):
        raise LabelGenerationError(
            f"non-finite {name} for {row.security_id}@{row.timestamp.isoformat()} "
            f"at horizon {horizon}"
        )


def _future_realized_volatility(
    start: LabelObservation,
    endpoint_time: datetime,
    lookup: dict[tuple[str, datetime], LabelObservation],
    ordered_times: list[datetime],
) -> float:
    selected = [
        timestamp
        for timestamp in ordered_times
        if start.timestamp <= timestamp <= endpoint_time
    ]
    if len(selected) < 3:
        return 0.0
    returns: list[float] = []
    for previous_time, current_time in zip(selected, selected[1:], strict=False):
        previous = lookup[(start.security_id, previous_time)]
        current = lookup[(start.security_id, current_time)]
        value = current.close / previous.close - 1.0
        if not math.isfinite(value):
            raise LabelGenerationError("future volatility path produced a non-finite return")
        returns.append(value)
    return statistics.pstdev(returns) if len(returns) > 1 else 0.0


def _percentile_rank_pairs(values: list[tuple[str, float]]) -> dict[str, float]:
    unique = sorted(set(value for _, value in values))
    if not unique:
        return {}
    if len(unique) == 1:
        return {security_id: 0.5 for security_id, _ in values}
    rank_by_value = {value: index / (len(unique) - 1) for index, value in enumerate(unique)}
    return {security_id: rank_by_value[value] for security_id, value in values}
