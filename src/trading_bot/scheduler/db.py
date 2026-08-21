"""Durable SQLite campaign state for the Phase 11 controller."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any

from trading_bot.scheduler.types import (
    CampaignState,
    ResourceSample,
    RuntimeObservation,
    TrialSpec,
    TrialState,
    require_campaign_transition,
    require_trial_transition,
)
from trading_bot.storage.base import StorageBackend, StorageObjectMetadata, sha256_file

_SCHEMA_VERSION = 1


class CampaignDatabaseError(RuntimeError):
    """Raised when durable scheduler state violates its immutable contract."""


class CampaignDB:
    """Single-writer SQLite authority with append-only scientific lineage tables."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = FULL")
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self._initialize_schema()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> CampaignDB:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _initialize_schema(self) -> None:
        current = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if current not in {0, _SCHEMA_VERSION}:
            raise CampaignDatabaseError(
                f"unsupported campaign DB schema version {current}; expected {_SCHEMA_VERSION}"
            )
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS campaign (
                campaign_id TEXT PRIMARY KEY,
                manifest_sha256 TEXT NOT NULL,
                manifest_json TEXT NOT NULL,
                state TEXT NOT NULL,
                started_at REAL NOT NULL,
                deadline_at REAL NOT NULL,
                drain_reserve_seconds REAL NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trials (
                trial_id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL REFERENCES campaign(campaign_id),
                parent_trial_id TEXT REFERENCES trials(trial_id),
                root_trial_id TEXT NOT NULL,
                family TEXT NOT NULL,
                scale TEXT NOT NULL,
                stage TEXT NOT NULL,
                state TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                budget_fraction REAL NOT NULL,
                priority TEXT NOT NULL,
                config_json TEXT NOT NULL,
                config_sha256 TEXT NOT NULL,
                fallback_runtime_seconds REAL NOT NULL,
                worker_pid INTEGER,
                stdout_path TEXT,
                stderr_path TEXT,
                created_at REAL NOT NULL,
                started_at REAL,
                finished_at REAL
            );

            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trial_id TEXT NOT NULL REFERENCES trials(trial_id),
                name TEXT NOT NULL,
                value REAL NOT NULL,
                step INTEGER,
                recorded_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trial_id TEXT NOT NULL REFERENCES trials(trial_id),
                storage_key TEXT NOT NULL,
                checksum_sha256 TEXT NOT NULL,
                step INTEGER NOT NULL,
                status TEXT NOT NULL,
                recorded_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runtime_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trial_id TEXT NOT NULL REFERENCES trials(trial_id),
                family TEXT NOT NULL,
                scale TEXT NOT NULL,
                context_length INTEGER,
                precision TEXT NOT NULL,
                budget_fraction REAL NOT NULL,
                runtime_seconds REAL NOT NULL,
                samples_per_second REAL,
                gpu_utilization_percent REAL,
                peak_vram_bytes INTEGER,
                recorded_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL REFERENCES campaign(campaign_id),
                trial_id TEXT REFERENCES trials(trial_id),
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                recorded_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trial_id TEXT NOT NULL REFERENCES trials(trial_id),
                failure_class TEXT NOT NULL,
                retryable INTEGER NOT NULL,
                message TEXT NOT NULL,
                recorded_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS promotions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_trial_id TEXT NOT NULL REFERENCES trials(trial_id),
                child_trial_id TEXT NOT NULL REFERENCES trials(trial_id),
                from_stage TEXT NOT NULL,
                to_stage TEXT NOT NULL,
                leaderboard_rank INTEGER NOT NULL,
                reason TEXT NOT NULL,
                recorded_at REAL NOT NULL,
                UNIQUE(parent_trial_id, child_trial_id)
            );

            CREATE TABLE IF NOT EXISTS resources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL REFERENCES campaign(campaign_id),
                trial_id TEXT REFERENCES trials(trial_id),
                cpu_percent REAL,
                gpu_utilization_percent REAL,
                peak_vram_bytes INTEGER,
                recorded_at REAL NOT NULL
            );
            """
        )
        self._connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        self._connection.commit()

    def create_campaign(
        self,
        *,
        campaign_id: str,
        manifest_sha256: str,
        manifest_json: str,
        started_at: float,
        deadline_at: float,
        drain_reserve_seconds: float,
    ) -> None:
        if deadline_at <= started_at:
            raise ValueError("campaign deadline must be after start")
        if drain_reserve_seconds <= 0 or drain_reserve_seconds >= deadline_at - started_at:
            raise ValueError("drain reserve must be positive and shorter than campaign duration")
        now = time.time()
        try:
            self._connection.execute(
                """
                INSERT INTO campaign(
                    campaign_id, manifest_sha256, manifest_json, state, started_at,
                    deadline_at, drain_reserve_seconds, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    campaign_id,
                    manifest_sha256,
                    manifest_json,
                    CampaignState.BOOTSTRAP.value,
                    started_at,
                    deadline_at,
                    drain_reserve_seconds,
                    now,
                    now,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise CampaignDatabaseError(f"campaign {campaign_id!r} already exists") from exc
        self._connection.commit()
        self.record_event(campaign_id, "campaign_created", {"deadline_at": deadline_at})

    def campaign_row(self, campaign_id: str) -> sqlite3.Row:
        row = self._connection.execute(
            "SELECT * FROM campaign WHERE campaign_id = ?", (campaign_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown campaign {campaign_id!r}")
        return row

    def campaign_state(self, campaign_id: str) -> CampaignState:
        return CampaignState(str(self.campaign_row(campaign_id)["state"]))

    def transition_campaign(self, campaign_id: str, target: CampaignState) -> None:
        current = self.campaign_state(campaign_id)
        require_campaign_transition(current, target)
        self._connection.execute(
            "UPDATE campaign SET state = ?, updated_at = ? WHERE campaign_id = ?",
            (target.value, time.time(), campaign_id),
        )
        self._connection.commit()
        self.record_event(
            campaign_id,
            "campaign_state_changed",
            {"from": current.value, "to": target.value},
        )

    def update_drain_reserve(self, campaign_id: str, reserve_seconds: float) -> None:
        if reserve_seconds <= 0:
            raise ValueError("drain reserve must be positive")
        self._connection.execute(
            "UPDATE campaign SET drain_reserve_seconds = ?, updated_at = ? WHERE campaign_id = ?",
            (reserve_seconds, time.time(), campaign_id),
        )
        self._connection.commit()

    def insert_trial(self, campaign_id: str, spec: TrialSpec) -> None:
        if spec.parent_trial_id is not None:
            parent = self.trial_row(spec.parent_trial_id)
            if str(parent["campaign_id"]) != campaign_id:
                raise CampaignDatabaseError("trial parent belongs to a different campaign")
            if str(parent["root_trial_id"]) != spec.effective_root_trial_id:
                raise CampaignDatabaseError("child trial root lineage does not match parent")
        try:
            self._connection.execute(
                """
                INSERT INTO trials(
                    trial_id, campaign_id, parent_trial_id, root_trial_id, family, scale,
                    stage, state, attempt, budget_fraction, priority, config_json,
                    config_sha256, fallback_runtime_seconds, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    spec.trial_id,
                    campaign_id,
                    spec.parent_trial_id,
                    spec.effective_root_trial_id,
                    spec.family,
                    spec.scale,
                    spec.stage,
                    TrialState.PENDING.value,
                    spec.attempt,
                    spec.budget_fraction,
                    spec.priority.value,
                    spec.canonical_config_json,
                    spec.config_sha256,
                    spec.fallback_runtime_seconds,
                    time.time(),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise CampaignDatabaseError(f"unable to insert immutable trial {spec.trial_id!r}") from exc
        self._connection.commit()
        self.record_event(campaign_id, "trial_registered", {"trial_id": spec.trial_id}, spec.trial_id)

    def trial_row(self, trial_id: str) -> sqlite3.Row:
        row = self._connection.execute(
            "SELECT * FROM trials WHERE trial_id = ?", (trial_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown trial {trial_id!r}")
        return row

    def trial_state(self, trial_id: str) -> TrialState:
        return TrialState(str(self.trial_row(trial_id)["state"]))

    def trial_spec(self, trial_id: str) -> TrialSpec:
        row = self.trial_row(trial_id)
        return TrialSpec(
            trial_id=str(row["trial_id"]),
            family=str(row["family"]),
            scale=str(row["scale"]),
            stage=str(row["stage"]),
            budget_fraction=float(row["budget_fraction"]),
            priority=str(row["priority"]),
            config=json.loads(str(row["config_json"])),
            parent_trial_id=(
                str(row["parent_trial_id"]) if row["parent_trial_id"] is not None else None
            ),
            root_trial_id=str(row["root_trial_id"]),
            attempt=int(row["attempt"]),
            fallback_runtime_seconds=float(row["fallback_runtime_seconds"]),
        )

    def transition_trial(self, trial_id: str, target: TrialState) -> None:
        row = self.trial_row(trial_id)
        current = TrialState(str(row["state"]))
        require_trial_transition(current, target)
        started_at = row["started_at"]
        finished_at = row["finished_at"]
        now = time.time()
        if target == TrialState.STARTING and started_at is None:
            started_at = now
        if target in {
            TrialState.COMPLETE,
            TrialState.PRUNED,
            TrialState.RETRYABLE_FAILURE,
            TrialState.TERMINAL_FAILURE,
            TrialState.INTERRUPTED,
        }:
            finished_at = now
        self._connection.execute(
            "UPDATE trials SET state = ?, started_at = ?, finished_at = ? WHERE trial_id = ?",
            (target.value, started_at, finished_at, trial_id),
        )
        self._connection.commit()
        self.record_event(
            str(row["campaign_id"]),
            "trial_state_changed",
            {"from": current.value, "to": target.value},
            trial_id,
        )

    def attach_worker(
        self, trial_id: str, *, pid: int, stdout_path: str, stderr_path: str
    ) -> None:
        self._connection.execute(
            "UPDATE trials SET worker_pid = ?, stdout_path = ?, stderr_path = ? WHERE trial_id = ?",
            (pid, stdout_path, stderr_path, trial_id),
        )
        self._connection.commit()

    def record_metric(self, trial_id: str, name: str, value: float, *, step: int | None = None) -> None:
        self._connection.execute(
            "INSERT INTO metrics(trial_id, name, value, step, recorded_at) VALUES (?, ?, ?, ?, ?)",
            (trial_id, name, value, step, time.time()),
        )
        self._connection.commit()

    def record_checkpoint(
        self,
        trial_id: str,
        *,
        storage_key: str,
        checksum_sha256: str,
        step: int,
        status: str,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO checkpoints(
                trial_id, storage_key, checksum_sha256, step, status, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (trial_id, storage_key, checksum_sha256, step, status, time.time()),
        )
        self._connection.commit()

    def record_runtime(self, observation: RuntimeObservation) -> None:
        self._connection.execute(
            """
            INSERT INTO runtime_stats(
                trial_id, family, scale, context_length, precision, budget_fraction,
                runtime_seconds, samples_per_second, gpu_utilization_percent,
                peak_vram_bytes, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                observation.trial_id,
                observation.family,
                observation.scale,
                observation.context_length,
                observation.precision,
                observation.budget_fraction,
                observation.runtime_seconds,
                observation.samples_per_second,
                observation.gpu_utilization_percent,
                observation.peak_vram_bytes,
                time.time(),
            ),
        )
        self._connection.commit()

    def runtime_observations(self, *, family: str, scale: str) -> tuple[RuntimeObservation, ...]:
        rows = self._connection.execute(
            "SELECT * FROM runtime_stats WHERE family = ? AND scale = ? ORDER BY id",
            (family, scale),
        ).fetchall()
        return tuple(
            RuntimeObservation(
                trial_id=str(row["trial_id"]),
                family=str(row["family"]),
                scale=str(row["scale"]),
                context_length=(
                    int(row["context_length"]) if row["context_length"] is not None else None
                ),
                precision=str(row["precision"]),
                budget_fraction=float(row["budget_fraction"]),
                runtime_seconds=float(row["runtime_seconds"]),
                samples_per_second=(
                    float(row["samples_per_second"])
                    if row["samples_per_second"] is not None
                    else None
                ),
                gpu_utilization_percent=(
                    float(row["gpu_utilization_percent"])
                    if row["gpu_utilization_percent"] is not None
                    else None
                ),
                peak_vram_bytes=(
                    int(row["peak_vram_bytes"]) if row["peak_vram_bytes"] is not None else None
                ),
            )
            for row in rows
        )

    def record_failure(
        self, trial_id: str, *, failure_class: str, retryable: bool, message: str
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO failures(trial_id, failure_class, retryable, message, recorded_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (trial_id, failure_class, int(retryable), message, time.time()),
        )
        self._connection.commit()

    def record_promotion(
        self,
        *,
        parent_trial_id: str,
        child_trial_id: str,
        from_stage: str,
        to_stage: str,
        leaderboard_rank: int,
        reason: str,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO promotions(
                parent_trial_id, child_trial_id, from_stage, to_stage,
                leaderboard_rank, reason, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                parent_trial_id,
                child_trial_id,
                from_stage,
                to_stage,
                leaderboard_rank,
                reason,
                time.time(),
            ),
        )
        self._connection.commit()

    def record_resource(
        self, campaign_id: str, sample: ResourceSample, *, trial_id: str | None = None
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO resources(
                campaign_id, trial_id, cpu_percent, gpu_utilization_percent,
                peak_vram_bytes, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                campaign_id,
                trial_id,
                sample.cpu_percent,
                sample.gpu_utilization_percent,
                sample.peak_vram_bytes,
                time.time(),
            ),
        )
        self._connection.commit()

    def record_event(
        self,
        campaign_id: str,
        event_type: str,
        payload: dict[str, Any],
        trial_id: str | None = None,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO events(campaign_id, trial_id, event_type, payload_json, recorded_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                campaign_id,
                trial_id,
                event_type,
                json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str),
                time.time(),
            ),
        )
        self._connection.commit()

    def rows(self, table: str) -> tuple[sqlite3.Row, ...]:
        allowed = {
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
        if table not in allowed:
            raise ValueError(f"unsupported scheduler table {table!r}")
        return tuple(self._connection.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall())

    def snapshot_to_storage(
        self, backend: StorageBackend, key: str
    ) -> StorageObjectMetadata:
        """Create a transactionally consistent SQLite backup and checksum-verified upload."""
        with tempfile.TemporaryDirectory(prefix="trading-bot-campaign-snapshot-") as directory:
            snapshot_path = Path(directory) / "campaign.sqlite"
            destination = sqlite3.connect(snapshot_path)
            try:
                self._connection.backup(destination)
                destination.commit()
            finally:
                destination.close()
            checksum = sha256_file(snapshot_path)
            metadata = backend.upload(snapshot_path, key, expected_sha256=checksum)
            if not backend.verify_checksum(key, checksum):
                raise CampaignDatabaseError("durable campaign snapshot checksum verification failed")
            return metadata
