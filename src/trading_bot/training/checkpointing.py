"""Atomic continuation checkpoints with identity validation and corruption detection."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class CheckpointError(RuntimeError):
    """Base error for checkpoint save/load failures."""


class CheckpointCorruptionError(CheckpointError):
    """Raised when checkpoint bytes fail the recorded integrity check."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised when a checkpoint does not match the requested run identity."""


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    model_config_sha256: str
    training_config_sha256: str
    dataset_version: str
    split_version: str

    def __post_init__(self) -> None:
        for value in (self.model_config_sha256, self.training_config_sha256):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError("config hashes must be lowercase SHA-256 hex digests")
        if not self.dataset_version or not self.split_version:
            raise ValueError("dataset_version and split_version must not be blank")


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    checkpoint_id: str
    path: str
    size_bytes: int
    sha256: str
    step: int
    cursor: int
    precision: str
    identity: CheckpointIdentity

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class RestoredCheckpoint:
    record: CheckpointRecord
    step: int
    cursor: int
    precision: str


def save_checkpoint(
    directory: str | Path,
    checkpoint_id: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    cursor: int,
    precision: str,
    identity: CheckpointIdentity,
    lr_scheduler: object | None = None,
    scaler: object | None = None,
    is_best: bool = False,
) -> CheckpointRecord:
    if not checkpoint_id.strip():
        raise ValueError("checkpoint_id must not be blank")
    if step < 0 or cursor < 0:
        raise ValueError("step and cursor must be non-negative")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    final_path = root / f"{checkpoint_id}.pt"
    if final_path.exists():
        raise CheckpointError(f"checkpoint already exists: {final_path}")
    payload = {
        "schema_version": 1,
        "checkpoint_id": checkpoint_id,
        "step": step,
        "cursor": cursor,
        "precision": precision,
        "identity": asdict(identity),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "lr_scheduler_state": _optional_state_dict(lr_scheduler),
        "scaler_state": _optional_state_dict(scaler),
        "rng_state": _capture_rng_state(),
    }
    fd, temporary_name = tempfile.mkstemp(prefix=f".{checkpoint_id}.", suffix=".tmp", dir=root)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        _fsync_file(temporary)
        size = temporary.stat().st_size
        checksum = _sha256_file(temporary)
        os.replace(temporary, final_path)
        _fsync_directory(root)
    finally:
        temporary.unlink(missing_ok=True)
    record = CheckpointRecord(
        checkpoint_id=checkpoint_id,
        path=final_path.name,
        size_bytes=size,
        sha256=checksum,
        step=step,
        cursor=cursor,
        precision=precision,
        identity=identity,
    )
    _write_record(root / f"{checkpoint_id}.json", record)
    _write_pointer(root / "latest.json", checkpoint_id)
    if is_best:
        _write_pointer(root / "best.json", checkpoint_id)
    return record


def load_checkpoint(
    directory: str | Path,
    checkpoint_id: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    expected_identity: CheckpointIdentity,
    lr_scheduler: object | None = None,
    scaler: object | None = None,
    restore_rng: bool = True,
) -> RestoredCheckpoint:
    root = Path(directory)
    record = read_checkpoint_record(root, checkpoint_id)
    if record.identity != expected_identity:
        raise CheckpointCompatibilityError(
            f"checkpoint identity mismatch: expected {expected_identity}, got {record.identity}"
        )
    checkpoint_path = root / record.path
    _verify_checkpoint_file(checkpoint_path, record)
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        message = f"unable to deserialize checkpoint {checkpoint_id}: {exc}"
        raise CheckpointCorruptionError(message) from exc
    if payload.get("schema_version") != 1 or payload.get("checkpoint_id") != checkpoint_id:
        raise CheckpointCorruptionError("checkpoint payload metadata does not match its record")
    if payload.get("identity") != asdict(expected_identity):
        raise CheckpointCompatibilityError("checkpoint payload identity does not match expected run")
    model.load_state_dict(payload["model_state"])
    _restore_optional_state(lr_scheduler, payload.get("lr_scheduler_state"), "lr_scheduler")
    _restore_optional_state(scaler, payload.get("scaler_state"), "scaler")
    optimizer.load_state_dict(payload["optimizer_state"])
    if restore_rng:
        _restore_rng_state(payload["rng_state"])
    return RestoredCheckpoint(
        record=record,
        step=int(payload["step"]),
        cursor=int(payload["cursor"]),
        precision=str(payload["precision"]),
    )


def read_checkpoint_record(directory: str | Path, checkpoint_id: str) -> CheckpointRecord:
    path = Path(directory) / f"{checkpoint_id}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["identity"] = CheckpointIdentity(**payload["identity"])
        return CheckpointRecord(**payload)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise CheckpointCorruptionError(f"invalid checkpoint record {path}: {exc}") from exc


def resolve_checkpoint_pointer(directory: str | Path, name: str = "latest") -> str:
    if name not in {"latest", "best"}:
        raise ValueError("checkpoint pointer must be latest or best")
    path = Path(directory) / f"{name}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        checkpoint_id = str(payload["checkpoint_id"])
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"invalid {name} checkpoint pointer: {exc}") from exc
    if not checkpoint_id:
        raise CheckpointError(f"invalid {name} checkpoint pointer")
    return checkpoint_id


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _optional_state_dict(value: object | None) -> dict[str, Any] | None:
    if value is None:
        return None
    state_dict = getattr(value, "state_dict", None)
    if not callable(state_dict):
        raise CheckpointError("stateful object must expose state_dict()")
    return state_dict()


def _restore_optional_state(value: object | None, state: dict[str, Any] | None, name: str) -> None:
    if state is None:
        return
    if value is None:
        message = f"checkpoint contains {name} state but no object was supplied"
        raise CheckpointCompatibilityError(message)
    load_state_dict = getattr(value, "load_state_dict", None)
    if not callable(load_state_dict):
        raise CheckpointCompatibilityError(f"{name} object must expose load_state_dict()")
    load_state_dict(state)


def _write_record(path: Path, record: CheckpointRecord) -> None:
    _atomic_write_text(path, record.canonical_json())


def _write_pointer(path: Path, checkpoint_id: str) -> None:
    _atomic_write_text(
        path,
        json.dumps({"checkpoint_id": checkpoint_id}, sort_keys=True, separators=(",", ":")),
    )


def _atomic_write_text(path: Path, content: str) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(content, encoding="utf-8")
        _fsync_file(temporary)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_checkpoint_file(path: Path, record: CheckpointRecord) -> None:
    if not path.is_file():
        raise CheckpointCorruptionError(f"checkpoint file is missing: {path}")
    if path.stat().st_size != record.size_bytes:
        raise CheckpointCorruptionError(f"checkpoint size mismatch: {path}")
    if _sha256_file(path) != record.sha256:
        raise CheckpointCorruptionError(f"checkpoint checksum mismatch: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
