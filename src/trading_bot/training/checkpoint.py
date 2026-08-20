"""Atomic, checksummed checkpoint bundles for true training continuation."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer

from trading_bot.storage.base import fsync_directory, fsync_file, temporary_local_path
from trading_bot.training.contracts import TradingModel
from trading_bot.training.trainer import Scheduler, TrainingState


class CheckpointError(RuntimeError):
    """Base checkpoint publication or restore error."""


class CheckpointCorruptionError(CheckpointError):
    """Raised when durable checkpoint bytes or metadata fail integrity validation."""


class CheckpointResumeError(CheckpointError):
    """Raised when a checkpoint does not belong to the requested run identity."""


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    """Scientific identity that must match before state is restored."""

    model_config_hash: str
    training_config_hash: str
    dataset_id: str
    split_id: str

    def __post_init__(self) -> None:
        values = (
            self.model_config_hash,
            self.training_config_hash,
            self.dataset_id,
            self.split_id,
        )
        if any(not value.strip() for value in values):
            raise ValueError("checkpoint identity fields must not be blank")


@dataclass(frozen=True, slots=True)
class CheckpointRestore:
    """Restored trainer cursor plus precision/scaler state for the caller."""

    training_state: TrainingState
    precision_state: dict[str, Any]
    path: Path


class CheckpointManager:
    """Publish and restore immutable checkpoint directories plus latest/best pointers."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        model: TradingModel,
        optimizer: Optimizer,
        scheduler: Scheduler | None,
        training_state: TrainingState,
        identity: CheckpointIdentity,
        precision: str,
        precision_state: dict[str, Any] | None = None,
        is_best: bool = False,
    ) -> Path:
        """Write, reload-verify, checksum, and atomically publish one checkpoint bundle."""
        name = f"step-{training_state.optimizer_step:08d}"
        final_path = self.root / name
        if final_path.exists():
            raise CheckpointError(f"checkpoint already exists: {final_path}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{name}.tmp-", dir=str(self.root)))
        try:
            state_path = temporary / "state.pt"
            payload = {
                "schema_version": 1,
                "identity": asdict(identity),
                "training_state": asdict(training_state),
                "precision": precision,
                "precision_state": precision_state or {},
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "rng_state": _capture_rng_state(),
            }
            torch.save(payload, state_path)
            fsync_file(state_path)
            state_sha256 = _sha256_file(state_path)
            _validate_payload(_load_torch(state_path), identity)
            manifest = {
                "schema_version": 1,
                "checkpoint_file": "state.pt",
                "checkpoint_sha256": state_sha256,
                "optimizer_step": training_state.optimizer_step,
                "identity": asdict(identity),
                "precision": precision,
                "has_scheduler": scheduler is not None,
            }
            manifest_bytes = _canonical_json_bytes(manifest)
            manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            (temporary / "manifest.json").write_bytes(manifest_bytes)
            (temporary / "manifest.sha256").write_text(
                f"{manifest_sha256}\n",
                encoding="ascii",
            )
            fsync_file(temporary / "manifest.json")
            fsync_file(temporary / "manifest.sha256")
            os.replace(temporary, final_path)
            fsync_directory(self.root)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
        self._write_pointer("latest.json", final_path, training_state.optimizer_step, state_sha256)
        if is_best:
            self._write_pointer(
                "best.json", final_path, training_state.optimizer_step, state_sha256
            )
        return final_path

    def restore(
        self,
        checkpoint: str | Path,
        *,
        model: TradingModel,
        optimizer: Optimizer,
        scheduler: Scheduler | None,
        expected_identity: CheckpointIdentity,
    ) -> CheckpointRestore:
        """Verify identity/integrity before mutating any caller-owned training state."""
        path = self._resolve(checkpoint)
        manifest = _load_manifest(path)
        _assert_identity(manifest.get("identity"), expected_identity)
        state_path = path / "state.pt"
        expected_sha256 = manifest.get("checkpoint_sha256")
        if not isinstance(expected_sha256, str) or _sha256_file(state_path) != expected_sha256:
            raise CheckpointCorruptionError("checkpoint state checksum mismatch")
        payload = _load_torch(state_path)
        _validate_payload(payload, expected_identity)
        model.load_state_dict(cast(dict[str, Tensor], payload["model_state"]), strict=True)
        optimizer.load_state_dict(cast(dict[str, Any], payload["optimizer_state"]))
        scheduler_state = payload["scheduler_state"]
        if scheduler is None and scheduler_state is not None:
            raise CheckpointResumeError(
                "checkpoint contains scheduler state but no scheduler was supplied"
            )
        if scheduler is not None:
            if not isinstance(scheduler_state, dict):
                raise CheckpointResumeError("checkpoint does not contain required scheduler state")
            scheduler.load_state_dict(cast(dict[str, Any], scheduler_state))
        _restore_rng_state(cast(dict[str, object], payload["rng_state"]))
        training_state = _training_state(cast(dict[str, object], payload["training_state"]))
        precision_state = payload.get("precision_state")
        if not isinstance(precision_state, dict):
            raise CheckpointCorruptionError("checkpoint precision state is invalid")
        return CheckpointRestore(
            training_state=training_state,
            precision_state=cast(dict[str, Any], precision_state),
            path=path,
        )

    def latest(self) -> Path:
        return self._read_pointer("latest.json")

    def best(self) -> Path:
        return self._read_pointer("best.json")

    def _resolve(self, value: str | Path) -> Path:
        raw = Path(value)
        if raw.name == "latest":
            return self.latest()
        if raw.name == "best":
            return self.best()
        return raw if raw.is_absolute() else self.root / raw

    def _write_pointer(self, name: str, path: Path, step: int, state_sha256: str) -> None:
        destination = self.root / name
        temporary = temporary_local_path(destination)
        temporary.write_bytes(
            _canonical_json_bytes(
                {
                    "checkpoint": path.name,
                    "optimizer_step": step,
                    "checkpoint_sha256": state_sha256,
                }
            )
        )
        fsync_file(temporary)
        os.replace(temporary, destination)
        fsync_directory(self.root)

    def _read_pointer(self, name: str) -> Path:
        pointer_path = self.root / name
        try:
            value = json.loads(pointer_path.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointCorruptionError(f"invalid checkpoint pointer {name}") from exc
        checkpoint = value.get("checkpoint") if isinstance(value, dict) else None
        if not isinstance(checkpoint, str) or Path(checkpoint).name != checkpoint:
            raise CheckpointCorruptionError(f"invalid checkpoint pointer {name}")
        path = self.root / checkpoint
        if not path.is_dir():
            raise CheckpointCorruptionError(f"checkpoint pointer target is missing: {checkpoint}")
        return path


def _capture_rng_state() -> dict[str, object]:
    python_version, python_state, python_gauss = random.getstate()
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_version": python_version,
        "python_state": list(python_state),
        "python_gauss": python_gauss,
        "numpy_algorithm": algorithm,
        "numpy_keys": torch.tensor(keys.astype(np.int64), dtype=torch.int64),
        "numpy_position": int(position),
        "numpy_has_gauss": int(has_gauss),
        "numpy_cached_gaussian": float(cached_gaussian),
    }


def _restore_rng_state(state: dict[str, object]) -> None:
    torch_cpu = state.get("torch_cpu")
    if not isinstance(torch_cpu, Tensor):
        raise CheckpointCorruptionError("checkpoint torch RNG state is invalid")
    torch.set_rng_state(torch_cpu.cpu())
    torch_cuda = state.get("torch_cuda")
    if torch.cuda.is_available() and isinstance(torch_cuda, list) and torch_cuda:
        torch.cuda.set_rng_state_all(cast(list[Tensor], torch_cuda))

    python_version = state.get("python_version")
    python_state = state.get("python_state")
    python_gauss = state.get("python_gauss")
    if (
        not isinstance(python_version, int)
        or not isinstance(python_state, list)
        or not all(isinstance(value, int) for value in python_state)
    ):
        raise CheckpointCorruptionError("checkpoint Python RNG state is invalid")
    if python_gauss is not None and not isinstance(python_gauss, float):
        raise CheckpointCorruptionError("checkpoint Python Gaussian RNG state is invalid")
    random.setstate(
        (
            python_version,
            tuple(cast(list[int], python_state)),
            cast(float | None, python_gauss),
        )
    )

    algorithm = state.get("numpy_algorithm")
    keys = state.get("numpy_keys")
    position = state.get("numpy_position")
    has_gauss = state.get("numpy_has_gauss")
    cached_gaussian = state.get("numpy_cached_gaussian")
    if (
        not isinstance(algorithm, str)
        or not isinstance(keys, Tensor)
        or not isinstance(position, int)
        or not isinstance(has_gauss, int)
        or not isinstance(cached_gaussian, float)
    ):
        raise CheckpointCorruptionError("checkpoint NumPy RNG state is invalid")
    np.random.set_state(
        (
            algorithm,
            keys.cpu().numpy().astype(np.uint32),
            position,
            has_gauss,
            cached_gaussian,
        )
    )


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        manifest_bytes = (path / "manifest.json").read_bytes()
        expected = (path / "manifest.sha256").read_text(encoding="ascii").strip().lower()
        value = json.loads(manifest_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointCorruptionError("invalid checkpoint manifest") from exc
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise CheckpointCorruptionError("invalid checkpoint manifest checksum")
    if hashlib.sha256(manifest_bytes).hexdigest() != expected:
        raise CheckpointCorruptionError("checkpoint manifest checksum mismatch")
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise CheckpointCorruptionError("invalid checkpoint manifest schema")
    return cast(dict[str, object], value)


def _load_torch(path: Path) -> dict[str, object]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointCorruptionError(f"unable to load checkpoint state: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckpointCorruptionError("checkpoint payload must be a mapping")
    return cast(dict[str, object], value)


def _validate_payload(payload: dict[str, object], identity: CheckpointIdentity) -> None:
    if payload.get("schema_version") != 1:
        raise CheckpointCorruptionError("invalid checkpoint payload schema")
    _assert_identity(payload.get("identity"), identity)
    for key in (
        "training_state",
        "model_state",
        "optimizer_state",
        "rng_state",
        "precision_state",
    ):
        if key not in payload:
            raise CheckpointCorruptionError(f"checkpoint payload missing {key}")


def _assert_identity(value: object, expected: CheckpointIdentity) -> None:
    if not isinstance(value, dict) or value != asdict(expected):
        raise CheckpointResumeError("checkpoint identity does not match requested run")


def _training_state(value: dict[str, object]) -> TrainingState:
    return TrainingState(
        optimizer_step=_require_int(value, "optimizer_step"),
        micro_step=_require_int(value, "micro_step"),
        samples_seen=_require_int(value, "samples_seen"),
        last_loss=_optional_float(value, "last_loss"),
        stopped_early=_require_bool(value, "stopped_early"),
    )


def _require_int(value: dict[str, object], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool):
        raise CheckpointCorruptionError(f"invalid checkpoint training state {key}")
    return item


def _require_bool(value: dict[str, object], key: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise CheckpointCorruptionError(f"invalid checkpoint training state {key}")
    return item


def _optional_float(value: dict[str, object], key: str) -> float | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, int | float) or isinstance(item, bool):
        raise CheckpointCorruptionError(f"invalid checkpoint training state {key}")
    return float(item)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
