"""Atomic checksummed fitted-state checkpoints for classical Phase 7 baselines."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from trading_bot.models.baseline_classical import ClassicalBaseline
from trading_bot.storage.base import fsync_directory, fsync_file

_STATE_FILE = "state.pkl"
_MANIFEST_FILE = "manifest.json"
_MANIFEST_SHA_FILE = "manifest.sha256"
_FORMAT = "phase7_classical_baseline_checkpoint"
_SCHEMA_VERSION = 1


class ClassicalCheckpointError(RuntimeError):
    """Raised when a classical baseline checkpoint fails identity/integrity validation."""


@dataclass(frozen=True, slots=True)
class ClassicalCheckpointIdentity:
    model_config_hash: str
    dataset_id: str
    split_id: str

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (self.model_config_hash, self.dataset_id, self.split_id)
        ):
            raise ValueError("classical checkpoint identity fields must not be blank")


def save_classical_checkpoint(
    model: ClassicalBaseline,
    destination: str | Path,
    *,
    identity: ClassicalCheckpointIdentity,
) -> Path:
    """Publish a fitted estimator atomically after durable checksums are written."""
    destination = Path(destination)
    if destination.exists():
        raise ClassicalCheckpointError(f"checkpoint destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = model.checkpoint_payload()
    state_sha = hashlib.sha256(payload).hexdigest()
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=str(destination.parent))
    )
    try:
        state_path = temporary / _STATE_FILE
        state_path.write_bytes(payload)
        fsync_file(state_path)
        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "format": _FORMAT,
            "family": model.family,
            "objective": model.objective.model_dump(mode="json"),
            "identity": asdict(identity),
            "state_file": {
                "name": _STATE_FILE,
                "size": len(payload),
                "sha256": state_sha,
            },
        }
        manifest_bytes = _canonical_json_bytes(manifest)
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        (temporary / _MANIFEST_FILE).write_bytes(manifest_bytes)
        (temporary / _MANIFEST_SHA_FILE).write_text(f"{manifest_sha}\n", encoding="ascii")
        fsync_file(temporary / _MANIFEST_FILE)
        fsync_file(temporary / _MANIFEST_SHA_FILE)
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return destination


def restore_classical_checkpoint(
    model: ClassicalBaseline,
    checkpoint: str | Path,
    *,
    expected_identity: ClassicalCheckpointIdentity,
) -> None:
    """Verify durable bytes and scientific identity before deserializing fitted state."""
    root = Path(checkpoint)
    manifest_bytes = _verified_manifest_bytes(root)
    try:
        value = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise ClassicalCheckpointError("invalid classical checkpoint manifest JSON") from exc
    manifest = _validated_manifest(value)
    if manifest["family"] != model.family:
        raise ClassicalCheckpointError("classical checkpoint family mismatch")
    if manifest["objective"] != model.objective.model_dump(mode="json"):
        raise ClassicalCheckpointError("classical checkpoint objective mismatch")
    if manifest["identity"] != asdict(expected_identity):
        raise ClassicalCheckpointError("classical checkpoint scientific identity mismatch")
    state_meta = cast(dict[str, object], manifest["state_file"])
    state_path = root / _STATE_FILE
    try:
        payload = state_path.read_bytes()
    except OSError as exc:
        raise ClassicalCheckpointError("classical checkpoint state file is missing") from exc
    expected_size = state_meta.get("size")
    expected_sha = state_meta.get("sha256")
    if (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size <= 0
        or len(payload) != expected_size
        or not isinstance(expected_sha, str)
        or hashlib.sha256(payload).hexdigest() != expected_sha
    ):
        raise ClassicalCheckpointError("classical checkpoint state checksum/size mismatch")
    model.restore_checkpoint_payload(payload)


def _verified_manifest_bytes(root: Path) -> bytes:
    try:
        manifest_bytes = (root / _MANIFEST_FILE).read_bytes()
        expected_sha = (root / _MANIFEST_SHA_FILE).read_text(encoding="ascii").strip().lower()
    except OSError as exc:
        raise ClassicalCheckpointError("classical checkpoint manifest files are missing") from exc
    if (
        len(expected_sha) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha)
        or hashlib.sha256(manifest_bytes).hexdigest() != expected_sha
    ):
        raise ClassicalCheckpointError("classical checkpoint manifest checksum mismatch")
    return manifest_bytes


def _validated_manifest(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ClassicalCheckpointError("classical checkpoint manifest must be an object")
    manifest = cast(dict[str, Any], value)
    if manifest.get("schema_version") != _SCHEMA_VERSION or manifest.get("format") != _FORMAT:
        raise ClassicalCheckpointError("unsupported classical checkpoint schema")
    family = manifest.get("family")
    objective = manifest.get("objective")
    identity = manifest.get("identity")
    state_file = manifest.get("state_file")
    if (
        not isinstance(family, str)
        or not family.strip()
        or not isinstance(objective, dict)
        or not isinstance(identity, dict)
        or not isinstance(state_file, dict)
        or state_file.get("name") != _STATE_FILE
    ):
        raise ClassicalCheckpointError("invalid classical checkpoint manifest fields")
    return manifest


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")
