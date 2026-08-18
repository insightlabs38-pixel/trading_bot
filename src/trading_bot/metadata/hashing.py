"""Deterministic hashing helpers for configuration and model identity."""

from __future__ import annotations

import hashlib
import json

from trading_bot.config import AppConfig, config_to_canonical_json
from trading_bot.config.schemas import ModelConfig
from trading_bot.metadata.identifiers import ModelConfigId


def sha256_text(value: str) -> str:
    """Return the lowercase SHA-256 hex digest of UTF-8 text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def config_sha256(config: AppConfig) -> str:
    """Hash the canonical, secret-redacted representation of an application config."""
    return sha256_text(config_to_canonical_json(config))


def model_config_to_canonical_json(config: ModelConfig) -> str:
    """Serialize model configuration deterministically for content identity."""
    return json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def model_config_id(config: ModelConfig) -> ModelConfigId:
    """Build a stable content-derived model configuration identifier."""
    digest = sha256_text(model_config_to_canonical_json(config))
    return ModelConfigId(f"model_{digest}")
