"""Configuration loading, environment interpolation, and manifest-safe serialization."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]
from pydantic import SecretStr

from trading_bot.config.schemas import AppConfig

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigLoadError(ValueError):
    """Raised when configuration input cannot be loaded or interpolated."""


class MissingEnvironmentVariableError(ConfigLoadError):
    """Raised when a required environment variable is not defined."""


def interpolate_environment(value: Any, environ: Mapping[str, str] | None = None) -> Any:
    """Recursively substitute ``${VAR}`` and ``${VAR:-default}`` placeholders."""
    environment = os.environ if environ is None else environ

    if isinstance(value, dict):
        return {key: interpolate_environment(item, environment) for key, item in value.items()}
    if isinstance(value, list):
        return [interpolate_environment(item, environment) for item in value]
    if isinstance(value, tuple):
        return tuple(interpolate_environment(item, environment) for item in value)
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in environment:
            return environment[name]
        if default is not None:
            return default
        raise MissingEnvironmentVariableError(f"environment variable {name!r} is not set")

    return _ENV_PATTERN.sub(replace, value)


def load_config(path: str | Path, environ: Mapping[str, str] | None = None) -> AppConfig:
    """Load YAML, interpolate environment variables, and validate the full config."""
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigLoadError(f"unable to read config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigLoadError("configuration root must be a mapping")

    interpolated = interpolate_environment(raw, environ)
    return AppConfig.model_validate(interpolated)


def _redact_secrets(value: Any) -> Any:
    """Convert models to JSON-safe values while redacting secret material."""
    if isinstance(value, SecretStr):
        return "***REDACTED***"
    if isinstance(value, dict):
        return {key: _redact_secrets(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_secrets(item) for item in value]
    return value


def config_to_manifest_dict(config: AppConfig) -> dict[str, Any]:
    """Return a manifest-safe configuration representation with secrets redacted."""
    payload = config.model_dump(mode="python")
    return cast(dict[str, Any], _redact_secrets(payload))


def config_to_canonical_json(config: AppConfig) -> str:
    """Serialize a config deterministically for manifests and later hashing."""
    return json.dumps(
        config_to_manifest_dict(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
