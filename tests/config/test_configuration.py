"""Tests for the validated configuration system."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from trading_bot.config import (
    AppConfig,
    MissingEnvironmentVariableError,
    config_to_canonical_json,
    config_to_manifest_dict,
    interpolate_environment,
    load_config,
)

EXAMPLE_CONFIG = Path(__file__).parents[2] / "configs" / "examples" / "minimal.yaml"


def test_example_config_loads() -> None:
    config = load_config(EXAMPLE_CONFIG, environ={})
    assert isinstance(config, AppConfig)
    assert config.dataset.primary_horizons_minutes == (15, 30)
    assert config.storage.root_path == "/tmp/trading-bot-data"


def test_config_round_trip() -> None:
    config = load_config(EXAMPLE_CONFIG, environ={})
    reconstructed = AppConfig.model_validate(config.model_dump(mode="python"))
    assert reconstructed == config


def test_unknown_top_level_field_is_rejected(tmp_path: Path) -> None:
    text = EXAMPLE_CONFIG.read_text(encoding="utf-8") + "\nunknown_section: true\n"
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_config(path, environ={})


def test_unknown_nested_field_is_rejected(tmp_path: Path) -> None:
    text = EXAMPLE_CONFIG.read_text(encoding="utf-8").replace(
        "  variant: smoke\n", "  variant: smoke\n  mystery_flag: true\n"
    )
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_config(path, environ={})


def test_environment_substitution_and_default(tmp_path: Path) -> None:
    path = tmp_path / "env.yaml"
    path.write_text(EXAMPLE_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    config = load_config(
        path,
        environ={"TRADING_DATA_ROOT": "/data", "TRADING_SCRATCH": "/scratch"},
    )
    assert config.storage.root_path == "/data"
    assert config.preprocessing.scratch_path == "/scratch"


def test_missing_environment_variable_is_rejected() -> None:
    with pytest.raises(MissingEnvironmentVariableError, match="MISSING"):
        interpolate_environment("${MISSING}", environ={})


def test_environment_interpolation_supports_embedded_values() -> None:
    assert (
        interpolate_environment("s3://${BUCKET}/dataset", environ={"BUCKET": "research"})
        == "s3://research/dataset"
    )


def test_enabled_notification_requires_webhook() -> None:
    config = load_config(EXAMPLE_CONFIG, environ={})
    payload = config.model_dump(mode="python")
    payload["notifications"]["enabled"] = True
    with pytest.raises(ValidationError, match="webhook_url"):
        AppConfig.model_validate(payload)


def test_objective_horizon_must_exist_in_dataset() -> None:
    config = load_config(EXAMPLE_CONFIG, environ={})
    payload = config.model_dump(mode="python")
    payload["objective"]["horizons_minutes"] = (15, 120)
    with pytest.raises(ValidationError, match="objective horizons"):
        AppConfig.model_validate(payload)


def test_config_is_frozen() -> None:
    config = load_config(EXAMPLE_CONFIG, environ={})
    with pytest.raises(ValidationError):
        config.training.batch_size = 128  # type: ignore[misc]


def test_manifest_serialization_redacts_secrets(tmp_path: Path) -> None:
    text = EXAMPLE_CONFIG.read_text(encoding="utf-8").replace(
        "notifications:\n  enabled: false",
        "notifications:\n  enabled: true\n  webhook_url: ${DISCORD_WEBHOOK}",
    )
    path = tmp_path / "secret.yaml"
    path.write_text(text, encoding="utf-8")
    config = load_config(path, environ={"DISCORD_WEBHOOK": "https://secret.example/token"})
    payload = config_to_manifest_dict(config)
    assert payload["notifications"]["webhook_url"] == "***REDACTED***"
    assert "secret.example" not in config_to_canonical_json(config)


def test_canonical_serialization_is_stable() -> None:
    config = load_config(EXAMPLE_CONFIG, environ={})
    first = config_to_canonical_json(config)
    second = config_to_canonical_json(config)
    assert first == second
    assert json.loads(first)["campaign"]["campaign_id"] == "dev_smoke"
