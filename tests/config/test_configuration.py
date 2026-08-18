"""Tests for the validated configuration system."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from trading_bot.config import (
    AIRepairConfig,
    AppConfig,
    MissingEnvironmentVariableError,
    PaperLiveRiskConfig,
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
    assert config.evaluation.spread_bps == 1.0
    assert not config.paper_live_risk.enabled


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


def test_environment_interpolation_supports_secret_and_endpoint_values() -> None:
    payload = {
        "endpoint": "${S3_ENDPOINT}",
        "secret": "${TOKEN}",
    }
    assert interpolate_environment(
        payload,
        environ={"S3_ENDPOINT": "https://s3.example", "TOKEN": "secret"},
    ) == {"endpoint": "https://s3.example", "secret": "secret"}


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


def test_canonical_serialization_is_stable_and_order_independent() -> None:
    config = load_config(EXAMPLE_CONFIG, environ={})
    payload = config.model_dump(mode="python")
    payload["model"]["parameters"] = {"layers": 2, "hidden_dim": 128}
    reordered = AppConfig.model_validate(payload)
    assert config_to_canonical_json(config) == config_to_canonical_json(reordered)
    assert json.loads(config_to_canonical_json(config))["campaign"]["campaign_id"] == "dev_smoke"


def test_model_parameters_must_be_json_compatible() -> None:
    config = load_config(EXAMPLE_CONFIG, environ={})
    payload = config.model_dump(mode="python")
    payload["model"]["parameters"] = {"bad": object()}
    with pytest.raises(ValidationError):
        AppConfig.model_validate(payload)


def test_ai_repair_provider_is_not_hardcoded() -> None:
    config = AIRepairConfig(
        enabled=True,
        provider="example-provider",
        model="repair-model-v1",
        api_base_url="https://api.example.test",
        api_key="secret",
    )
    assert config.provider == "example-provider"
    assert config.model == "repair-model-v1"


def test_enabled_ai_repair_requires_explicit_provider_model_and_key() -> None:
    with pytest.raises(ValidationError, match="provider settings"):
        AIRepairConfig(enabled=True)


def test_disabled_paper_live_risk_does_not_invent_numeric_limits() -> None:
    config = PaperLiveRiskConfig()
    assert not config.enabled
    assert config.max_position_weight is None
    assert config.daily_loss_limit_fraction is None


def test_enabled_paper_live_risk_requires_explicit_limits() -> None:
    with pytest.raises(ValidationError, match="explicit limits"):
        PaperLiveRiskConfig(enabled=True)


def test_enabled_paper_live_risk_requires_safety_gates() -> None:
    payload = {
        "enabled": True,
        "max_position_weight": 0.02,
        "max_gross_exposure": 1.0,
        "max_abs_net_exposure": 0.2,
        "max_leverage": 1.0,
        "max_order_nav_fraction": 0.01,
        "max_participation_rate": 0.05,
        "daily_loss_limit_fraction": 0.02,
        "drawdown_stop_fraction": 0.05,
        "max_data_age_seconds": 90,
        "model_inference_timeout_seconds": 10,
        "max_outstanding_orders": 50,
        "kill_switch_enabled": False,
    }
    with pytest.raises(ValidationError, match="safety gates"):
        PaperLiveRiskConfig.model_validate(payload)


def test_evaluation_cost_accounting_requires_spread() -> None:
    config = load_config(EXAMPLE_CONFIG, environ={})
    payload = config.model_dump(mode="python")
    del payload["evaluation"]["spread_bps"]
    with pytest.raises(ValidationError, match="spread_bps"):
        AppConfig.model_validate(payload)
