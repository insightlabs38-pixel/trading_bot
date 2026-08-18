"""Validated configuration system for trading-bot workflows."""

from trading_bot.config.loader import (
    ConfigLoadError,
    MissingEnvironmentVariableError,
    config_to_canonical_json,
    config_to_manifest_dict,
    interpolate_environment,
    load_config,
)
from trading_bot.config.schemas import (
    AIRepairConfig,
    AppConfig,
    CampaignConfig,
    DatasetConfig,
    EvaluationConfig,
    ModelConfig,
    NotificationsConfig,
    ObjectiveConfig,
    PaperLiveRiskConfig,
    PreprocessingConfig,
    SchedulerConfig,
    StorageConfig,
    TrainingConfig,
)

__all__ = [
    "AIRepairConfig",
    "AppConfig",
    "CampaignConfig",
    "ConfigLoadError",
    "DatasetConfig",
    "EvaluationConfig",
    "MissingEnvironmentVariableError",
    "ModelConfig",
    "NotificationsConfig",
    "ObjectiveConfig",
    "PaperLiveRiskConfig",
    "PreprocessingConfig",
    "SchedulerConfig",
    "StorageConfig",
    "TrainingConfig",
    "config_to_canonical_json",
    "config_to_manifest_dict",
    "interpolate_environment",
    "load_config",
]
