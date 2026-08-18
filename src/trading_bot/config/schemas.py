"""Strongly validated configuration schemas for the trading system."""

from __future__ import annotations

from typing import Literal

from pydantic import (
    Field,
    JsonValue,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    field_validator,
    model_validator,
)

from trading_bot.config.base import FrozenConfigModel


class StrictConfigModel(FrozenConfigModel):
    """Semantic alias for project configuration sections."""


class StorageConfig(StrictConfigModel):
    """Local or S3-compatible storage settings."""

    backend: Literal["local", "s3"]
    root_path: str | None = None
    endpoint_url: str | None = None
    bucket: str | None = None
    region: str | None = None
    prefix: str = ""
    access_key: SecretStr | None = None
    secret_key: SecretStr | None = None
    session_token: SecretStr | None = None
    multipart_threshold_mb: PositiveInt = 128

    @model_validator(mode="after")
    def validate_backend_requirements(self) -> StorageConfig:
        """Require an unambiguous location for the selected backend."""
        if self.backend == "local":
            if not self.root_path:
                raise ValueError("local storage requires root_path")
            if self.bucket is not None:
                raise ValueError("local storage must not define bucket")
        elif not self.bucket:
            raise ValueError("s3 storage requires bucket")
        return self


class DatasetConfig(StrictConfigModel):
    """Dataset identity and medium-frequency research scope."""

    version: str = Field(min_length=1)
    universe_target_size: PositiveInt = Field(default=1000, le=5000)
    base_bar_frequency: Literal["1m"] = "1m"
    primary_horizons_minutes: tuple[PositiveInt, ...] = (15, 30)
    auxiliary_horizons_minutes: tuple[PositiveInt, ...] = (5, 60)
    final_holdout_id: str = Field(min_length=1)

    @field_validator("primary_horizons_minutes", "auxiliary_horizons_minutes")
    @classmethod
    def horizons_must_be_unique(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Reject duplicated horizons because they create ambiguous objectives."""
        if not value:
            raise ValueError("at least one horizon is required")
        if len(set(value)) != len(value):
            raise ValueError("horizons must be unique")
        return value


class PreprocessingConfig(StrictConfigModel):
    """CPU preprocessing execution settings."""

    worker_count: PositiveInt = 8
    compression: Literal["zstd"] = "zstd"
    parquet_target_mb: PositiveInt = Field(default=512, le=2048)
    scratch_path: str = Field(min_length=1)
    resume: bool = True
    verify_checksums: bool = True


ModelHead = Literal[
    "return",
    "rank",
    "direction",
    "volatility",
    "uncertainty",
    "quantiles",
]


class ModelConfig(StrictConfigModel):
    """Architecture identity plus model-specific JSON-compatible parameters."""

    family: str = Field(min_length=1)
    variant: str = Field(default="default", min_length=1)
    heads: tuple[ModelHead, ...] = ("return", "rank", "volatility")
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("heads")
    @classmethod
    def heads_must_be_unique(cls, value: tuple[ModelHead, ...]) -> tuple[ModelHead, ...]:
        if not value:
            raise ValueError("at least one model head is required")
        if len(set(value)) != len(value):
            raise ValueError("model heads must be unique")
        return value


class TrainingConfig(StrictConfigModel):
    """Common trainer settings shared by all architectures."""

    batch_size: PositiveInt
    gradient_accumulation_steps: PositiveInt = 1
    max_steps: PositiveInt
    learning_rate: PositiveFloat
    weight_decay: float = Field(default=0.0, ge=0.0)
    gradient_clip_norm: PositiveFloat | None = 1.0
    precision: Literal["fp32", "bf16", "fp8"] = "bf16"
    compile_mode: Literal["off", "default", "reduce-overhead"] = "off"
    seed: int = Field(default=42, ge=0)


class ObjectiveConfig(StrictConfigModel):
    """Prediction objective and target horizons."""

    kind: Literal[
        "excess_return",
        "direction",
        "ranking",
        "multitask",
        "distributional",
    ]
    horizons_minutes: tuple[PositiveInt, ...]
    loss: Literal["huber", "mse", "bce", "pairwise_rank", "quantile", "composite"]
    task_weights: dict[str, float] = Field(default_factory=dict)

    @field_validator("horizons_minutes")
    @classmethod
    def objective_horizons_must_be_unique(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("at least one objective horizon is required")
        if len(set(value)) != len(value):
            raise ValueError("objective horizons must be unique")
        return value

    @field_validator("task_weights")
    @classmethod
    def task_weights_must_be_nonnegative(cls, value: dict[str, float]) -> dict[str, float]:
        if any(weight < 0 for weight in value.values()):
            raise ValueError("task weights must be non-negative")
        return value

    @model_validator(mode="after")
    def validate_multitask_weights(self) -> ObjectiveConfig:
        if self.kind == "multitask" and not self.task_weights:
            raise ValueError("multitask objective requires task_weights")
        return self


class EvaluationConfig(StrictConfigModel):
    """Frozen economic-evaluation assumptions used during a campaign."""

    annualization_days: PositiveInt = 252
    risk_free_rate_annual: float = 0.0
    fee_bps: float = Field(ge=0.0)
    spread_bps: float = Field(ge=0.0)
    slippage_bps: float = Field(ge=0.0)
    impact_bps: float = Field(ge=0.0)
    cost_stress_multipliers: tuple[PositiveFloat, ...] = (1.0, 1.25, 1.5, 2.0)
    latency_stress_seconds: tuple[float, ...] = (0.0, 0.25, 1.0, 5.0, 15.0, 30.0)
    minimum_positive_fold_fraction: float = Field(default=0.70, ge=0.0, le=1.0)

    @field_validator("cost_stress_multipliers")
    @classmethod
    def cost_stress_grid_must_be_valid(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value:
            raise ValueError("cost stress grid must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("cost stress multipliers must be unique")
        if 1.0 not in value:
            raise ValueError("cost stress multipliers must include the 1.0 baseline")
        return value

    @field_validator("latency_stress_seconds")
    @classmethod
    def latency_values_must_be_valid(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value:
            raise ValueError("latency stress grid must not be empty")
        if any(delay < 0 for delay in value):
            raise ValueError("latency stress values must be non-negative")
        if len(set(value)) != len(value):
            raise ValueError("latency stress values must be unique")
        return value


class CampaignConfig(StrictConfigModel):
    """Top-level experiment campaign identity and wall-clock budget."""

    campaign_id: str = Field(min_length=1)
    max_duration_hours: PositiveFloat = Field(default=48.0, le=168.0)
    seeds: tuple[int, ...] = (17, 29, 43)
    mandatory_families: tuple[str, ...] = ()
    optional_families: tuple[str, ...] = ()

    @field_validator("seeds")
    @classmethod
    def seeds_must_be_unique_and_nonnegative(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("at least one campaign seed is required")
        if any(seed < 0 for seed in value):
            raise ValueError("campaign seeds must be non-negative")
        if len(set(value)) != len(value):
            raise ValueError("campaign seeds must be unique")
        return value

    @model_validator(mode="after")
    def experiment_pools_must_not_overlap(self) -> CampaignConfig:
        overlap = set(self.mandatory_families) & set(self.optional_families)
        if overlap:
            raise ValueError(f"mandatory and optional families overlap: {sorted(overlap)}")
        return self


class SchedulerConfig(StrictConfigModel):
    """Deadline, heartbeat, retry, and recovery settings."""

    heartbeat_interval_seconds: PositiveInt = 15
    heartbeat_timeout_seconds: PositiveInt = 180
    kill_grace_seconds: PositiveInt = 45
    max_trial_retries: int = Field(default=2, ge=0, le=10)
    max_oom_fallbacks: int = Field(default=2, ge=0, le=10)
    initial_drain_reserve_minutes: PositiveInt = 90
    circuit_breaker_failures: PositiveInt = 3
    circuit_breaker_window_minutes: PositiveInt = 10

    @model_validator(mode="after")
    def timeout_must_exceed_interval(self) -> SchedulerConfig:
        if self.heartbeat_timeout_seconds <= self.heartbeat_interval_seconds:
            raise ValueError("heartbeat_timeout_seconds must exceed heartbeat_interval_seconds")
        return self


class NotificationsConfig(StrictConfigModel):
    """Human-facing notification settings."""

    enabled: bool = False
    provider: Literal["discord"] = "discord"
    webhook_url: SecretStr | None = None
    summary_interval_minutes: PositiveInt = 240
    spool_path: str = "/scratch/notifications"

    @model_validator(mode="after")
    def webhook_required_when_enabled(self) -> NotificationsConfig:
        if self.enabled and self.webhook_url is None:
            raise ValueError("webhook_url is required when notifications are enabled")
        return self


class AIRepairConfig(StrictConfigModel):
    """Sandboxed AI repair escalation settings without a hardcoded provider."""

    enabled: bool = False
    provider: str | None = None
    model: str | None = None
    api_base_url: str | None = None
    api_key: SecretStr | None = None
    primary_timeout_seconds: PositiveInt = 45
    reasoning_timeout_seconds: PositiveInt = 120
    max_repair_attempts: int = Field(default=2, ge=0, le=5)
    allow_reasoning_escalation: bool = True

    @model_validator(mode="after")
    def provider_configuration_required_when_enabled(self) -> AIRepairConfig:
        if not self.enabled:
            return self
        missing = [
            name
            for name, value in (
                ("provider", self.provider),
                ("model", self.model),
                ("api_key", self.api_key),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"AI repair requires enabled provider settings: {', '.join(missing)}")
        return self


class PaperLiveRiskConfig(StrictConfigModel):
    """Deterministic paper/live risk controls with deliberately unfrozen numeric limits."""

    enabled: bool = False
    max_position_weight: float | None = Field(default=None, gt=0.0, le=1.0)
    max_gross_exposure: float | None = Field(default=None, gt=0.0)
    max_abs_net_exposure: float | None = Field(default=None, ge=0.0)
    max_leverage: float | None = Field(default=None, gt=0.0)
    max_order_nav_fraction: float | None = Field(default=None, gt=0.0, le=1.0)
    max_participation_rate: float | None = Field(default=None, gt=0.0, le=1.0)
    daily_loss_limit_fraction: float | None = Field(default=None, gt=0.0, le=1.0)
    drawdown_stop_fraction: float | None = Field(default=None, gt=0.0, le=1.0)
    max_data_age_seconds: PositiveInt | None = None
    model_inference_timeout_seconds: PositiveInt | None = None
    max_outstanding_orders: PositiveInt | None = None
    require_broker_reconciliation: bool = True
    duplicate_order_protection: bool = True
    session_checks: bool = True
    kill_switch_enabled: bool = True

    @model_validator(mode="after")
    def validate_risk_controls(self) -> PaperLiveRiskConfig:
        """Require explicit numeric limits before paper/live risk controls are enabled."""
        if not self.enabled:
            return self

        required_numeric = {
            "max_position_weight": self.max_position_weight,
            "max_gross_exposure": self.max_gross_exposure,
            "max_abs_net_exposure": self.max_abs_net_exposure,
            "max_leverage": self.max_leverage,
            "max_order_nav_fraction": self.max_order_nav_fraction,
            "max_participation_rate": self.max_participation_rate,
            "daily_loss_limit_fraction": self.daily_loss_limit_fraction,
            "drawdown_stop_fraction": self.drawdown_stop_fraction,
            "max_data_age_seconds": self.max_data_age_seconds,
            "model_inference_timeout_seconds": self.model_inference_timeout_seconds,
            "max_outstanding_orders": self.max_outstanding_orders,
        }
        missing = [name for name, value in required_numeric.items() if value is None]
        if missing:
            raise ValueError(
                "enabled paper/live risk config requires explicit limits: " + ", ".join(missing)
            )

        if not all(
            (
                self.require_broker_reconciliation,
                self.duplicate_order_protection,
                self.session_checks,
                self.kill_switch_enabled,
            )
        ):
            raise ValueError(
                "enabled paper/live risk config requires all deterministic safety gates"
            )

        assert self.max_position_weight is not None
        assert self.max_gross_exposure is not None
        assert self.max_abs_net_exposure is not None
        assert self.max_leverage is not None
        if self.max_position_weight > self.max_gross_exposure:
            raise ValueError("max_position_weight cannot exceed max_gross_exposure")
        if self.max_abs_net_exposure > self.max_gross_exposure:
            raise ValueError("max_abs_net_exposure cannot exceed max_gross_exposure")
        if self.max_gross_exposure > self.max_leverage:
            raise ValueError("max_gross_exposure cannot exceed max_leverage")
        return self


class AppConfig(StrictConfigModel):
    """Complete validated configuration consumed by project workflows."""

    schema_version: Literal[1] = 1
    storage: StorageConfig
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    training: TrainingConfig
    objective: ObjectiveConfig
    evaluation: EvaluationConfig
    campaign: CampaignConfig
    scheduler: SchedulerConfig
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    ai_repair: AIRepairConfig = Field(default_factory=AIRepairConfig)
    paper_live_risk: PaperLiveRiskConfig = Field(default_factory=PaperLiveRiskConfig)

    @model_validator(mode="after")
    def validate_cross_section_constraints(self) -> AppConfig:
        campaign_minutes = self.campaign.max_duration_hours * 60
        if self.scheduler.initial_drain_reserve_minutes >= campaign_minutes:
            raise ValueError("scheduler drain reserve must be shorter than campaign duration")
        configured_horizons = set(self.dataset.primary_horizons_minutes) | set(
            self.dataset.auxiliary_horizons_minutes
        )
        missing = set(self.objective.horizons_minutes) - configured_horizons
        if missing:
            raise ValueError(
                "objective horizons must be declared by dataset configuration; "
                f"missing={sorted(missing)}"
            )
        return self
