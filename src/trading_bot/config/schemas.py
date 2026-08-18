"""Strongly validated configuration schemas for the trading system."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    field_validator,
    model_validator,
)


class StrictConfigModel(BaseModel):
    """Base for immutable configuration objects that reject unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


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
        """Require the location fields needed by each storage backend."""
        if self.backend == "local" and not self.root_path:
            raise ValueError("local storage requires root_path")
        if self.backend == "s3" and not self.bucket:
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
    parameters: dict[str, Any] = Field(default_factory=dict)

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


class EvaluationConfig(StrictConfigModel):
    """Frozen economic-evaluation assumptions used during a campaign."""

    annualization_days: PositiveInt = 252
    risk_free_rate_annual: float = 0.0
    fee_bps: float = Field(default=0.0, ge=0.0)
    slippage_bps: float = Field(default=0.0, ge=0.0)
    impact_bps: float = Field(default=0.0, ge=0.0)
    cost_stress_multipliers: tuple[PositiveFloat, ...] = (1.0, 1.25, 1.5, 2.0)
    latency_stress_seconds: tuple[float, ...] = (0.0, 0.25, 1.0, 5.0, 15.0, 30.0)
    minimum_positive_fold_fraction: float = Field(default=0.70, ge=0.0, le=1.0)

    @field_validator("latency_stress_seconds")
    @classmethod
    def latency_values_must_be_nonnegative(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if any(delay < 0 for delay in value):
            raise ValueError("latency stress values must be non-negative")
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
    """Sandboxed AI repair escalation settings."""

    enabled: bool = False
    provider: Literal["deepseek"] = "deepseek"
    model: str = "deepseek-v4-flash"
    api_base_url: str = "https://api.deepseek.com"
    api_key: SecretStr | None = None
    primary_timeout_seconds: PositiveInt = 45
    thinking_timeout_seconds: PositiveInt = 120
    max_repair_attempts: int = Field(default=2, ge=0, le=5)
    allow_reasoning_escalation: bool = True

    @model_validator(mode="after")
    def api_key_required_when_enabled(self) -> AIRepairConfig:
        if self.enabled and self.api_key is None:
            raise ValueError("api_key is required when AI repair is enabled")
        return self


class PaperLiveRiskConfig(StrictConfigModel):
    """Deterministic paper/live risk limits."""

    max_position_weight: float = Field(default=0.02, gt=0.0, le=1.0)
    max_gross_exposure: float = Field(default=1.0, gt=0.0)
    max_abs_net_exposure: float = Field(default=0.20, ge=0.0)
    max_order_nav_fraction: float = Field(default=0.01, gt=0.0, le=1.0)
    daily_loss_limit_fraction: float = Field(default=0.02, gt=0.0, le=1.0)
    max_data_age_seconds: PositiveInt = 90
    max_outstanding_orders: PositiveInt = 50
    kill_switch_enabled: bool = True

    @model_validator(mode="after")
    def validate_exposure_relationships(self) -> PaperLiveRiskConfig:
        if self.max_position_weight > self.max_gross_exposure:
            raise ValueError("max_position_weight cannot exceed max_gross_exposure")
        if self.max_abs_net_exposure > self.max_gross_exposure:
            raise ValueError("max_abs_net_exposure cannot exceed max_gross_exposure")
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
