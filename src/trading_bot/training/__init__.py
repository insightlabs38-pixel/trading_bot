"""Common training contracts and utilities."""

from trading_bot.training.contracts import (
    InferenceTiming,
    ModelBatch,
    ModelContractError,
    ModelOutput,
    TradingModel,
    parameter_count,
    time_inference,
)
from trading_bot.training.trainer import (
    NonFiniteTrainingError,
    TrainerSettings,
    TrainingError,
    TrainingHeartbeat,
    TrainingResult,
    UnsupportedPrecisionError,
    configure_reproducibility,
    train_model,
)

__all__ = [
    "InferenceTiming",
    "ModelBatch",
    "ModelContractError",
    "ModelOutput",
    "TradingModel",
    "parameter_count",
    "time_inference",
    "NonFiniteTrainingError",
    "TrainerSettings",
    "TrainingError",
    "TrainingHeartbeat",
    "TrainingResult",
    "UnsupportedPrecisionError",
    "configure_reproducibility",
    "train_model",
]
