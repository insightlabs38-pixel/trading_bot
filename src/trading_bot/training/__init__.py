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
from trading_bot.training.checkpointing import (
    CheckpointCompatibilityError,
    CheckpointCorruptionError,
    CheckpointError,
    CheckpointIdentity,
    CheckpointRecord,
    RestoredCheckpoint,
    load_checkpoint,
    read_checkpoint_record,
    resolve_checkpoint_pointer,
    save_checkpoint,
)

__all__ = [
    "InferenceTiming", "ModelBatch", "ModelContractError", "ModelOutput", "TradingModel",
    "parameter_count", "time_inference", "NonFiniteTrainingError", "TrainerSettings",
    "TrainingError", "TrainingHeartbeat", "TrainingResult", "UnsupportedPrecisionError",
    "configure_reproducibility", "train_model", "CheckpointCompatibilityError",
    "CheckpointCorruptionError", "CheckpointError", "CheckpointIdentity", "CheckpointRecord",
    "RestoredCheckpoint", "load_checkpoint", "read_checkpoint_record",
    "resolve_checkpoint_pointer", "save_checkpoint",
]
