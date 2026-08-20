"""Common architecture-independent training framework."""

from trading_bot.training.checkpoint import (
    CheckpointCorruptionError,
    CheckpointError,
    CheckpointIdentity,
    CheckpointManager,
    CheckpointRestore,
    CheckpointResumeError,
)
from trading_bot.training.contracts import (
    InferenceBenchmark,
    ModelOutput,
    TradingModel,
    TrainingBatch,
    benchmark_inference,
    count_parameters,
)
from trading_bot.training.predictions import (
    PredictionArtifact,
    PredictionArtifactError,
    PredictionRecord,
    PredictionWriteResult,
    predict_records,
    write_prediction_artifact,
)
from trading_bot.training.trainer import (
    GpuMemoryTelemetry,
    NonFiniteTrainingError,
    Trainer,
    TrainerError,
    TrainerRuntimeOptions,
    TrainingHeartbeat,
    TrainingState,
    UnsupportedPrecisionError,
)

__all__ = [
    "CheckpointCorruptionError",
    "CheckpointError",
    "CheckpointIdentity",
    "CheckpointManager",
    "CheckpointRestore",
    "CheckpointResumeError",
    "GpuMemoryTelemetry",
    "InferenceBenchmark",
    "ModelOutput",
    "NonFiniteTrainingError",
    "PredictionArtifact",
    "PredictionArtifactError",
    "PredictionRecord",
    "PredictionWriteResult",
    "TradingModel",
    "Trainer",
    "TrainerError",
    "TrainerRuntimeOptions",
    "TrainingBatch",
    "TrainingHeartbeat",
    "TrainingState",
    "UnsupportedPrecisionError",
    "benchmark_inference",
    "count_parameters",
    "predict_records",
    "write_prediction_artifact",
]
