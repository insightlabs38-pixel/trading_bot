"""Model families and common baseline implementations."""

from trading_bot.models.baseline_checkpoint import (
    ClassicalCheckpointError,
    ClassicalCheckpointIdentity,
    restore_classical_checkpoint,
    save_classical_checkpoint,
)
from trading_bot.models.baseline_classical import (
    ClassicalBaseline,
    ClassicalBaselineError,
    ElasticNetBaseline,
    LightGBMBaseline,
    LogisticDirectionBaseline,
    RidgeBaseline,
    XGBoostBaseline,
)
from trading_bot.models.baseline_common import (
    BaselineComplexity,
    BaselineInferenceBenchmark,
    BaselineSplit,
    BaselineTargetNames,
)
from trading_bot.models.baseline_neural import (
    BaselineMLPModel,
    CausalTransformerReturnModel,
    GRUReturnModel,
    LSTMReturnModel,
    TCNReturnModel,
)
from trading_bot.models.baseline_objectives import build_baseline_loss
from trading_bot.models.toy import (
    LinearReturnModel,
    MLPReturnModel,
    ResidualGatedReturnModel,
)

__all__ = [
    "BaselineComplexity",
    "BaselineInferenceBenchmark",
    "BaselineMLPModel",
    "BaselineSplit",
    "BaselineTargetNames",
    "CausalTransformerReturnModel",
    "ClassicalBaseline",
    "ClassicalBaselineError",
    "ClassicalCheckpointError",
    "ClassicalCheckpointIdentity",
    "ElasticNetBaseline",
    "GRUReturnModel",
    "LSTMReturnModel",
    "LightGBMBaseline",
    "LinearReturnModel",
    "LogisticDirectionBaseline",
    "MLPReturnModel",
    "ResidualGatedReturnModel",
    "RidgeBaseline",
    "TCNReturnModel",
    "XGBoostBaseline",
    "build_baseline_loss",
    "restore_classical_checkpoint",
    "save_classical_checkpoint",
]
