"""Small architecturally distinct models used to validate the common trainer."""

from __future__ import annotations

import torch
from torch import nn

from trading_bot.training.contracts import ModelOutput, TradingModel, TrainingBatch


class LinearReturnModel(TradingModel):
    """Single affine return head."""

    def __init__(self, input_features: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_features, 1)

    def forward(self, batch: TrainingBatch) -> ModelOutput:
        return ModelOutput(expected_return=self.projection(batch.features).squeeze(-1))


class MLPReturnModel(TradingModel):
    """Nonlinear feed-forward baseline with dropout to exercise RNG checkpointing."""

    def __init__(self, input_features: int, hidden_features: int = 16) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, batch: TrainingBatch) -> ModelOutput:
        return ModelOutput(expected_return=self.network(batch.features).squeeze(-1))


class ResidualGatedReturnModel(TradingModel):
    """Residual gated mixer distinct from both affine and ordinary MLP baselines."""

    def __init__(self, input_features: int, hidden_features: int = 16) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_features, hidden_features)
        self.value_projection = nn.Linear(hidden_features, hidden_features)
        self.gate_projection = nn.Linear(hidden_features, hidden_features)
        self.output_projection = nn.Linear(hidden_features, 1)

    def forward(self, batch: TrainingBatch) -> ModelOutput:
        hidden = torch.tanh(self.input_projection(batch.features))
        mixed = hidden + torch.tanh(self.value_projection(hidden)) * torch.sigmoid(
            self.gate_projection(hidden)
        )
        return ModelOutput(expected_return=self.output_projection(mixed).squeeze(-1))
