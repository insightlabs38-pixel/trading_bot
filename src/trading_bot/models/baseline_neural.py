"""CPU-verifiable neural baseline families for Phase 7."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from trading_bot.training.contracts import ModelOutput, TradingModel, TrainingBatch


class _BaselineHeads(nn.Module):
    """Shared return/rank and direction heads used by every neural baseline."""

    def __init__(self, hidden_features: int) -> None:
        super().__init__()
        self.return_head = nn.Linear(hidden_features, 1)
        self.direction_head = nn.Linear(hidden_features, 1)

    def forward(self, hidden: Tensor) -> ModelOutput:
        score = self.return_head(hidden).squeeze(-1)
        direction = torch.sigmoid(self.direction_head(hidden).squeeze(-1))
        return ModelOutput(
            expected_return=score,
            rank_score=score,
            direction_probability=direction,
        )


class BaselineMLPModel(TradingModel):
    """Per-token MLP with mean pooling over the causal input context."""

    def __init__(self, input_features: int, hidden_features: int = 32) -> None:
        super().__init__()
        _validate_widths(input_features, hidden_features)
        self.input_features = input_features
        self.encoder = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, hidden_features),
            nn.GELU(),
        )
        self.heads = _BaselineHeads(hidden_features)

    def forward(self, batch: TrainingBatch) -> ModelOutput:
        sequence = _sequence_features(batch, expected_features=self.input_features)
        hidden = self.encoder(sequence).mean(dim=1)
        return self.heads.forward(hidden)


class GRUReturnModel(TradingModel):
    """Single-layer GRU sequence baseline."""

    def __init__(self, input_features: int, hidden_features: int = 32) -> None:
        super().__init__()
        _validate_widths(input_features, hidden_features)
        self.input_features = input_features
        self.gru = nn.GRU(input_features, hidden_features, batch_first=True)
        self.heads = _BaselineHeads(hidden_features)

    def forward(self, batch: TrainingBatch) -> ModelOutput:
        sequence = _sequence_features(batch, expected_features=self.input_features)
        _, hidden = self.gru(sequence)
        return self.heads.forward(hidden[-1])


class LSTMReturnModel(TradingModel):
    """Single-layer LSTM sequence baseline paired with the GRU reference."""

    def __init__(self, input_features: int, hidden_features: int = 32) -> None:
        super().__init__()
        _validate_widths(input_features, hidden_features)
        self.input_features = input_features
        self.lstm = nn.LSTM(input_features, hidden_features, batch_first=True)
        self.heads = _BaselineHeads(hidden_features)

    def forward(self, batch: TrainingBatch) -> ModelOutput:
        sequence = _sequence_features(batch, expected_features=self.input_features)
        _, (hidden, _) = self.lstm(sequence)
        return self.heads.forward(hidden[-1])


class TCNReturnModel(TradingModel):
    """Small causal temporal-convolution baseline with left-only padding."""

    def __init__(
        self,
        input_features: int,
        hidden_features: int = 32,
        *,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        _validate_widths(input_features, hidden_features)
        if kernel_size < 2:
            raise ValueError("TCN kernel_size must be at least 2")
        self.input_features = input_features
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(input_features, hidden_features, kernel_size)
        self.conv2 = nn.Conv1d(hidden_features, hidden_features, kernel_size)
        self.heads = _BaselineHeads(hidden_features)

    def forward(self, batch: TrainingBatch) -> ModelOutput:
        sequence = _sequence_features(batch, expected_features=self.input_features)
        hidden = sequence.transpose(1, 2)
        hidden = F.gelu(self.conv1(F.pad(hidden, (self.kernel_size - 1, 0))))
        hidden = F.gelu(self.conv2(F.pad(hidden, (self.kernel_size - 1, 0))))
        return self.heads.forward(hidden[:, :, -1])


class CausalTransformerReturnModel(TradingModel):
    """Compact causal Transformer encoder baseline with learned positional embeddings."""

    def __init__(
        self,
        input_features: int,
        model_features: int = 32,
        *,
        num_heads: int = 4,
        num_layers: int = 1,
        feedforward_features: int = 64,
        max_sequence_length: int = 64,
    ) -> None:
        super().__init__()
        _validate_widths(input_features, model_features)
        if num_heads < 1 or model_features % num_heads != 0:
            raise ValueError("Transformer model_features must be divisible by num_heads")
        if num_layers < 1 or feedforward_features < 1 or max_sequence_length < 1:
            raise ValueError("Transformer layer/feedforward/context sizes must be positive")
        self.input_features = input_features
        self.max_sequence_length = max_sequence_length
        self.input_projection = nn.Linear(input_features, model_features)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_sequence_length, model_features))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=model_features,
            nhead=num_heads,
            dim_feedforward=feedforward_features,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(model_features)
        self.heads = _BaselineHeads(model_features)

    def forward(self, batch: TrainingBatch) -> ModelOutput:
        sequence = _sequence_features(batch, expected_features=self.input_features)
        length = int(sequence.shape[1])
        if length > self.max_sequence_length:
            raise ValueError("input sequence exceeds configured Transformer context length")
        hidden = self.input_projection(sequence) + self.position_embedding[:, :length]
        causal_mask = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=hidden.device),
            diagonal=1,
        )
        hidden = self.encoder(hidden, mask=causal_mask)
        return self.heads.forward(self.final_norm(hidden[:, -1]))


def _sequence_features(batch: TrainingBatch, *, expected_features: int) -> Tensor:
    features = batch.features.float()
    if features.ndim == 2:
        features = features.unsqueeze(1)
    if features.ndim != 3:
        raise ValueError("neural baseline features must have shape [batch, time, features]")
    if int(features.shape[-1]) != expected_features:
        raise ValueError(
            f"baseline expected {expected_features} input features, got {features.shape[-1]}"
        )
    return features


def _validate_widths(input_features: int, hidden_features: int) -> None:
    if input_features < 1 or hidden_features < 1:
        raise ValueError("baseline feature widths must be positive")
