"""CPU-reference advanced model families for Phase 8 screening.

The implementations in this module are deliberately dependency-light PyTorch references.
They preserve the common :class:`TrainingBatch` / :class:`ModelOutput` boundary so the
same trainer, checkpoints, prediction artifacts, and evaluator can exercise every
family on standard CPU CI. Hardware-specialized kernels and external pretrained
checkpoints remain separate acceptance concerns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from trading_bot.training.contracts import ModelOutput, TradingModel, TrainingBatch

AdvancedArchitecture = Literal[
    "patchtst",
    "itransformer",
    "mamba_reference",
    "vsn_lstm",
    "temporal_cross_sectional_transformer",
    "temporal_graph",
]
AdvancedScale = Literal["small", "medium", "large"]

CORE_ADVANCED_ARCHITECTURES: tuple[AdvancedArchitecture, ...] = (
    "patchtst",
    "itransformer",
    "mamba_reference",
    "vsn_lstm",
    "temporal_cross_sectional_transformer",
    "temporal_graph",
)


@dataclass(frozen=True, slots=True)
class AdvancedModelSpec:
    """One versionable CPU-reference architecture/scale configuration."""

    architecture: AdvancedArchitecture
    scale: AdvancedScale
    input_features: int
    max_sequence_length: int
    model_features: int
    num_layers: int
    num_heads: int
    feedforward_features: int
    patch_length: int = 2
    patch_stride: int = 2
    graph_top_k: int = 4

    def __post_init__(self) -> None:
        positive = (
            self.input_features,
            self.max_sequence_length,
            self.model_features,
            self.num_layers,
            self.num_heads,
            self.feedforward_features,
            self.patch_length,
            self.patch_stride,
            self.graph_top_k,
        )
        if any(value < 1 for value in positive):
            raise ValueError("advanced model dimensions must be positive")
        if self.model_features % self.num_heads != 0:
            raise ValueError("model_features must be divisible by num_heads")
        if self.patch_length > self.max_sequence_length:
            raise ValueError("patch_length cannot exceed max_sequence_length")


@dataclass(frozen=True, slots=True)
class AdvancedModelProfile:
    """Deterministic CPU-side learned-state memory accounting."""

    parameter_count: int
    trainable_parameter_count: int
    parameter_bytes: int
    buffer_bytes: int
    total_state_bytes: int

    def __post_init__(self) -> None:
        values = (
            self.parameter_count,
            self.trainable_parameter_count,
            self.parameter_bytes,
            self.buffer_bytes,
            self.total_state_bytes,
        )
        if any(value < 0 for value in values):
            raise ValueError("advanced model profile values must be non-negative")


@dataclass(frozen=True, slots=True)
class FoundationModelIdentity:
    """Immutable identity for a caller-supplied pretrained time-series checkpoint."""

    provider: str
    model_id: str
    revision: str
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        if any(not value.strip() for value in (self.provider, self.model_id, self.revision)):
            raise ValueError("foundation model identity fields must not be blank")
        checksum = self.checkpoint_sha256.lower()
        if len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum):
            raise ValueError("foundation checkpoint_sha256 must be a 64-character hex digest")


class FoundationBackbone(nn.Module):
    """Typed offline boundary for an already-loaded time-series foundation encoder."""

    output_features: int

    def forward(self, sequence: Tensor) -> Tensor:
        """Return either ``[batch, hidden]`` or ``[batch, time, hidden]`` embeddings."""
        raise NotImplementedError


class FrozenFoundationAdapter(TradingModel):
    """Adapt a verified caller-supplied frozen foundation backbone to common heads.

    This class intentionally performs no network access and does not select/download a
    checkpoint. Selection, licensing, artifact acquisition, and real checkpoint
    acceptance remain external to the CPU reference gate.
    """

    def __init__(
        self,
        backbone: FoundationBackbone,
        identity: FoundationModelIdentity,
        *,
        input_features: int,
        model_features: int = 32,
    ) -> None:
        super().__init__()
        _validate_widths(input_features, model_features)
        if backbone.output_features < 1:
            raise ValueError("foundation backbone output_features must be positive")
        self.backbone = backbone
        self.identity = identity
        self.input_features = input_features
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.projection = nn.Linear(backbone.output_features, model_features)
        self.heads = _AdvancedHeads(model_features)

    def forward(self, batch: TrainingBatch) -> ModelOutput:
        sequence = _sequence_features(batch, expected_features=self.input_features)
        encoded = self.backbone.forward(sequence)
        if encoded.ndim == 3:
            encoded = encoded[:, -1]
        if encoded.ndim != 2 or int(encoded.shape[0]) != batch.batch_size:
            raise ValueError("foundation backbone returned an invalid embedding shape")
        if int(encoded.shape[1]) != self.backbone.output_features:
            raise ValueError("foundation backbone output width does not match its contract")
        return self.heads.forward(self.projection(encoded))


class _AdvancedHeads(nn.Module):
    """Consistent prediction heads shared by every Phase 8 trainable family."""

    def __init__(self, hidden_features: int) -> None:
        super().__init__()
        self.return_head = nn.Linear(hidden_features, 1)
        self.direction_head = nn.Linear(hidden_features, 1)
        self.volatility_head = nn.Linear(hidden_features, 1)
        self.uncertainty_head = nn.Linear(hidden_features, 1)

    def forward(self, hidden: Tensor) -> ModelOutput:
        score = self.return_head(hidden).squeeze(-1)
        direction = torch.sigmoid(self.direction_head(hidden).squeeze(-1))
        volatility = F.softplus(self.volatility_head(hidden).squeeze(-1))
        uncertainty = F.softplus(self.uncertainty_head(hidden).squeeze(-1))
        return ModelOutput(
            expected_return=score,
            rank_score=score,
            direction_probability=direction,
            volatility=volatility,
            uncertainty=uncertainty,
        )


class PatchTSTReturnModel(TradingModel):
    """Channel-independent patch Transformer reference inspired by PatchTST."""

    def __init__(
        self,
        input_features: int,
        model_features: int = 32,
        *,
        num_heads: int = 4,
        num_layers: int = 1,
        feedforward_features: int = 64,
        patch_length: int = 4,
        patch_stride: int = 2,
        max_sequence_length: int = 64,
    ) -> None:
        super().__init__()
        _validate_transformer_dimensions(
            input_features,
            model_features,
            num_heads,
            num_layers,
            feedforward_features,
            max_sequence_length,
        )
        if patch_length < 1 or patch_stride < 1 or patch_length > max_sequence_length:
            raise ValueError("invalid PatchTST patch configuration")
        self.input_features = input_features
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.max_sequence_length = max_sequence_length
        self.patch_projection = nn.Linear(patch_length, model_features)
        max_patches = 2 + max_sequence_length // patch_stride
        self.position_embedding = nn.Parameter(torch.zeros(1, max_patches, model_features))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=model_features,
            nhead=num_heads,
            dim_feedforward=feedforward_features,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(model_features)
        self.heads = _AdvancedHeads(model_features)

    def forward(self, batch: TrainingBatch) -> ModelOutput:
        sequence = _sequence_features(batch, expected_features=self.input_features)
        length = int(sequence.shape[1])
        if length > self.max_sequence_length:
            raise ValueError("input sequence exceeds configured PatchTST context length")
        channels = sequence.transpose(1, 2)
        if length < self.patch_length:
            left_padding = self.patch_length - length
        else:
            remainder = (length - self.patch_length) % self.patch_stride
            left_padding = (self.patch_stride - remainder) % self.patch_stride
        channels = F.pad(channels, (left_padding, 0))
        patches = channels.unfold(2, self.patch_length, self.patch_stride)
        batch_size, feature_count, patch_count, _ = patches.shape
        tokens = self.patch_projection(patches)
        tokens = tokens.reshape(batch_size * feature_count, patch_count, -1)
        tokens = tokens + self.position_embedding[:, :patch_count]
        encoded = self.encoder(tokens)
        hidden = self.final_norm(encoded[:, -1])
        hidden = hidden.reshape(batch_size, feature_count, -1).mean(dim=1)
        return self.heads.forward(hidden)


class ITransformerReturnModel(TradingModel):
    """Inverted Transformer reference using variables as attention tokens."""

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
        _validate_transformer_dimensions(
            input_features,
            model_features,
            num_heads,
            num_layers,
            feedforward_features,
            max_sequence_length,
        )
        self.input_features = input_features
        self.max_sequence_length = max_sequence_length
        self.temporal_projection = nn.Linear(max_sequence_length, model_features)
        self.feature_embedding = nn.Parameter(torch.zeros(1, input_features, model_features))
        nn.init.normal_(self.feature_embedding, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=model_features,
            nhead=num_heads,
            dim_feedforward=feedforward_features,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(model_features)
        self.heads = _AdvancedHeads(model_features)

    def forward(self, batch: TrainingBatch) -> ModelOutput:
        sequence = _sequence_features(batch, expected_features=self.input_features)
        length = int(sequence.shape[1])
        if length > self.max_sequence_length:
            raise ValueError("input sequence exceeds configured iTransformer context length")
        variables = sequence.transpose(1, 2)
        variables = F.pad(variables, (self.max_sequence_length - length, 0))
        tokens = self.temporal_projection(variables) + self.feature_embedding
        encoded = self.encoder(tokens)
        hidden = self.final_norm(encoded).mean(dim=1)
        return self.heads.forward(hidden)


class _SelectiveStateSpaceBlock(nn.Module):
    """Readable selective state-space recurrence used by the Mamba reference family."""

    def __init__(self, model_features: int) -> None:
        super().__init__()
        self.in_projection = nn.Linear(model_features, model_features * 2)
        self.delta_projection = nn.Linear(model_features, model_features)
        self.input_projection = nn.Linear(model_features, model_features)
        self.output_gate_projection = nn.Linear(model_features, model_features)
        self.log_decay = nn.Parameter(torch.zeros(model_features))
        self.skip = nn.Parameter(torch.ones(model_features))
        self.out_projection = nn.Linear(model_features, model_features)
        self.norm = nn.LayerNorm(model_features)

    def forward(self, sequence: Tensor) -> Tensor:
        content, gate = self.in_projection(sequence).chunk(2, dim=-1)
        state = torch.zeros_like(content[:, 0])
        outputs: list[Tensor] = []
        decay_rate = torch.exp(self.log_decay)
        for index in range(int(sequence.shape[1])):
            current = content[:, index]
            delta = F.softplus(self.delta_projection(current)) + 1e-4
            decay = torch.exp(-delta * decay_rate)
            update = torch.tanh(self.input_projection(current)) * current
            state = decay * state + (1.0 - decay) * update
            output_gate = torch.sigmoid(self.output_gate_projection(current))
            value = output_gate * state + self.skip * current
            value = value * torch.sigmoid(gate[:, index])
            outputs.append(value)
        encoded = self.out_projection(torch.stack(outputs, dim=1))
        return self.norm(sequence + encoded)


class MambaReferenceReturnModel(TradingModel):
    """Pure-PyTorch selective state-space reference for the Mamba/Mamba-2 family.

    This is a correctness/screening reference, not a claim of fused Mamba-2 kernel
    equivalence or H200 performance.
    """

    def __init__(
        self,
        input_features: int,
        model_features: int = 32,
        *,
        num_layers: int = 1,
        max_sequence_length: int = 64,
    ) -> None:
        super().__init__()
        _validate_widths(input_features, model_features)
        if num_layers < 1 or max_sequence_length < 1:
            raise ValueError("Mamba reference layers/context must be positive")
        self.input_features = input_features
        self.max_sequence_length = max_sequence_length
        self.input_projection = nn.Linear(input_features, model_features)
        self.blocks = nn.ModuleList(
            [_SelectiveStateSpaceBlock(model_features) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(model_features)
        self.heads = _AdvancedHeads(model_features)

    def forward(self, batch: TrainingBatch) -> ModelOutput:
        sequence = _sequence_features(batch, expected_features=self.input_features)
        if int(sequence.shape[1]) > self.max_sequence_length:
            raise ValueError("input sequence exceeds configured Mamba reference context length")
        hidden = self.input_projection(sequence)
        for block in self.blocks:
            hidden = block(hidden)
        return self.heads.forward(self.final_norm(hidden[:, -1]))


class VSNLSTMReturnModel(TradingModel):
    """Variable-selection network feeding a recurrent LSTM reference."""

    def __init__(
        self,
        input_features: int,
        model_features: int = 32,
        *,
        num_layers: int = 1,
        max_sequence_length: int = 64,
    ) -> None:
        super().__init__()
        _validate_widths(input_features, model_features)
        if num_layers < 1 or max_sequence_length < 1:
            raise ValueError("VSN/LSTM layers/context must be positive")
        self.input_features = input_features
        self.max_sequence_length = max_sequence_length
        self.value_projection = nn.Linear(1, model_features)
        self.selection_network = nn.Sequential(
            nn.Linear(1, model_features),
            nn.GELU(),
            nn.Linear(model_features, 1),
        )
        self.feature_bias = nn.Parameter(torch.zeros(input_features))
        self.lstm = nn.LSTM(
            model_features,
            model_features,
            num_layers=num_layers,
            batch_first=True,
        )
        self.final_norm = nn.LayerNorm(model_features)
        self.heads = _AdvancedHeads(model_features)

    def forward(self, batch: TrainingBatch) -> ModelOutput:
        sequence = _sequence_features(batch, expected_features=self.input_features)
        if int(sequence.shape[1]) > self.max_sequence_length:
            raise ValueError("input sequence exceeds configured VSN/LSTM context length")
        values = sequence.unsqueeze(-1)
        embeddings = self.value_projection(values)
        logits = self.selection_network(values).squeeze(-1) + self.feature_bias
        weights = torch.softmax(logits, dim=2)
        selected = (embeddings * weights.unsqueeze(-1)).sum(dim=2)
        _, (hidden, _) = self.lstm(selected)
        return self.heads.forward(self.final_norm(hidden[-1]))


class TemporalCrossSectionalTransformerReturnModel(TradingModel):
    """Causal temporal encoder followed by same-timestamp cross-sectional attention."""

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
        _validate_transformer_dimensions(
            input_features,
            model_features,
            num_heads,
            num_layers,
            feedforward_features,
            max_sequence_length,
        )
        self.input_features = input_features
        self.max_sequence_length = max_sequence_length
        self.input_projection = nn.Linear(input_features, model_features)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_sequence_length, model_features)
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=model_features,
            nhead=num_heads,
            dim_feedforward=feedforward_features,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        cross_sectional_layer = nn.TransformerEncoderLayer(
            d_model=model_features,
            nhead=num_heads,
            dim_feedforward=feedforward_features,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=num_layers)
        self.cross_sectional_encoder = nn.TransformerEncoder(
            cross_sectional_layer,
            num_layers=num_layers,
        )
        self.final_norm = nn.LayerNorm(model_features)
        self.heads = _AdvancedHeads(model_features)

    def forward(self, batch: TrainingBatch) -> ModelOutput:
        _require_single_cross_section(batch)
        sequence = _sequence_features(batch, expected_features=self.input_features)
        length = int(sequence.shape[1])
        if length > self.max_sequence_length:
            raise ValueError("input exceeds temporal+cross-sectional context length")
        hidden = self.input_projection(sequence) + self.position_embedding[:, :length]
        mask = _causal_mask(length, hidden.device)
        temporal = self.temporal_encoder(hidden, mask=mask)[:, -1]
        cross_sectional = self.cross_sectional_encoder(temporal.unsqueeze(0)).squeeze(0)
        return self.heads.forward(self.final_norm(cross_sectional))


class TemporalGraphReturnModel(TradingModel):
    """Temporal encoder with a causal same-timestamp learned similarity graph."""

    def __init__(
        self,
        input_features: int,
        model_features: int = 32,
        *,
        num_layers: int = 1,
        graph_top_k: int = 4,
        max_sequence_length: int = 64,
    ) -> None:
        super().__init__()
        _validate_widths(input_features, model_features)
        if num_layers < 1 or graph_top_k < 1 or max_sequence_length < 1:
            raise ValueError("temporal graph layers/top-k/context must be positive")
        self.input_features = input_features
        self.model_features = model_features
        self.graph_top_k = graph_top_k
        self.max_sequence_length = max_sequence_length
        self.temporal_encoder = nn.GRU(
            input_features,
            model_features,
            num_layers=num_layers,
            batch_first=True,
        )
        self.message_projection = nn.Linear(model_features * 2, model_features)
        self.final_norm = nn.LayerNorm(model_features)
        self.heads = _AdvancedHeads(model_features)

    def forward(self, batch: TrainingBatch) -> ModelOutput:
        _require_single_cross_section(batch)
        sequence = _sequence_features(batch, expected_features=self.input_features)
        if int(sequence.shape[1]) > self.max_sequence_length:
            raise ValueError("input sequence exceeds configured temporal graph context length")
        _, hidden_stack = self.temporal_encoder(sequence)
        hidden = hidden_stack[-1]
        normalized = F.normalize(hidden, p=2.0, dim=-1, eps=1e-8)
        scores = normalized @ normalized.transpose(0, 1)
        scores = scores / math.sqrt(float(self.model_features))
        neighbor_count = min(self.graph_top_k, batch.batch_size)
        indices = torch.topk(scores, k=neighbor_count, dim=-1).indices
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, indices, True)
        attention = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
        message = attention @ hidden
        combined = self.message_projection(torch.cat((hidden, message), dim=-1))
        return self.heads.forward(self.final_norm(hidden + F.gelu(combined)))


def advanced_model_spec(
    architecture: AdvancedArchitecture,
    scale: AdvancedScale,
    *,
    input_features: int,
    max_sequence_length: int,
) -> AdvancedModelSpec:
    """Return deterministic small/medium/large CPU-reference configurations."""
    if scale == "small":
        model_features, num_layers, num_heads = 12, 1, 2
    elif scale == "medium":
        model_features, num_layers, num_heads = 24, 2, 4
    else:
        model_features, num_layers, num_heads = 48, 3, 4
    patch_length = min(4, max_sequence_length)
    patch_stride = min(2, patch_length)
    graph_top_k = 2 if scale == "small" else 4 if scale == "medium" else 8
    return AdvancedModelSpec(
        architecture=architecture,
        scale=scale,
        input_features=input_features,
        max_sequence_length=max_sequence_length,
        model_features=model_features,
        num_layers=num_layers,
        num_heads=num_heads,
        feedforward_features=model_features * 2,
        patch_length=patch_length,
        patch_stride=patch_stride,
        graph_top_k=graph_top_k,
    )


def build_advanced_model(spec: AdvancedModelSpec) -> TradingModel:
    """Construct one advanced family from its versionable reference specification."""
    common = {
        "input_features": spec.input_features,
        "model_features": spec.model_features,
        "max_sequence_length": spec.max_sequence_length,
    }
    if spec.architecture == "patchtst":
        return PatchTSTReturnModel(
            **common,
            num_heads=spec.num_heads,
            num_layers=spec.num_layers,
            feedforward_features=spec.feedforward_features,
            patch_length=spec.patch_length,
            patch_stride=spec.patch_stride,
        )
    if spec.architecture == "itransformer":
        return ITransformerReturnModel(
            **common,
            num_heads=spec.num_heads,
            num_layers=spec.num_layers,
            feedforward_features=spec.feedforward_features,
        )
    if spec.architecture == "mamba_reference":
        return MambaReferenceReturnModel(
            **common,
            num_layers=spec.num_layers,
        )
    if spec.architecture == "vsn_lstm":
        return VSNLSTMReturnModel(
            **common,
            num_layers=spec.num_layers,
        )
    if spec.architecture == "temporal_cross_sectional_transformer":
        return TemporalCrossSectionalTransformerReturnModel(
            **common,
            num_heads=spec.num_heads,
            num_layers=spec.num_layers,
            feedforward_features=spec.feedforward_features,
        )
    if spec.architecture == "temporal_graph":
        return TemporalGraphReturnModel(
            **common,
            num_layers=spec.num_layers,
            graph_top_k=spec.graph_top_k,
        )
    raise AssertionError(f"unhandled advanced architecture {spec.architecture!r}")


def profile_advanced_model(model: nn.Module) -> AdvancedModelProfile:
    """Count parameters/buffers and exact tensor storage bytes without a GPU."""
    parameters = tuple(model.parameters())
    buffers = tuple(model.buffers())
    parameter_count = sum(parameter.numel() for parameter in parameters)
    trainable_count = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in buffers)
    return AdvancedModelProfile(
        parameter_count=parameter_count,
        trainable_parameter_count=trainable_count,
        parameter_bytes=parameter_bytes,
        buffer_bytes=buffer_bytes,
        total_state_bytes=parameter_bytes + buffer_bytes,
    )


def _sequence_features(batch: TrainingBatch, *, expected_features: int) -> Tensor:
    features = batch.features.float()
    if features.ndim == 2:
        features = features.unsqueeze(1)
    if features.ndim != 3:
        raise ValueError("advanced model features must have shape [batch, time, features]")
    if int(features.shape[-1]) != expected_features:
        raise ValueError(
            f"advanced model expected {expected_features} features, got {features.shape[-1]}"
        )
    return features


def _require_single_cross_section(batch: TrainingBatch) -> None:
    if int(torch.unique(batch.timestamps_ns).numel()) != 1:
        raise ValueError("cross-sectional advanced models require one decision timestamp per batch")


def _causal_mask(length: int, device: torch.device) -> Tensor:
    return torch.triu(
        torch.ones((length, length), dtype=torch.bool, device=device),
        diagonal=1,
    )


def _validate_widths(input_features: int, model_features: int) -> None:
    if input_features < 1 or model_features < 1:
        raise ValueError("advanced model feature widths must be positive")


def _validate_transformer_dimensions(
    input_features: int,
    model_features: int,
    num_heads: int,
    num_layers: int,
    feedforward_features: int,
    max_sequence_length: int,
) -> None:
    _validate_widths(input_features, model_features)
    if num_heads < 1 or model_features % num_heads != 0:
        raise ValueError("Transformer model_features must be divisible by num_heads")
    if num_layers < 1 or feedforward_features < 1 or max_sequence_length < 1:
        raise ValueError("Transformer layer/feedforward/context sizes must be positive")
