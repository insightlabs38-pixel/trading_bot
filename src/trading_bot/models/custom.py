"""CPU/reference custom model families and temporal operators for Phase 9.

The module deliberately keeps the custom math in plain PyTorch.  The multi-decay
operator is the numerical reference that any future Triton implementation must match.
No Triton or CUDA dependency is imported here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from trading_bot.training.contracts import ModelOutput, TradingModel, TrainingBatch

CustomArchitecture = Literal["market_mixer", "heterogeneous_moe"]
CustomScale = Literal["small", "medium", "large"]
TemporalBackend = Literal["auto", "reference", "triton"]

CORE_CUSTOM_ARCHITECTURES: tuple[CustomArchitecture, ...] = (
    "market_mixer",
    "heterogeneous_moe",
)


class TritonUnavailableError(RuntimeError):
    """Raised when the explicitly requested Triton temporal backend is unavailable."""


@dataclass(frozen=True, slots=True)
class MarketMixerAblations:
    """Independent switches for the major Market Mixer components."""

    short_branch: bool = True
    long_branch: bool = True
    gated_fusion: bool = True
    cross_sectional: bool = True
    market_context: bool = True

    def __post_init__(self) -> None:
        if not self.short_branch and not self.long_branch:
            raise ValueError("Market Mixer requires at least one temporal branch")


@dataclass(frozen=True, slots=True)
class MarketMixerAblationCase:
    """Stable identifier plus switches for one reference ablation."""

    name: str
    ablations: MarketMixerAblations

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Market Mixer ablation name must not be blank")


@dataclass(frozen=True, slots=True)
class CustomModelSpec:
    """One versionable CPU/reference custom architecture configuration."""

    architecture: CustomArchitecture
    scale: CustomScale
    input_features: int
    max_sequence_length: int
    model_features: int
    num_heads: int
    num_layers: int
    feedforward_features: int
    num_decays: int
    moe_top_k: int
    include_frequency_expert: bool = True
    market_mixer_ablations: MarketMixerAblations = MarketMixerAblations()

    def __post_init__(self) -> None:
        positive = (
            self.input_features,
            self.max_sequence_length,
            self.model_features,
            self.num_heads,
            self.num_layers,
            self.feedforward_features,
            self.num_decays,
            self.moe_top_k,
        )
        if any(value < 1 for value in positive):
            raise ValueError("custom model dimensions must be positive")
        if self.model_features % self.num_heads != 0:
            raise ValueError("custom model_features must be divisible by num_heads")


@dataclass(frozen=True, slots=True)
class CustomModelProfile:
    """Deterministic learned-state accounting for custom models."""

    parameter_count: int
    trainable_parameter_count: int
    active_parameter_upper_bound: int
    parameter_bytes: int
    buffer_bytes: int
    total_state_bytes: int

    def __post_init__(self) -> None:
        values = (
            self.parameter_count,
            self.trainable_parameter_count,
            self.active_parameter_upper_bound,
            self.parameter_bytes,
            self.buffer_bytes,
            self.total_state_bytes,
        )
        if any(value < 0 for value in values):
            raise ValueError("custom model profile values must be non-negative")
        if self.active_parameter_upper_bound > self.parameter_count:
            raise ValueError("active parameter upper bound cannot exceed total parameters")


@dataclass(frozen=True, slots=True)
class MoERouterDiagnostics:
    """Detached router utilization summary suitable for structured logging."""

    expert_names: tuple[str, ...]
    assignment_counts: tuple[int, ...]
    mean_weights: tuple[float, ...]
    mean_entropy: float
    top_k: int

    def __post_init__(self) -> None:
        if not self.expert_names:
            raise ValueError("MoE diagnostics require at least one expert")
        if len(self.assignment_counts) != len(self.expert_names):
            raise ValueError("MoE assignment counts must match expert names")
        if len(self.mean_weights) != len(self.expert_names):
            raise ValueError("MoE mean weights must match expert names")
        if any(count < 0 for count in self.assignment_counts):
            raise ValueError("MoE assignment counts must be non-negative")
        if any(not math.isfinite(weight) or weight < 0 for weight in self.mean_weights):
            raise ValueError("MoE mean weights must be finite and non-negative")
        if not math.isfinite(self.mean_entropy) or self.mean_entropy < 0:
            raise ValueError("MoE mean entropy must be finite and non-negative")
        if self.top_k < 1 or self.top_k > len(self.expert_names):
            raise ValueError("MoE top_k is outside the expert set")


@dataclass(frozen=True, slots=True)
class TemporalOperatorBenchmark:
    """CPU/reference timing and state-size result for the custom temporal operator."""

    iterations: int
    samples: int
    elapsed_seconds: float
    samples_per_second: float
    state_bytes: int
    backend: str

    def __post_init__(self) -> None:
        if self.iterations < 1 or self.samples < 1:
            raise ValueError("temporal benchmark counts must be positive")
        if self.elapsed_seconds <= 0 or self.samples_per_second <= 0:
            raise ValueError("temporal benchmark timing must be positive")
        if self.state_bytes < 0:
            raise ValueError("temporal benchmark state_bytes must be non-negative")
        if not self.backend.strip():
            raise ValueError("temporal benchmark backend must not be blank")


class _CustomHeads(nn.Module):
    """Common architecture-neutral prediction heads."""

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


def multi_decay_reference(
    sequence: Tensor,
    decay_logits: Tensor,
    mix_logits: Tensor,
) -> Tensor:
    r"""Apply the causal learnable multi-decay reference recurrence.

    For decay channel ``k`` and feature ``d``:

    ``a[k,d] = sigmoid(decay_logits[k,d])``

    ``s[k,t,d] = a[k,d] * s[k,t-1,d] + (1-a[k,d]) * x[t,d]``

    ``q[k,d] = softmax(mix_logits[:,d])[k]``

    ``y[t,d] = sum_k q[k,d] * s[k,t,d]``

    The recurrence starts from zero state and therefore cannot use future inputs.
    """
    if sequence.ndim != 3:
        raise ValueError("multi-decay sequence must have shape [batch, time, features]")
    if decay_logits.ndim != 2 or mix_logits.shape != decay_logits.shape:
        raise ValueError("multi-decay logits must share shape [decays, features]")
    if int(decay_logits.shape[0]) < 1:
        raise ValueError("multi-decay operator requires at least one decay")
    if int(decay_logits.shape[1]) != int(sequence.shape[-1]):
        raise ValueError("multi-decay logits must match sequence feature width")

    decays = torch.sigmoid(decay_logits)
    mixing = torch.softmax(mix_logits, dim=0)
    state = torch.zeros(
        (
            int(sequence.shape[0]),
            int(decay_logits.shape[0]),
            int(sequence.shape[-1]),
        ),
        dtype=sequence.dtype,
        device=sequence.device,
    )
    decay_view = decays.unsqueeze(0)
    update_view = (1.0 - decays).unsqueeze(0)
    mixing_view = mixing.unsqueeze(0)
    outputs: list[Tensor] = []
    for index in range(int(sequence.shape[1])):
        current = sequence[:, index].unsqueeze(1)
        state = decay_view * state + update_view * current
        outputs.append((mixing_view * state).sum(dim=1))
    return torch.stack(outputs, dim=1)


def multi_decay_dispatch(
    sequence: Tensor,
    decay_logits: Tensor,
    mix_logits: Tensor,
    *,
    backend: TemporalBackend = "auto",
) -> Tensor:
    """Dispatch the custom operator while failing safely around unvalidated Triton."""
    if backend not in {"auto", "reference", "triton"}:
        raise ValueError(f"unsupported multi-decay backend {backend!r}")
    if backend == "triton":
        raise TritonUnavailableError(
            "the Phase 9 Triton backend is intentionally unavailable until GPU validation"
        )
    return multi_decay_reference(sequence, decay_logits, mix_logits)


class MultiDecayTemporalOperator(nn.Module):
    """Learnable causal multi-timescale decay/mixer with a residual projection."""

    def __init__(
        self,
        features: int,
        *,
        num_decays: int = 4,
        backend: TemporalBackend = "auto",
    ) -> None:
        super().__init__()
        if features < 1 or num_decays < 1:
            raise ValueError("multi-decay features and num_decays must be positive")
        if backend not in {"auto", "reference", "triton"}:
            raise ValueError(f"unsupported multi-decay backend {backend!r}")
        self.features = features
        self.num_decays = num_decays
        self.backend = backend
        self.input_projection = nn.Linear(features, features)
        initial = torch.linspace(-2.0, 2.0, steps=num_decays).unsqueeze(1)
        self.decay_logits = nn.Parameter(initial.repeat(1, features))
        self.mix_logits = nn.Parameter(torch.zeros(num_decays, features))
        self.output_projection = nn.Linear(features, features)
        self.gate_projection = nn.Linear(features, features)
        self.norm = nn.LayerNorm(features)

    def forward(self, sequence: Tensor) -> Tensor:
        if sequence.ndim != 3 or int(sequence.shape[-1]) != self.features:
            raise ValueError(
                f"multi-decay module expects [batch, time, {self.features}] input"
            )
        projected = self.input_projection(sequence)
        mixed = multi_decay_dispatch(
            projected,
            self.decay_logits,
            self.mix_logits,
            backend=self.backend,
        )
        gated = torch.sigmoid(self.gate_projection(projected))
        update = gated * self.output_projection(F.gelu(mixed))
        return cast(Tensor, self.norm(projected + update))


def benchmark_temporal_operator(
    operator: MultiDecayTemporalOperator,
    sequence: Tensor,
    *,
    warmup: int = 1,
    iterations: int = 5,
) -> TemporalOperatorBenchmark:
    """Measure eager reference throughput without asserting GPU relevance."""
    if warmup < 0 or iterations < 1:
        raise ValueError("temporal benchmark warmup/iterations are invalid")
    device = next(operator.parameters()).device
    value = sequence.to(device)
    operator.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            operator(value)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = perf_counter()
        for _ in range(iterations):
            operator(value)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = perf_counter() - started
    samples = iterations * int(sequence.shape[0])
    state_bytes = _module_state_bytes(operator)
    return TemporalOperatorBenchmark(
        iterations=iterations,
        samples=samples,
        elapsed_seconds=elapsed,
        samples_per_second=samples / elapsed,
        state_bytes=state_bytes,
        backend="reference" if operator.backend == "auto" else operator.backend,
    )


class _CausalLocalExpert(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            features,
            features,
            kernel_size=3,
            groups=features,
        )
        self.pointwise = nn.Conv1d(features, features, kernel_size=1)
        self.norm = nn.LayerNorm(features)

    def forward(self, sequence: Tensor) -> Tensor:
        channels = sequence.transpose(1, 2)
        local = self.depthwise(F.pad(channels, (2, 0)))
        local = self.pointwise(F.gelu(local)).transpose(1, 2)
        hidden = sequence + local
        return cast(Tensor, self.norm(hidden[:, -1]))


class _LongMemoryExpert(nn.Module):
    def __init__(self, features: int, num_decays: int) -> None:
        super().__init__()
        self.operator = MultiDecayTemporalOperator(
            features,
            num_decays=num_decays,
            backend="reference",
        )

    def forward(self, sequence: Tensor) -> Tensor:
        return self.operator(sequence)[:, -1]


class _TemporalAttentionExpert(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        num_heads: int,
        num_layers: int,
        feedforward_features: int,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=features,
            nhead=num_heads,
            dim_feedforward=feedforward_features,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(features)

    def forward(self, sequence: Tensor) -> Tensor:
        length = int(sequence.shape[1])
        mask = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=sequence.device),
            diagonal=1,
        )
        encoded = self.encoder(sequence, mask=mask)
        return cast(Tensor, self.norm(encoded[:, -1]))


class _FrequencyExpert(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.projection = nn.Linear(features, features)
        self.norm = nn.LayerNorm(features)

    def forward(self, sequence: Tensor) -> Tensor:
        spectrum = torch.fft.rfft(sequence, dim=1, norm="ortho")
        magnitude = spectrum.abs().mean(dim=1)
        return cast(Tensor, self.norm(F.gelu(self.projection(magnitude))))


class MultiScaleMarketMixerReturnModel(TradingModel):
    """Multi-timescale temporal mixer with same-timestamp market interaction."""

    def __init__(
        self,
        input_features: int,
        model_features: int = 32,
        *,
        num_heads: int = 4,
        num_decays: int = 4,
        max_sequence_length: int = 64,
        ablations: MarketMixerAblations | None = None,
    ) -> None:
        super().__init__()
        resolved_ablations = ablations or MarketMixerAblations()
        _validate_custom_dimensions(
            input_features,
            model_features,
            num_heads,
            max_sequence_length,
        )
        if num_decays < 1:
            raise ValueError("Market Mixer num_decays must be positive")
        self.input_features = input_features
        self.model_features = model_features
        self.max_sequence_length = max_sequence_length
        self.ablations = resolved_ablations
        self.feature_encoder = nn.Linear(input_features, model_features)
        self.shared_norm = nn.LayerNorm(model_features)
        self.short_branch = (
            _CausalLocalExpert(model_features) if resolved_ablations.short_branch else None
        )
        self.long_branch = (
            _LongMemoryExpert(model_features, num_decays)
            if resolved_ablations.long_branch
            else None
        )
        both_branches = resolved_ablations.short_branch and resolved_ablations.long_branch
        self.fusion_gate = (
            nn.Linear(model_features * 2, 2)
            if both_branches and resolved_ablations.gated_fusion
            else None
        )
        if resolved_ablations.cross_sectional:
            cross_layer = nn.TransformerEncoderLayer(
                d_model=model_features,
                nhead=num_heads,
                dim_feedforward=model_features * 2,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.cross_sectional_encoder: nn.TransformerEncoder | None = (
                nn.TransformerEncoder(cross_layer, num_layers=1)
            )
        else:
            self.cross_sectional_encoder = None
        self.market_projection = (
            nn.Linear(model_features * 2, model_features)
            if resolved_ablations.market_context
            else None
        )
        self.final_norm = nn.LayerNorm(model_features)
        self.heads = _CustomHeads(model_features)

    def forward(self, batch: TrainingBatch) -> ModelOutput:
        sequence = _sequence_features(batch, expected_features=self.input_features)
        if int(sequence.shape[1]) > self.max_sequence_length:
            raise ValueError("Market Mixer input exceeds configured context length")
        shared = cast(Tensor, self.shared_norm(self.feature_encoder(sequence)))

        branches: list[Tensor] = []
        if self.short_branch is not None:
            branches.append(self.short_branch(shared))
        if self.long_branch is not None:
            branches.append(self.long_branch(shared))
        if not branches:
            raise AssertionError("validated Market Mixer produced no temporal branches")
        if len(branches) == 1:
            hidden = branches[0]
        elif self.fusion_gate is None:
            hidden = torch.stack(branches, dim=0).mean(dim=0)
        else:
            concatenated = torch.cat((branches[0], branches[1]), dim=-1)
            weights = torch.softmax(self.fusion_gate(concatenated), dim=-1)
            hidden = weights[:, :1] * branches[0] + weights[:, 1:] * branches[1]

        if self.cross_sectional_encoder is not None or self.market_projection is not None:
            _require_single_cross_section(batch)
        if self.cross_sectional_encoder is not None:
            interacted = self.cross_sectional_encoder(hidden.unsqueeze(0)).squeeze(0)
            hidden = hidden + interacted
        if self.market_projection is not None:
            market = hidden.mean(dim=0, keepdim=True).expand_as(hidden)
            context = F.gelu(self.market_projection(torch.cat((hidden, market), dim=-1)))
            hidden = hidden + context
        return self.heads.forward(cast(Tensor, self.final_norm(hidden)))


class HeterogeneousMoEReturnModel(TradingModel):
    """Sparse top-k router over genuinely different temporal expert operators."""

    def __init__(
        self,
        input_features: int,
        model_features: int = 32,
        *,
        num_heads: int = 4,
        num_layers: int = 1,
        feedforward_features: int = 64,
        num_decays: int = 4,
        top_k: int = 2,
        include_frequency_expert: bool = True,
        max_sequence_length: int = 64,
    ) -> None:
        super().__init__()
        _validate_custom_dimensions(
            input_features,
            model_features,
            num_heads,
            max_sequence_length,
        )
        if num_layers < 1 or feedforward_features < 1 or num_decays < 1:
            raise ValueError("MoE layers/feedforward/decays must be positive")
        self.input_features = input_features
        self.model_features = model_features
        self.max_sequence_length = max_sequence_length
        self.feature_encoder = nn.Linear(input_features, model_features)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_sequence_length, model_features)
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        experts: list[nn.Module] = [
            _CausalLocalExpert(model_features),
            _LongMemoryExpert(model_features, num_decays),
            _TemporalAttentionExpert(
                model_features,
                num_heads=num_heads,
                num_layers=num_layers,
                feedforward_features=feedforward_features,
            ),
        ]
        expert_names = ["local_tcn", "long_memory", "temporal_attention"]
        if include_frequency_expert:
            experts.append(_FrequencyExpert(model_features))
            expert_names.append("frequency")
        self.experts = nn.ModuleList(experts)
        self.expert_names = tuple(expert_names)
        if top_k < 1 or top_k > len(self.experts):
            raise ValueError("MoE top_k must be within the configured expert count")
        self.top_k = top_k
        self.router = nn.Sequential(
            nn.Linear(model_features * 2, model_features),
            nn.GELU(),
            nn.Linear(model_features, len(self.experts)),
        )
        self.final_norm = nn.LayerNorm(model_features)
        self.heads = _CustomHeads(model_features)
        self._last_router_diagnostics: MoERouterDiagnostics | None = None

    def forward(self, batch: TrainingBatch) -> ModelOutput:
        _require_single_cross_section(batch)
        sequence = _sequence_features(batch, expected_features=self.input_features)
        length = int(sequence.shape[1])
        if length > self.max_sequence_length:
            raise ValueError("heterogeneous MoE input exceeds configured context length")
        encoded = self.feature_encoder(sequence) + self.position_embedding[:, :length]
        sample_state = encoded[:, -1]
        market_state = sample_state.mean(dim=0, keepdim=True).expand_as(sample_state)
        logits = self.router(torch.cat((sample_state, market_state), dim=-1))
        dense_weights = torch.softmax(logits, dim=-1)
        top_values, top_indices = torch.topk(dense_weights, k=self.top_k, dim=-1)
        sparse_weights = torch.zeros_like(dense_weights)
        sparse_weights.scatter_(1, top_indices, top_values)
        sparse_weights = sparse_weights / sparse_weights.sum(dim=-1, keepdim=True)

        combined = torch.zeros_like(sample_state)
        assignment_counts: list[int] = []
        for expert_index, expert in enumerate(self.experts):
            assigned = (top_indices == expert_index).any(dim=1)
            assignment_counts.append(int(assigned.sum().item()))
            if not bool(assigned.any().item()):
                continue
            expert_output = cast(Tensor, expert(encoded[assigned]))
            contribution = torch.zeros_like(sample_state)
            contribution[assigned] = (
                expert_output * sparse_weights[assigned, expert_index].unsqueeze(-1)
            )
            combined = combined + contribution

        detached = sparse_weights.detach()
        entropy = -(
            detached
            * torch.log(detached.clamp_min(torch.finfo(detached.dtype).eps))
        ).sum(dim=-1)
        self._last_router_diagnostics = MoERouterDiagnostics(
            expert_names=self.expert_names,
            assignment_counts=tuple(assignment_counts),
            mean_weights=tuple(float(value) for value in detached.mean(dim=0).cpu().tolist()),
            mean_entropy=float(entropy.mean().cpu().item()),
            top_k=self.top_k,
        )
        hidden = cast(Tensor, self.final_norm(sample_state + combined))
        return self.heads.forward(hidden)

    def router_diagnostics(self) -> MoERouterDiagnostics:
        """Return diagnostics from the most recent forward pass."""
        if self._last_router_diagnostics is None:
            raise RuntimeError("router diagnostics are unavailable before the first forward pass")
        return self._last_router_diagnostics

    def active_parameter_upper_bound(self) -> int:
        """Conservative learned-parameter bound for one sparse routed sample."""
        shared_modules: tuple[nn.Module, ...] = (
            self.feature_encoder,
            self.router,
            self.final_norm,
            self.heads,
        )
        shared = sum(
            parameter.numel()
            for module in shared_modules
            for parameter in module.parameters()
        )
        shared += self.position_embedding.numel()
        expert_counts = sorted(
            (
                sum(parameter.numel() for parameter in expert.parameters())
                for expert in self.experts
            ),
            reverse=True,
        )
        return shared + sum(expert_counts[: self.top_k])


def market_mixer_ablation_suite() -> tuple[MarketMixerAblationCase, ...]:
    """Return the stable one-component-off reference ablation matrix."""
    return (
        MarketMixerAblationCase("full", MarketMixerAblations()),
        MarketMixerAblationCase(
            "no_short",
            MarketMixerAblations(short_branch=False),
        ),
        MarketMixerAblationCase(
            "no_long",
            MarketMixerAblations(long_branch=False),
        ),
        MarketMixerAblationCase(
            "no_gated_fusion",
            MarketMixerAblations(gated_fusion=False),
        ),
        MarketMixerAblationCase(
            "no_cross_sectional",
            MarketMixerAblations(cross_sectional=False),
        ),
        MarketMixerAblationCase(
            "no_market_context",
            MarketMixerAblations(market_context=False),
        ),
    )


def custom_model_spec(
    architecture: CustomArchitecture,
    scale: CustomScale,
    *,
    input_features: int,
    max_sequence_length: int,
    market_mixer_ablations: MarketMixerAblations | None = None,
    include_frequency_expert: bool = True,
) -> CustomModelSpec:
    """Return deterministic small/medium/large CPU/reference configurations."""
    if scale == "small":
        model_features, num_layers, num_heads, num_decays, top_k = 16, 1, 2, 3, 2
    elif scale == "medium":
        model_features, num_layers, num_heads, num_decays, top_k = 32, 2, 4, 4, 2
    elif scale == "large":
        model_features, num_layers, num_heads, num_decays, top_k = 64, 3, 4, 6, 2
    else:
        raise AssertionError(f"unhandled custom scale {scale!r}")
    return CustomModelSpec(
        architecture=architecture,
        scale=scale,
        input_features=input_features,
        max_sequence_length=max_sequence_length,
        model_features=model_features,
        num_heads=num_heads,
        num_layers=num_layers,
        feedforward_features=model_features * 2,
        num_decays=num_decays,
        moe_top_k=top_k,
        include_frequency_expert=include_frequency_expert,
        market_mixer_ablations=market_mixer_ablations or MarketMixerAblations(),
    )


def build_custom_model(spec: CustomModelSpec) -> TradingModel:
    """Construct one Phase 9 custom family from a versionable specification."""
    if spec.architecture == "market_mixer":
        return MultiScaleMarketMixerReturnModel(
            input_features=spec.input_features,
            model_features=spec.model_features,
            num_heads=spec.num_heads,
            num_decays=spec.num_decays,
            max_sequence_length=spec.max_sequence_length,
            ablations=spec.market_mixer_ablations,
        )
    if spec.architecture == "heterogeneous_moe":
        return HeterogeneousMoEReturnModel(
            input_features=spec.input_features,
            model_features=spec.model_features,
            num_heads=spec.num_heads,
            num_layers=spec.num_layers,
            feedforward_features=spec.feedforward_features,
            num_decays=spec.num_decays,
            top_k=spec.moe_top_k,
            include_frequency_expert=spec.include_frequency_expert,
            max_sequence_length=spec.max_sequence_length,
        )
    raise AssertionError(f"unhandled custom architecture {spec.architecture!r}")


def profile_custom_model(model: nn.Module) -> CustomModelProfile:
    """Count learned state and sparse-active upper bound without a GPU."""
    parameters = tuple(model.parameters())
    buffers = tuple(model.buffers())
    parameter_count = sum(parameter.numel() for parameter in parameters)
    trainable_count = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in buffers)
    active_upper_bound = (
        model.active_parameter_upper_bound()
        if isinstance(model, HeterogeneousMoEReturnModel)
        else parameter_count
    )
    return CustomModelProfile(
        parameter_count=parameter_count,
        trainable_parameter_count=trainable_count,
        active_parameter_upper_bound=active_upper_bound,
        parameter_bytes=parameter_bytes,
        buffer_bytes=buffer_bytes,
        total_state_bytes=parameter_bytes + buffer_bytes,
    )


def _module_state_bytes(module: nn.Module) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in (*tuple(module.parameters()), *tuple(module.buffers()))
    )


def _sequence_features(batch: TrainingBatch, *, expected_features: int) -> Tensor:
    features = batch.features.float()
    if features.ndim == 2:
        features = features.unsqueeze(1)
    if features.ndim != 3:
        raise ValueError("custom model features must have shape [batch, time, features]")
    if int(features.shape[-1]) != expected_features:
        raise ValueError(
            f"custom model expected {expected_features} features, got {features.shape[-1]}"
        )
    return features


def _require_single_cross_section(batch: TrainingBatch) -> None:
    if int(torch.unique(batch.timestamps_ns).numel()) != 1:
        raise ValueError("custom cross-sectional models require one decision timestamp per batch")


def _validate_custom_dimensions(
    input_features: int,
    model_features: int,
    num_heads: int,
    max_sequence_length: int,
) -> None:
    if input_features < 1 or model_features < 1 or max_sequence_length < 1:
        raise ValueError("custom model feature/context dimensions must be positive")
    if num_heads < 1 or model_features % num_heads != 0:
        raise ValueError("custom model_features must be divisible by num_heads")
