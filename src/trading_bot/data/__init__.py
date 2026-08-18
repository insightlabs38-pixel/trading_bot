"""Data acquisition and causal preprocessing primitives."""

from trading_bot.data.acquisition import (
    AcquisitionError,
    AcquisitionRecord,
    AcquisitionRetryPolicy,
    AcquisitionRunner,
    PermanentAcquisitionError,
    RequestRateLimiter,
    TransientAcquisitionError,
    VendorAdapter,
    VendorPayload,
    VendorRequest,
)

__all__ = [
    "AcquisitionError",
    "AcquisitionRecord",
    "AcquisitionRetryPolicy",
    "AcquisitionRunner",
    "PermanentAcquisitionError",
    "RequestRateLimiter",
    "TransientAcquisitionError",
    "VendorAdapter",
    "VendorPayload",
    "VendorRequest",
]

from trading_bot.data.raw_validation import (
    AnomalyCode,
    RawBar,
    RawDataAnomaly,
    RawValidationReport,
    validate_raw_bars,
)

__all__ += [
    "AnomalyCode",
    "RawBar",
    "RawDataAnomaly",
    "RawValidationReport",
    "validate_raw_bars",
]

from trading_bot.data.security_master import (
    CorporateAction,
    CorporateActionType,
    SecurityMaster,
    SecurityRecord,
    SecurityType,
    SymbolPeriod,
)

__all__ += [
    "CorporateAction",
    "CorporateActionType",
    "SecurityMaster",
    "SecurityRecord",
    "SecurityType",
    "SymbolPeriod",
]

from trading_bot.data.canonicalization import (
    CanonicalBar,
    CanonicalizationError,
    canonicalize_bars,
    total_return_between,
)

__all__ += [
    "CanonicalBar",
    "CanonicalizationError",
    "canonicalize_bars",
    "total_return_between",
]

from trading_bot.data.resampling import (
    ResampledBar,
    ResamplingError,
    SessionSpec,
    resample_canonical_bars,
)

__all__ += [
    "ResampledBar",
    "ResamplingError",
    "SessionSpec",
    "resample_canonical_bars",
]

from trading_bot.data.universe import (
    LiquidityObservation,
    UniverseConstructionError,
    UniverseMember,
    UniversePolicy,
    UniverseSnapshot,
    build_universe_snapshot,
    build_universe_snapshots,
)

__all__ += [
    "LiquidityObservation",
    "UniverseConstructionError",
    "UniverseMember",
    "UniversePolicy",
    "UniverseSnapshot",
    "build_universe_snapshot",
    "build_universe_snapshots",
]

from trading_bot.data.features import (
    FeatureObservation,
    FeaturePipelineError,
    FeaturePolicy,
    FeatureRow,
    compute_features,
)

__all__ += [
    "FeatureObservation",
    "FeaturePipelineError",
    "FeaturePolicy",
    "FeatureRow",
    "compute_features",
]

from trading_bot.data.labels import (
    LabelGenerationError,
    LabelObservation,
    LabelPolicy,
    LabelRow,
    generate_labels,
)

__all__ += [
    "LabelGenerationError",
    "LabelObservation",
    "LabelPolicy",
    "LabelRow",
    "generate_labels",
]

from trading_bot.data.splits import (
    DateRange,
    FinalHoldoutAccess,
    FinalHoldoutAccessError,
    SplitManifest,
    SplitManifestError,
    WalkForwardFold,
)

__all__ += [
    "DateRange",
    "FinalHoldoutAccess",
    "FinalHoldoutAccessError",
    "SplitManifest",
    "SplitManifestError",
    "WalkForwardFold",
]
