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
