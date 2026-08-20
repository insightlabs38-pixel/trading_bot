"""Model families and common baseline implementations."""

from trading_bot.models.toy import (
    LinearReturnModel,
    MLPReturnModel,
    ResidualGatedReturnModel,
)

__all__ = [
    "LinearReturnModel",
    "MLPReturnModel",
    "ResidualGatedReturnModel",
]
