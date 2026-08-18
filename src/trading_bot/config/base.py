"""Shared behavior for immutable project configuration models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class FrozenConfigModel(BaseModel):
    """Base class for strongly validated, immutable configuration sections."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    def manifest_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible, secret-redacted representation for run manifests."""
        return self.model_dump(mode="json")
