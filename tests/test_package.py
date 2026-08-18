"""Smoke tests for the base Python package."""

import trading_bot


def test_package_version() -> None:
    """The package exposes the project version."""
    assert trading_bot.__version__ == "0.1.0"
