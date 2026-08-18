"""Regression coverage for non-secret vendor pagination cursors."""

from trading_bot.data.acquisition import VendorRequest


def test_plain_token_parameter_can_represent_pagination_cursor() -> None:
    request = VendorRequest(
        provider="fake",
        dataset="bars",
        parameters={"token": "page-2"},
    )
    assert request.parameters["token"] == "page-2"
