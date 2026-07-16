"""Tests for Donchian cog entry-delay resolution from strategy info."""

from EventDriven.riskmanager.position.cogs.donchian_cog import _entry_delay_days_from_info


def test_momentum_prefixed_entry_delay_key() -> None:
    assert _entry_delay_days_from_info({"momentum_entry_delay_days": 1}) == 1


def test_standalone_entry_delay_key_fallback() -> None:
    assert _entry_delay_days_from_info({"entry_delay_days": 2}) == 2


def test_entry_delay_defaults_to_zero() -> None:
    assert _entry_delay_days_from_info({}) == 0


def test_momentum_entry_delay_takes_precedence() -> None:
    assert _entry_delay_days_from_info({"momentum_entry_delay_days": 1, "entry_delay_days": 5}) == 1
