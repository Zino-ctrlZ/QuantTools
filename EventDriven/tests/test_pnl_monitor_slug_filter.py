"""Tests for optional PnL monitor SignalID prefix filtering."""

from __future__ import annotations

from datetime import date

from EventDriven.configs.core import PnlMonitorConfig, PnLMonitorConfigConfigurable
from EventDriven.dataclasses.orders import OrderRequest
from EventDriven.riskmanager.position.cogs.pnl_monitor import PnLMonitorCog

MOMENTUM_SIGNAL = "donchian_momentum::AAPL20260820LONG"
MR_SIGNAL = "long_us_eq_mean_reversion::AAPL20260820LONG"


def _order_request(signal_id: str) -> OrderRequest:
    """Build a minimal order request for tick-cash mutation tests.

    Args:
        signal_id: Signal identifier string.

    Returns:
        OrderRequest with unscaled tick cash and positive symbol PnL.
    """
    return OrderRequest(
        date=date(2026, 9, 21),
        symbol="AAPL",
        option_type="c",
        max_close=10.0,
        tick_cash=10.0,
        direction="LONG",
        signal_id=signal_id,
        spot=200.0,
        is_tick_cash_scaled=False,
        symbol_total_pnl=200.0,
    )


def test_property_config_allows_every_slug() -> None:
    """PnlMonitorConfig has no prefix field and must not filter."""
    cog = PnLMonitorCog(PnlMonitorConfig())
    assert cog._allows_signal_id(MOMENTUM_SIGNAL)
    assert cog._allows_signal_id(MR_SIGNAL)
    assert cog._allows_signal_id(None)


def test_configurable_unset_prefixes_allow_every_slug() -> None:
    """Unset Configurable prefixes keep pre-filter behavior."""
    cog = PnLMonitorCog(PnLMonitorConfigConfigurable())
    assert cog.config.signal_slug_prefixes is None
    assert cog._allows_signal_id(MOMENTUM_SIGNAL)
    assert cog._allows_signal_id(MR_SIGNAL)


def test_empty_prefixes_allow_every_slug() -> None:
    """Empty tuple is treated as no filter."""
    cog = PnLMonitorCog(PnLMonitorConfigConfigurable(signal_slug_prefixes=()))
    assert cog._allows_signal_id(MR_SIGNAL)


def test_prefix_allows_only_matching_slugs() -> None:
    """Configured prefixes restrict analyze and order-request paths."""
    cog = PnLMonitorCog(
        PnLMonitorConfigConfigurable(signal_slug_prefixes=("donchian_momentum",))
    )
    assert cog._allows_signal_id(MOMENTUM_SIGNAL)
    assert not cog._allows_signal_id(MR_SIGNAL)
    assert not cog._allows_signal_id(None)


def test_on_new_order_request_skips_non_matching_slug() -> None:
    """Non-matching slugs must not mutate tick_cash."""
    cog = PnLMonitorCog(
        PnLMonitorConfigConfigurable(
            signal_slug_prefixes=("donchian_momentum",),
            max_trade_dollar_size=500,
            profit_lock_in_lvl=2,
        )
    )
    request = _order_request(MR_SIGNAL)
    cog.on_new_order_request(request)
    assert request.tick_cash == 10.0
    assert request.is_tick_cash_scaled is False
