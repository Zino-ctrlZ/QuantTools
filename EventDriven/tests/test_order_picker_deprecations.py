"""Tests for hard-deprecated legacy order-picker entry points.

These tests lock in the expected failure mode for legacy APIs while the old
source remains in place for short-term reference.
"""

import importlib
import sys

import pytest

from EventDriven.dataclasses.orders import OrderRequest
from EventDriven.riskmanager.picker.order_picker import OrderPicker


def _make_request() -> OrderRequest:
    return OrderRequest(
        date="2026-01-02",
        symbol="AAPL",
        option_type="p",
        max_close=1.0,
        tick_cash=100.0,
        direction="LONG",
        signal_id="sig-1",
        spot=100.0,
    )


def test_get_order_schema_raises_hard_error() -> None:
    picker = OrderPicker(start_date="2026-01-01", end_date="2026-12-31")
    with pytest.raises(AttributeError, match="get_order_schema"):
        picker.get_order_schema(ticker="AAPL", option_type="P", max_total_price=1.0)


def test_construct_inputs_raises_hard_error() -> None:
    picker = OrderPicker(start_date="2026-01-01", end_date="2026-12-31")
    request = _make_request()
    with pytest.raises(AttributeError, match="construct_inputs"):
        picker.construct_inputs(request=request, schema=None)


def test_importing_orders_module_raises_import_error() -> None:
    module_name = "EventDriven.riskmanager._orders"
    sys.modules.pop(module_name, None)

    with pytest.raises(ImportError):
        importlib.import_module(module_name)


def test_importing_order_validator_module_raises_import_error() -> None:
    module_name = "EventDriven.riskmanager._order_validator"
    sys.modules.pop(module_name, None)

    with pytest.raises(ImportError):
        importlib.import_module(module_name)
