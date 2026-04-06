"""Scoring-based option order picker.

This module contains the scoring-only `OrderPicker` runtime used by RiskManager.
Legacy schema-driven helpers remain only as hard-error compatibility entrypoints.

Core Class:
    OrderPicker: Selects and formats orders from `OrderRequest` using scoring.

Processing Flow:
    1. Normalize request-level cash constraints.
    2. Build scored candidate order via `order_builder`.
    3. Attach signal metadata and normalize output shape.
    4. Return typed `Order` with `ResultsEnum` status.

Risk/Assumptions:
    - `get_order(request=...)` is the only supported selection API.
    - Legacy methods (`get_order_schema`, `get_order_new`, `_get_order`,
      `construct_inputs`) raise hard deprecation errors.
    - Scoring limits are derived from `ScoringConfigs` with per-request caps.

Usage:
    >>> picker = OrderPicker(start_date="2024-01-01", end_date="2024-12-31")
    >>> order = picker.get_order(request)
"""

from copy import deepcopy
from datetime import datetime
from trade.datamanager.market_data import Optional
from ..utils import (
    LOOKBACKS,
    precompute_lookbacks,
)
from EventDriven.configs.core import OrderPickerConfig, ScoringConfigs
from EventDriven.types import OrderData, ResultsEnum

from ..utils import parse_position_id
from .builder import order_builder
from trade.helpers.Logging import setup_logger
from EventDriven.riskmanager.picker import OrderSchema, _order_formatting
from EventDriven.dataclasses.orders import OrderRequest
from EventDriven.types import Order
import numpy as np
from trade.helpers.helper import to_datetime

logger = setup_logger("EventDriven.riskmanager.picker.order_picker")


def order_failed(order: dict) -> bool:
    """Return True when the picker result is not successful."""
    return order.get("result") != ResultsEnum.SUCCESSFUL.value


class OrderPicker:
    """
    This class is solely responsible for picking orders based on predefined schemas and configurations.
    All it does is find the right order based on the schema provided. It does not execute trades or manage risk.
    """

    def __init__(self, start_date: str | datetime, end_date: str | datetime):
        """
        initializes the OrderPicker class
        """
        self.start_date = start_date
        self.end_date = end_date
        self.order_cache = {}
        self.__lookback = 30

        ## Setting up configs
        self._order_picker_config = OrderPickerConfig(start_date=start_date, end_date=end_date)
        self._scoring_config = ScoringConfigs()

        ## Others
        self.preset_orders = {}

    def __repr__(self):
        return f"OrderPicker(start_date={self.start_date}, end_date={self.end_date})"

    @property
    def lookback(self):
        return self.__lookback

    def _get_order_with_scoring(self, req: OrderRequest) -> OrderData:
        """
        This function is an alternative to the standard _get_order function that incorporates scoring into the order selection process. It builds the order using the builder function that includes scoring, and then validates the resulting order before returning it.
        """

        ## Create a copy per request.
        ## This is because we'll be changing mid_max from req
        configs = deepcopy(self._scoring_config)

        ## Limit the mid price to the tick cash value from the request. Don't entertain any orders that are priced above the cash available
        configs.mid_upper_limit = req.tick_cash / 100 if req.is_tick_cash_scaled else req.tick_cash
        order = order_builder(
            req=req,
            configs=configs,
        )
        return order

    def register_preset_order(self, signal_id: str, trade_id: str, date: str | datetime, close_price: float = np.nan):
        """
        Register a preset order to be used instead of generating a new one.
        This is useful for backtesting scenarios where specific orders need to be enforced.
        """
        self.preset_orders[signal_id] = {
            "trade_id": trade_id,
            "date": to_datetime(date, format="%Y-%m-%d").date(),
            "close_price": close_price,
        }

    def clear_preset_orders(self):
        """
        Clear all registered preset orders.
        """
        self.preset_orders = {}

    def get_preset_order(self, signal_id: str, date: str | datetime) -> dict:
        """
        Check if a preset order exists for the given signal_id and date
        If it exists, return the preset order details; otherwise, return an empty dictionary.
        It we will format the order as expected by the rest of the system.
        """
        preset_order = self.preset_orders.get(signal_id, None)
        if preset_order and preset_order["date"] == to_datetime(date, format="%Y-%m-%d").date():
            _, legs = parse_position_id(preset_order["trade_id"])
            data = _order_formatting(trade_id=preset_order["trade_id"], legs=legs, close=preset_order["close_price"])
            return {
                "result": ResultsEnum.SUCCESSFUL.value,
                "data": data,
                "map_signal_id": signal_id,
                "signal_id": signal_id,
            }
        return {}

    def get_order_schema(self, ticker: str, option_type: str = "P", max_total_price: float = None) -> OrderSchema:
        raise AttributeError(
            "OrderPicker.get_order_schema is deprecated and has been removed. "
            "Use OrderPicker.get_order(request=OrderRequest(...)) instead."
        )

    @lookback.setter
    def lookback(self, value):
        global LOOKBACKS
        initial_lookback_key = list(LOOKBACKS.keys())[0]
        if value not in LOOKBACKS[initial_lookback_key].keys():
            precompute_lookbacks("2000-01-01", "2030-12-31", _range=[value])
        self.__lookback = value

    def get_order_new(
        self,
        schema: OrderSchema,
        date: str | datetime,
        spot,
        chain_spot: float = None,
        print_url: bool = False,
        delta_lmt: Optional[float] = None,
    ):
        raise AttributeError(
            "OrderPicker.get_order_new is deprecated and has been removed. "
            "Use OrderPicker.get_order(request=OrderRequest(...)) instead."
        )

    # @dynamic_memoize
    def _get_order(
        self,
        schema: tuple,
        date: str | datetime,
        spot: float,
        chain_spot: float = None,
        print_url: bool = False,
        delta_lmt: Optional[float] = None,
    ) -> dict:
        raise AttributeError(
            "OrderPicker._get_order is deprecated and has been removed. "
            "Use OrderPicker.get_order(request=OrderRequest(...)) instead."
        )

    # @timeit
    def get_order(self, request: OrderRequest) -> Order:
        """
        Get the order based on the request.
        """
        preset_order = self.get_preset_order(signal_id=request.signal_id, date=request.date)
        if preset_order:
            if preset_order.get("data") is not None and "quantity" not in preset_order["data"]:
                preset_order["data"]["quantity"] = 1
            preset_order["date"] = to_datetime(request.date).date()
            return Order.from_dict(preset_order)

        tick_cash = request.tick_cash if not request.is_tick_cash_scaled else request.tick_cash / 100
        if request.max_close > tick_cash:
            logger.warning(
                f"Request max_close {request.max_close} is greater than tick_cash {tick_cash}. Adjusting max_close to tick_cash."
            )
            request.max_close = tick_cash

        order = self._get_order_with_scoring(request)

        ## Add necessary tags for identification
        order["signal_id"] = request.signal_id
        order["map_signal_id"] = request.signal_id
        if order_failed(order):
            logger.warning(f"Order failed to resolve for request: {request}")
            return Order.from_dict(order)
        order["data"]["quantity"] = 1
        order["date"] = to_datetime(request.date).date()
        order = Order.from_dict(order)
        return order

    def construct_inputs(self, request: OrderRequest, schema: OrderSchema) -> None:
        """Deprecated legacy entrypoint retained for API compatibility."""
        raise AttributeError(
            "OrderPicker.construct_inputs is deprecated and has been removed. "
            "Use OrderPicker.get_order(request=OrderRequest(...)) instead."
        )


def _get_open_order_backtest(
    picker: OrderPicker,
    request: OrderRequest,
) -> Order:
    raise AttributeError(
        "_get_open_order_backtest is deprecated and has been removed. "
        "Use OrderPicker.get_order(request=OrderRequest(...)) instead."
    )
