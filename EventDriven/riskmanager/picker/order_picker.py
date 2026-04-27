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
import pandas as pd
from trade.helpers.helper import to_datetime
from EventDriven.riskmanager.utils import populate_cache_with_chain
from .iv_helper import _add_greeks_and_iv_to_chain
from .naked_option import naked_option_by_exp
from .vertical_spread import vertical_spread_pairer_by_exp
from . import filter_contracts

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
        
        ## L1 Resolution with optional hybrid fallback
        if order_failed(order):
            logger.warning(f"Order failed to resolve for request: {request}")

            ## If hybrid strategy is disabled, return the failed order immediately without attempting fallback. 
            ## This allows the system to recognize the failure and handle it according to its design.
            if not self._scoring_config.hybrid_strategy_enabled:
                logger.warning("Hybrid strategy is disabled. Returning failed order without fallback.")
                return Order.from_dict(order)
            
            ## If hybrid strategy is enabled, attempt a single fallback with the reverse strategy before giving up
            ## This provides a safety net for cases where the primary strategy fails, while still avoiding infinite fallback loops.
            ## Reverse strategy is determined based on the original strategy: if the original is "vertical", the fallback will be "naked", and vice versa.
            else:
                reverse_strategy = "naked" if self._scoring_config.strategy == "vertical" else "vertical"
                original_strategy = self._scoring_config.strategy
                self._scoring_config.strategy = reverse_strategy
                logger.warning(f"Primary strategy '{original_strategy}' failed. Attempting fallback with reverse strategy '{reverse_strategy}'.")
                order = self._get_order_with_scoring(request)
                self._scoring_config.strategy = original_strategy ## Reset strategy to original after fallback attempt
                order["signal_id"] = request.signal_id
                order["map_signal_id"] = request.signal_id
                
                ## If the fallback also fails, log a warning and return the original failed order to ensure the system can recognize the failure state. 
                if order_failed(order):
                    logger.warning(f"Fallback order also failed for request: {request}. Returning original failed order.")
                    return Order.from_dict(order)
                
                ## If the fallback succeeds, log a warning indicating that the fallback was successful and return the fallback order
                else:
                    logger.warning(f"Fallback order succeeded for request: {request}. Returning fallback order.")

        ## Final post order gen processing
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

    def display_chain(
        self,
        req: OrderRequest,
        filter_chain: bool = True,
        return_mask: bool = True,
    ) -> pd.DataFrame:
        """Return the paired option chain for inspection without scoring.

        Executes steps 1-4 of the scored chain pipeline: fetch chain,
        enrich with Greeks/IV, pair contracts by expiration, then optionally
        filter paired candidates using scoring-config thresholds. Useful for
        debugging contract selection or reviewing available spreads before committing
        to an order.

        Args:
            req: Order request supplying the symbol, date, option type, and chain spot.
            filter_chain: If True, applies spread-level moneyness, DTE, and pricing
                constraints from the scoring config after pairing. If False, returns
                all paired candidates.

        Returns:
            pd.DataFrame: Paired contracts (vertical spreads or naked options indexed
            by expiration), with Greeks and IV columns added. Empty DataFrame if no
            contracts survive filtering or pairing.

        Examples:
            >>> picker = OrderPicker(start_date="2025-01-01", end_date="2025-12-31")
            >>> req = OrderRequest(symbol="AAPL", date="2025-06-01", option_type="P",
            ...                   chain_spot=190.0, tick_cash=3.0)
            >>> df = picker.display_chain(req)
            >>> df = picker.display_chain(req, filter_chain=False)  # full chain
        """
        configs = self._scoring_config
        configs = deepcopy(configs)  # Avoid mutating shared config state, especially if we adjust mid price limits for display purposes
        configs.mid_upper_limit = req.tick_cash / 100 if req.is_tick_cash_scaled else req.tick_cash

        ## Step 1: Fetch chain
        chain = populate_cache_with_chain(tick=req.symbol, date=req.date, chain_spot=req.chain_spot)


        ## Step 2: Filter chain
        chain = filter_contracts(
            df=chain,
            option_type=req.option_type,
            spot=req.chain_spot,
            min_moneyness=configs.min_moneyness,
            max_moneyness=configs.max_moneyness,
            target_dte=configs.target_dte,
            dte_tol=configs.dte_tolerance,
        )

        if chain.empty:
            print("No contracts available after initial filtering. Returning empty DataFrame.")
            return chain

        ## Step 3: Add Greeks and IV
        chain = _add_greeks_and_iv_to_chain(filtered=chain, date=req.date, chain_spot=req.chain_spot)

        ## Step 4: Pair contracts by expiration
        is_call = req.option_type.lower() == "c"
        chain = chain.sort_values(by="strike", ascending=is_call).reset_index(drop=True)
        pairer = naked_option_by_exp if configs.strategy == "naked" else vertical_spread_pairer_by_exp
        paired = (
            chain.groupby("expiration")
            .apply(
                pairer,
                spread_tick=configs.spread_ticks,
                min_total_price=configs.mid_lower_limit,
                max_total_price=configs.mid_upper_limit,
                max_pct_width=configs.pct_spread_max if filter_chain else np.inf,
                min_oi=-25 if filter_chain else -np.inf,
                delta_lmt=np.inf,
                return_mask=return_mask,
            )
            .reset_index(drop=True)
        )

        return paired
