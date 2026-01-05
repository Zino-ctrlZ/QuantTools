"""Intelligent Order Selection and Option Chain Filtering.

This module implements the OrderPicker class responsible for selecting optimal option
orders from available chains based on strategy criteria, risk constraints, and market
conditions. It handles order schema creation, chain filtering, strategy building, and
retry logic when initial criteria cannot be met.

Core Class:
    OrderPicker: Main order selection engine with caching and retry logic

Key Features:
    - Strategy-aware option selection (spreads, condors, butterflies, etc.)
    - Moneyness-based filtering (ITM, ATM, OTM ranges)
    - DTE (days to expiration) targeting with tolerance bands
    - Price constraints (min/max total price limits)
    - Liquidity filtering (volume, open interest, bid-ask spreads)
    - Automatic retry with relaxed constraints on failure
    - Memoization for performance optimization
    - Call/Put moneyness adjustment logic

Order Selection Process:
    1. Schema Creation:
       - Build OrderSchema from request parameters
       - Set target DTE, moneyness range, price limits
       - Configure strategy type and structure direction

    2. Chain Retrieval:
       - Fetch option chain for date and underlying
       - Cache chains to avoid redundant API calls
       - Filter by expiration dates within tolerance

    3. Strike Filtering:
       - Apply moneyness bounds (min/max strike/spot ratios)
       - Special handling for calls vs puts
       - ITM/OTM width constraints

    4. Strategy Building:
       - Select strikes for strategy legs
       - Apply spread_ticks for multi-leg structures
       - Validate structure meets requirements

    5. Price Validation:
       - Check total structure price against limits
       - Consider bid-ask spreads
       - Verify liquidity thresholds

    6. Result Extraction:
       - Package successful order with all details
       - Include order metadata and trade_id
       - Return ResultsEnum status

Moneyness Adjustment for Calls:
    Puts: moneyness = strike / spot
        - OTM: < 1.0
        - ATM: ~1.0
        - ITM: > 1.0

    Calls: moneyness inverted for consistency
        - Original put range: [0.90, 1.10]
        - Call range: [2-1.10, 2-0.90] = [0.90, 1.10]
        - Maintains same OTM/ITM interpretation

Configuration Objects:
    ChainConfig:
        - Data source settings
        - Chain retrieval parameters
        - Filtering options

    OrderSchemaConfigs:
        - Default DTE, moneyness, prices
        - Strategy defaults
        - Structure direction defaults

    OrderPickerConfig:
        - Start/end dates for data range
        - Cache configuration
        - Retry settings

    OrderResolutionConfig:
        - Max tries
        - Tolerance expansions
        - Resolution priorities

Lookback Management:
    - Precomputed date lookback dictionaries
    - Efficient date arithmetic for DTE calculations
    - Configurable lookback ranges (30, 60, 90 days)
    - Global LOOKBACKS dict for performance

Caching Strategy:
    Memoization:
        - @dynamic_memoize decorator on _get_order()
        - Schema tuple used as cache key
        - Avoids redundant chain fetching
        - Cleared between backtest runs

    Chain Caching:
        - populate_cache_with_chain() for data
        - Persistent across order requests
        - Indexed by (ticker, date, spot)

Retry Logic Integration:
    - Uses order_resolve_loop from _orders module
    - Progressive constraint relaxation
    - Logged attempts for debugging
    - Configurable max_tries limit
    - Returns best available or failure

Order Schema Structure:
    Required Fields:
        - tick: Underlying ticker
        - target_dte: Target days to expiration
        - strategy: Strategy name (e.g., 'vertical_spread')
        - structure_direction: 'long' or 'short'
        - spread_ticks: Strike separation for spreads
        - dte_tolerance: Acceptable DTE deviation
        - min_moneyness: Lower bound for strike/spot
        - max_moneyness: Upper bound for strike/spot
        - min_total_price: Minimum structure price
        - option_type: 'C' or 'P'
        - max_total_price: Upper price limit

    Chain Config Fields (merged):
        - Liquidity filters
        - Data source settings
        - Additional constraints

Usage Examples:
    # Initialize order picker
    picker = OrderPicker(
        start_date='2024-01-01',
        end_date='2024-12-31'
    )

    # Create order request
    request = OrderRequest(
        symbol='AAPL',
        date='2024-03-15',
        spot=185.50,
        option_type='P',
        target_dte=45,
        max_close=3.50
    )

    # Get order
    order = picker.get_order(request)

    # Check result
    if not order_failed(order):
        print(f"Selected: {order['data']['trade_id']}")
        print(f"Price: ${order['data']['total_price']:.2f}")
    else:
        print(f"Failed: {order['result']}")

Performance Optimizations:
    - Schema memoization prevents redundant calculations
    - Chain caching reduces API calls
    - Precomputed lookbacks for date math
    - Efficient pandas filtering operations
    - Lazy loading of option data

Integration Points:
    - Called by RiskManager for new position orders
    - Used in position rolling for replacement orders
    - Feeds OrderRequest dataclasses
    - Returns Order type for execution

Error Handling:
    - Invalid option types caught and corrected
    - Missing chains logged and handled gracefully
    - Failed orders return descriptive ResultsEnum
    - Schema validation before processing

Notes:
    - Lookback setter triggers precomputation if needed
    - get_order_new() is the public interface
    - _get_order() is the memoized implementation
    - Schema converted to tuple for hashability in cache
    - print_url param useful for debugging data sources
"""

from datetime import datetime
import pandas as pd
from EventDriven.riskmanager._order_validator import OrderInputs
from ..utils import (
    LOOKBACKS,
    get_cache,
    populate_cache_with_chain,
    precompute_lookbacks,
)
from EventDriven.configs.core import ChainConfig, OrderSchemaConfigs, OrderPickerConfig, OrderResolutionConfig

from ..utils import (
    dynamic_memoize,
)
from trade.helpers.Logging import setup_logger
from trade.helpers.decorators import timeit
from EventDriven.riskmanager.picker import OrderSchema, build_strategy, extract_order
from EventDriven.dataclasses.orders import OrderRequest
from EventDriven.riskmanager._orders import order_resolve_loop, order_failed
from EventDriven.types import Order

logger = setup_logger("EventDriven.riskmanager.picker.order_picker")


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
        self._chain_config = ChainConfig()
        self._order_picker_config = OrderPickerConfig(start_date=start_date, end_date=end_date)
        self._order_schema_config = OrderSchemaConfigs()
        self._order_resolution_config = OrderResolutionConfig()

    def __repr__(self):
        return f"OrderPicker(start_date={self.start_date}, end_date={self.end_date})"

    @property
    def lookback(self):
        return self.__lookback

    def get_order_schema(self, ticker: str, option_type: str = "P", max_total_price: float = None) -> OrderSchema:
        """
        Get the current order schema based on the order schema configurations.
        """
        schema = OrderSchema(
            {
                "tick": ticker,
                "target_dte": self._order_schema_config.target_dte,
                "strategy": self._order_schema_config.strategy,
                "structure_direction": self._order_schema_config.structure_direction,
                "spread_ticks": self._order_schema_config.spread_ticks,
                "dte_tolerance": self._order_schema_config.dte_tolerance,
                "min_moneyness": self._order_schema_config.min_moneyness,
                "max_moneyness": self._order_schema_config.max_moneyness,
                "min_total_price": self._order_schema_config.min_total_price,
                "option_type": option_type,  # Default to Put options
                "max_total_price": max_total_price,
            }
        )
        schema.data.update(self._chain_config.__dict__)
        return schema

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
    ):
        schema = tuple(schema.data.items())
        chain_spot = spot
        return self._get_order(schema, date, spot, chain_spot, print_url=print_url)

    @dynamic_memoize
    def _get_order(
        self,
        schema: tuple,
        date: str | datetime,
        spot: float,
        chain_spot: float = None,
        print_url: bool = False,
    ) -> dict:
        """
        Get the order for the given schema, date, and spot price.
        """
        assert isinstance(schema, tuple), "Schema must be a tuple of items."
        schema = OrderSchema(dict(schema))
        if schema["option_type"].lower() == "c":  ## This ensures that both call and put OTM are < 1.0 and ITM are > 1.0
            logger.info(
                f"Call Option Detected, Pre-Adjustment Moneyness: {schema['min_moneyness']} - {schema['max_moneyness']}"
            )
            min_m, max_m = 2 - schema["min_moneyness"], 2 - schema["max_moneyness"]
            schema["min_moneyness"] = min(min_m, max_m)  ## For Calls, we want the min moneyness to be 2 - min_moneyness
            schema["max_moneyness"] = max(min_m, max_m)
            logger.info(
                f"Call Option Detected, Adjusting Moneyness: {schema['min_moneyness']} - {schema['max_moneyness']}"
            )
        elif (
            schema["option_type"].lower() == "p"
        ):  ## This ensures that both call and put OTM are < 1.0 and ITM are > 1.0
            logger.info(
                f"Put Option Detected, Pre-Adjustment Moneyness: {schema['min_moneyness']} - {schema['max_moneyness']}"
            )
        else:
            raise ValueError(f"Invalid option type: {schema['option_type']}. Must be 'c' or 'p'.")

        chain = populate_cache_with_chain(schema["tick"], date, chain_spot, print_url=print_url)

        cache = get_cache("spot")
        cache = {k: v for k, v in cache.items()}

        raw_order = build_strategy(chain, schema, chain_spot, cache)
        return extract_order(raw_order)

    @timeit
    def get_order(self, request: OrderRequest) -> Order:
        """
        Get the order based on the request.
        """
        schema = self.get_order_schema(
            ticker=request.symbol, option_type=request.option_type, max_total_price=request.max_close
        )

        inputs = self.construct_inputs(
            request=request, schema=schema, order_resolution_config=self._order_resolution_config
        )
        return _get_open_order_backtest(
            picker=self,
            request=request,
            inputs=inputs,
        )

    def construct_inputs(
        self, request: OrderRequest, schema: OrderSchema, order_resolution_config: OrderResolutionConfig = None
    ) -> OrderInputs:
        """
        Construct OrderInputs dataclass from OrderRequest and OrderSchema.
        """
        if order_resolution_config is None:
            order_resolution_config = self._order_resolution_config

        if request.max_close > request.tick_cash:
            logger.warning(
                f"Request max_close {request.max_close} is greater than tick_cash {request.tick_cash}. Adjusting max_close to tick_cash."
            )
            request.max_close = request.tick_cash

        inputs = OrderInputs(
            tick=request.symbol,
            date=request.date,
            option_type=request.option_type,
            signal_id=request.signal_id,
            spot=request.spot,
            option_strategy=schema["strategy"],
            structure_direction=schema["structure_direction"],
            spread_ticks=schema.data.get("spread_ticks", 0),
            dte_tolerance=schema.data.get("dte_tolerance", 0),
            min_moneyness=schema.data.get("min_moneyness", 0),
            max_moneyness=schema.data.get("max_moneyness", float("inf")),
            target_dte=schema.data.get("target_dte", 0),
            min_total_price=schema.data.get("min_total_price", 0),
            direction=request.direction,
            tick_cash=request.tick_cash,
            **order_resolution_config.__dict__,
        )
        return inputs


def _get_open_order_backtest(
    picker: OrderPicker,
    request: OrderRequest,
    inputs: OrderInputs,
) -> Order:
    """
    Helper function to get open order in backtest mode.
    OR at least with order picker.

    params:
    picker: OrderPicker: The OrderPicker instance to use for getting the order.
    request: OrderRequest: The order request containing necessary parameters.
    inputs: OrderInputs: The order inputs constructed from the request and schema.

    returns:
    Order: The resolved order object.
    """
    schema = picker.get_order_schema(
        ticker=request.symbol, option_type=request.option_type, max_total_price=request.max_close
    )
    schema_as_tuple = tuple(schema.data.items())
    order = picker._get_order(
        schema=schema_as_tuple, date=request.date, spot=request.spot, chain_spot=request.chain_spot, print_url=False
    )

    ## Resolve order if failed and resolution is enabled
    if picker._order_resolution_config.resolve_enabled:
        order = order_resolve_loop(
            order=order,
            schema=schema,
            date=inputs.date,
            spot=inputs.spot,
            max_close=inputs.tick_cash / 100,  ## Use tick cash to determine max close. Normalize to 100 contracts
            max_dte_tolerance=inputs.max_dte_tolerance,
            max_tries=inputs.max_tries,
            otm_moneyness_width=inputs.otm_moneyness_width,
            itm_moneyness_width=inputs.itm_moneyness_width,
            logger=logger,
            signalID=inputs.signal_id,
            schema_cache={},
            picker=picker,
        )

    ## Add necessary tags for identification
    order["signal_id"] = inputs.signal_id
    order["map_signal_id"] = inputs.signal_id
    if order_failed(order):
        logger.warning(f"Order failed to resolve for request: {request} with schema: {schema}")
        return Order.from_dict(order)
    order["data"]["quantity"] = 1
    order["date"] = pd.to_datetime(request.date).date()
    order = Order.from_dict(order)
    return order
