
from datetime import datetime
import pandas as pd

from EventDriven.riskmanager._order_validator import OrderInputs
from ..utils import (
    LOOKBACKS,
    get_cache,
    populate_cache_with_chain,
    precompute_lookbacks,
)
from EventDriven.configs.core import (
    ChainConfig, 
    OrderSchemaConfigs, 
    OrderPickerConfig,
    OrderResolutionConfig
)

from ..utils import (
    logger,
    dynamic_memoize,
)
from EventDriven.riskmanager.picker import OrderSchema, build_strategy, extract_order
from EventDriven.dataclasses.orders import OrderRequest
from EventDriven.riskmanager._orders import order_resolve_loop  
from EventDriven.types import Order




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
        if schema["option_type"] == "C":  ## This ensures that both call and put OTM are < 1.0 and ITM are > 1.0
            logger.info(
                f"Call Option Detected, Pre-Adjustment Moneyness: {schema['min_moneyness']} - {schema['max_moneyness']}"
            )
            min_m, max_m = 2 - schema["min_moneyness"], 2 - schema["max_moneyness"]
            schema["min_moneyness"] = min(min_m, max_m)  ## For Calls, we want the min moneyness to be 2 - min_moneyness
            schema["max_moneyness"] = max(min_m, max_m)
            logger.info(
                f"Call Option Detected, Adjusting Moneyness: {schema['min_moneyness']} - {schema['max_moneyness']}"
            )
        elif schema["option_type"] == "P":  ## This ensures that both call and put OTM are < 1.0 and ITM are > 1.0
            logger.info(
                f"Put Option Detected, Pre-Adjustment Moneyness: {schema['min_moneyness']} - {schema['max_moneyness']}"
            )

        chain = populate_cache_with_chain(schema["tick"], date, chain_spot, print_url=print_url)

        cache = get_cache("spot")
        cache = {k: v for k, v in cache.items()}

        raw_order = build_strategy(chain, schema, spot, cache)
        return extract_order(raw_order)

    def get_order(self, request: OrderRequest) -> Order:
        """
        Get the order based on the request.
        """
        schema = self.get_order_schema(ticker=request.symbol, 
                                       option_type=request.option_type, 
                                       max_total_price=request.max_close)
        
        inputs = self.construct_inputs(request=request, 
                                       schema=schema, 
                                       order_resolution_config=self._order_resolution_config)
        try:
            return _get_open_order_backtest(
                picker=self,
                request=request,
                inputs=inputs,
            )
        except Exception as e:
            logger.error(f"Error getting order: {e}")
            return {}

    def construct_inputs(self, 
                        request: OrderRequest, 
                        schema: OrderSchema, 
                        order_resolution_config: OrderResolutionConfig = None
                        ) -> OrderInputs:
        """
        Construct OrderInputs dataclass from OrderRequest and OrderSchema.
        """
        if order_resolution_config is None:
            order_resolution_config = self._order_resolution_config

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
            schema=schema_as_tuple, 
            date=request.date, 
            spot=request.spot, 
            chain_spot=request.chain_spot, 
            print_url=False
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
    order["data"]["quantity"] = 1
    order["date"] = pd.to_datetime(request.date).date()
    order = Order.from_dict(order)
    return order