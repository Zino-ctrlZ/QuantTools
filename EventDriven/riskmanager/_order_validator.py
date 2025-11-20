from datetime import datetime
from typing import Any, Dict, Optional, TYPE_CHECKING, Union, Literal
import pandas as pd
import numbers
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from trade.helpers.Logging import setup_logger
from EventDriven.riskmanager.picker import STRATEGY_MAP
from EventDriven.riskmanager.market_data import get_timeseries_obj, OPTION_TIMESERIES_START_DATE
from EventDriven.riskmanager.picker import OrderSchema

logger = setup_logger('EventDriven.riskmanager._order_validator', stream_log_level='WARNING')

@dataclass(kw_only=True)
class _SlugConfig(ABC):
    initial_capital: float
    config: Dict[str, Any] = field(default_factory=dict)
    portfolio_settings: Dict[str, Any] = field(default_factory=dict)
    strategy_settings: Dict[str, Any] = field(default_factory=dict)
    rm_settings: Dict[str, Any] = field(default_factory=dict)
    sizer_settings: Dict[str, Any] = field(default_factory=dict)
    executor_settings: Dict[str, Any] = field(default_factory=dict)
    haircut_weights: Dict[str, float] = field(default_factory=dict)
    option_settings: Dict[str, Any] = field(default_factory=dict)
    cash_weights: Dict[str, float] = field(default_factory=dict)
    order_settings: Dict[str, Any] = field(default_factory=dict)
    cash_map: Dict[str, float] = field(default_factory=dict)

    @abstractmethod
    def load(self, source: Optional[str] = None) -> None:
        pass






INPUTS = {
        'tick': (str, "This should be a string representing the stock ticker.", ),
        'date': ([str, pd.Timestamp, datetime], "This should be a date in string format or a pandas Timestamp/datetime object."),
        'spot': (numbers.Number, "This should be a float representing chain_spot for the tick."),
        'signal_id': (str, "This should be a string representing the signal ID."),
        'max_close': (numbers.Number, "Max price for the order search engine."),
        'option_strategy': (str, f"This should be a string representing the option strategy. Available: {STRATEGY_MAP.keys()}"),
        'initial_cash': (numbers.Number, "This should be a float representing the initial cash for the strategy."),
        'option_type': (str, "This should be a string representing the option type, e.g., 'standard'.", ["C", "P"]),
        'structure_direction': (str, "This should be a string representing the structure direction.", ["long", "short"]),
        'spread_ticks': (numbers.Number, "This should be an integer representing the spread ticks."),
        'dte_tolerance': (numbers.Number, "This should be an integer representing the DTE tolerance."),
        'min_moneyness': (numbers.Number, "This should be a float representing the minimum moneyness."),
        'max_moneyness': (numbers.Number, "This should be a float representing the maximum moneyness."),
        'target_dte': (numbers.Number, "This should be an integer representing the target DTE."),
        'min_total_price': (numbers.Number, "This should be a float representing the minimum total price."),
        'direction': (str, "This should be a str of either LONG or SHORT"),
        'max_dte_tolerance': (numbers.Number, "This should be an integer representing the maximum DTE tolerance."),
        'max_tries': (numbers.Number, "This should be an integer representing the maximum number of tries."),
        'otm_moneyness_width': (numbers.Number, "This should be a float representing the OTM moneyness width max for ATM against OTM."),
        'itm_moneyness_width': (numbers.Number, "This should be a float representing the ITM moneyness width max for ATM against ITM."),
        'tick_cash': (numbers.Number, "This should be a float representing the cash allocated to this tick."),
    }


class OrderInputs:

    __slots__ = list(INPUTS.keys())
    if TYPE_CHECKING:
        tick: str
        date: Union[str, pd.Timestamp, datetime]
        spot: float
        signal_id: str
        max_close: float
        option_strategy: str
        initial_cash: float
        option_type: Literal["C", "P"]
        structure_direction: Literal["long", "short"]
        spread_ticks: numbers.Number
        dte_tolerance: numbers.Number
        min_moneyness: float
        max_moneyness: float
        target_dte: numbers.Number
        min_total_price: float
        direction: Literal["LONG", "SHORT"]
        max_dte_tolerance: numbers.Number
        max_tries: numbers.Number
        otm_moneyness_width: float
        itm_moneyness_width: float
        tick_cash: float

    def __init__(self, **kwargs):
        for k in kwargs:
            if k in self.__slots__:
                setattr(self, k, kwargs.get(k))
            else:
                logger.info(f"Unknown input key: {k}. This key will be ignored.")

OrderInputs.__doc__ = f"""
Order Container that holds all necessary variables in one place.
Args:
{chr(10).join([f"    {key} ({' | '.join([t.__name__ if isinstance(t, type) else str(t) for t in (types if isinstance(types, list) else [types])])}): {desc}" for key, (types, desc, *_) in INPUTS.items()])}
"""

def verify_order_selection_inputs(**kwargs):
    ## Key:
    #{input: (type, description, Optional[expected values])}

    for key, (expected_type, description, *expected_values) in INPUTS.items():
        if key not in kwargs:
            raise ValueError(f"Missing required input: '{key}'. Desc: {description}")
        if not isinstance(kwargs[key], tuple(expected_type) if isinstance(expected_type, list) else expected_type):
            raise TypeError(f"Input '{key}' must be of type {expected_type}, but got {type(kwargs[key])}. {description}")
        if expected_values and kwargs[key] not in expected_values[0]:
            raise ValueError(f"Input '{key}' must be one of {expected_values[0]}, but got '{kwargs[key]}'. {description}")
        

def build_inputs_with_config(config: _SlugConfig, 
                             max_close: float,
                             row: pd.Series,
                             tick_cash: float,
                             tick: str) -> tuple[OrderSchema, OrderInputs]:
    """
    Builds the inputs for the order selection engine based on the strategy config and trade row.
    Args:
        config (SlugConfig): The strategy configuration.
        max_close (float): The maximum price for the order selection engine.
        row (pd.Series): The trade row containing trade details. Expected keys: ['PT_BKTEST_SIG_ID', 'Size', 'EntryTime']
        tick (str): The stock ticker symbol.
    Returns:
        Tuple[OrderSchema, OrderInputs]: A tuple containing the OrderSchema and OrderInputs dataclass.
    """
    ## Necessary inputs
    assert isinstance(config, _SlugConfig), "config must be an instance of SlugConfig"
    assert all(k in row for k in ['PT_BKTEST_SIG_ID', 'Size', 'EntryTime']), "row must contain 'PT_BKTEST_SIG_ID', 'Size', and 'EntryTime' keys"

    ## Date is signal entry date + t+n days
    date = pd.to_datetime(row.EntryTime).strftime('%Y-%m-%d')
    signal_id = row.PT_BKTEST_SIG_ID

    
    ## The option strategy to deploy. Eg: 'vertical'
    option_strategy = config.order_settings.get('strategy', 'Unknown')
    if option_strategy == 'Unknown':
        raise ValueError("Unknown strategy. Not set in config?")
    
    ## The option type to deploy. Eg: 'C' or 'P'
    option_type = 'C' if row.Size > 0 else 'P'
    
    ## The structure direction. Eg: 'long' or 'short'. E.g., long vertical call spread
    structure_direction = config.order_settings.get('structure_direction', 'Unknown')
    if structure_direction == 'Unknown':
        raise ValueError("Unknown structure_direction. Not set in config?")
    
    ## Spread btwn strikes in ticks
    spread_ticks = config.order_settings.get('spread_ticks', 1)

    ## The Min & Max DTE tolerance
    dte_tolerance = config.order_settings.get('dte_tolerance', 90)

    ## Min & Max moneyness for the options to consider
    min_moneyness = config.order_settings.get('min_moneyness', 0.5)
    max_moneyness = config.order_settings.get('max_moneyness', 1.25)
    
    ## Target DTE for the options to consider
    target_dte = config.order_settings.get('target_dte', 'Unknown')
    if target_dte == 'Unknown':
        raise ValueError("Unknown target_dte. Not set in config?")
    
    ## Minimum total price for the order selection
    min_total_price = config.order_settings.get('min_total_price', max_close/2)

    ## Direction
    if option_type.upper() == 'C':
        direction = 'LONG'
    elif option_type.upper() == 'P':
        direction = 'SHORT'
    else:
        raise ValueError("Invalid option type. Must be 'C' or 'P'.")
    
    timeseries = get_timeseries_obj()
    timeseries.load_timeseries(tick, OPTION_TIMESERIES_START_DATE, datetime.now())
    
    ## Get spot price for the tick at the date. chain_spot is used for option pricing
    spot = timeseries.get_at_index(tick, date).chain_spot.close

    ## Min DTE Threshold
    max_dte_tolerance = config.rm_settings.get('max_dte_tolerance', 180)

    ## Amount of tries for the order selection engine if no orders are found
    max_tries = config.rm_settings.get('max_tries', 3)

    ## OTM & ITM moneyness width for multi-leg strategies
    otm_moneyness_width = config.rm_settings.get('otm_moneyness_width', 0.2)
    itm_moneyness_width = config.rm_settings.get('itm_moneyness_width', 0.2)
    initial_cash = config.initial_capital
    inputs = dict(
        tick=tick,
        date=date,
        spot=spot,
        signal_id=signal_id,
        max_close=max_close,
        option_strategy=option_strategy,
        initial_cash=initial_cash,
        option_type=option_type,
        structure_direction=structure_direction,
        spread_ticks=spread_ticks,
        dte_tolerance=dte_tolerance,
        min_moneyness=min_moneyness,
        max_moneyness=max_moneyness,
        target_dte=target_dte,
        min_total_price=min_total_price,
        direction=direction,
        max_dte_tolerance=max_dte_tolerance,
        max_tries=max_tries,
        otm_moneyness_width=otm_moneyness_width,
        itm_moneyness_width=itm_moneyness_width,
        tick_cash=tick_cash
    )
    verify_order_selection_inputs(**inputs)
    return OrderSchema({
            "strategy": option_strategy, "option_type": option_type, "tick": tick,
            "target_dte": target_dte, "dte_tolerance": dte_tolerance,
            "structure_direction": structure_direction, "max_total_price": max_close,
            "spread_ticks":spread_ticks, "min_moneyness": min_moneyness, "max_moneyness": max_moneyness,
            "min_total_price": min_total_price
        }), OrderInputs(**inputs)
    



