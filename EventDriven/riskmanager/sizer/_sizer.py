"""Position Sizing Engines with Dynamic Greek Limit Calculations.

This module implements various position sizing strategies for options portfolios,
including fixed sizing, delta-neutral sizing, and volatility-adjusted sizing.
Each sizer calculates appropriate position quantities and Greek limits based on
available capital, risk parameters, and market conditions.

Core Classes:
    BaseSizer: Abstract base class for all sizing strategies
    DefaultSizer: Fixed delta limit sizing (static risk)
    ZscoreRVolSizer: Volatility-adjusted sizing (dynamic risk scaling)

Sizing Philosophy:
    Capital Allocation:
        - Sizes based on available cash per position/signal/ticker
        - Respects portfolio leverage constraints
        - Accounts for existing positions
        - Cash rules determine allocation method

    Risk Management:
        - Greek limits (especially delta) enforce directional risk
        - Limits scale with volatility (ZscoreRVolSizer)
        - Position size inversely related to option price
        - Quantity calculated to stay within delta limits

Cash Use Rules:
    Rule 1 - USE_CASH_EVERY_NEW_POSITION_ID (default):
        - Each unique position gets fresh capital allocation
        - Most granular cash management
        - Position-specific limits
        - Recommended for live trading

    Rule 2 - USE_CASH_EVERY_NEW_TICK:
        - Cash allocated per ticker symbol
        - Shared across all positions in same underlying
        - Ticker-level capital tracking

    Rule 3 - USE_CASH_EVERY_NEW_SIGNAL_ID:
        - Cash allocated per strategy signal
        - Multiple positions can share signal allocation
        - Signal-level capital management

BaseSizer Interface:
    Abstract Methods (must implement):
        - update_delta_limit(): Calculate delta limit for position
        - calculate_position_size(): Determine contract quantity
        - pre_analyze_task(): Daily setup before analysis
        - post_analyze_task(): Daily cleanup after analysis

    Common Functionality:
        - Cash tracking by rule
        - Sizing level (leverage) application
        - Position metadata logging
        - Delta limit history

DefaultSizer:
    Fixed Delta Limit Approach:
        - Static delta limit per position
        - No volatility adjustment
        - Simple and predictable
        - Good for stable markets

    Delta Limit Calculation:
        delta_limit = cash_available * sizing_lev / spot_price

    Position Size:
        quantity = delta_limit / (delta * option_price)

    Use Cases:
        - Conservative strategies
        - Low volatility environments
        - Backtests requiring consistency
        - Beginner-friendly sizing

ZscoreRVolSiger:
    Volatility-Adjusted Approach:
        - Delta limits scale with realized volatility
        - Higher vol → Lower limits (less risk)
        - Lower vol → Higher limits (more risk)
        - Adapts to market regime changes

    Z-Score Calculation:
        - Compares current vol to historical distribution
        - Positive z-score: Higher than average vol
        - Negative z-score: Lower than average vol
        - Lookback period configurable (default: 252 days)

    Scaling Formula:
        scalar = 1 / (1 + abs(zscore) * sensitivity)
        delta_limit = base_limit * scalar

    Benefits:
        - Automatic risk reduction in volatile markets
        - Increased exposure in calm markets
        - Data-driven risk management
        - Regime-adaptive sizing

Position Sizing Formula:
    General equation:
        quantity = delta_position_sizing(
            cash_available=current_cash,
            option_price_at_time=option_price,
            delta=option_delta,
            delta_limit=calculated_limit
        )

    Ensures:
        - Position delta ≤ delta_limit
        - Notional exposure ≤ available cash
        - Integer contract quantities
        - Minimum position size (typically 1 contract)

Cash Tracking:
    Starting Cash Registration:
        - register_signal_starting_cash(signal_id, cash)
        - register_position_id_starting_cash(position_id, cash)
        - Ticker cash from portfolio allocation

    Cash Retrieval (by rule):
        - get_cash(tick, signal_id, position_id)
        - Returns appropriate allocation
        - Based on active cash rule

Delta Limit Logging:
    - Tracks all delta limit calculations
    - Keys: position_id or date
    - Values: calculated limits with metadata
    - Used for audit and analysis
    - Accessible via delta_limit_log attribute

Re-update on Roll:
    - Controls whether limits recalculated on rolls
    - True (default): Fresh limits for rolled positions
    - False: Preserve original limits
    - Configurable via re_update_on_roll flag

Configuration:
    DefaultSizerConfigs:
        - sizing_lev: Leverage multiplier (default: 1.0)
        - delta_lmt_type: "default"
        - Additional parameters

    ZscoreSizerConfigs:
        - sizing_lev: Leverage multiplier
        - delta_lmt_type: "zscore"
        - sensitivity: Vol sensitivity parameter
        - lookback: Days for vol calculation
        - underlier_list: Symbols for vol tracking

Usage Examples:
    # Default sizer
    default_sizer = DefaultSizer(
        pm=portfolio_manager,
        rm=risk_manager,
        sizing_lev=1.0
    )

    # Calculate position size
    quantity = default_sizer.calculate_position_size(
        position_id='AAPL_20240315_175P_long',
        date='2024-02-15',
        option_price=3.50,
        delta=0.35,
        spot_price=185.00
    )

    # ZScore sizer with vol adjustment
    zscore_sizer = ZscoreRVolSizer(
        pm=portfolio_manager,
        rm=risk_manager,
        sizing_lev=1.5,
        sensitivity=0.5,
        lookback=252,
        underlier_list=['AAPL', 'MSFT', 'GOOGL']
    )

    # Set cash rule
    zscore_sizer.set_cash_rule(1)  # Per position

Integration Points:
    - Used by LimitsAndSizingCog for position sizing
    - Called by RiskManager on new position creation
    - Feeds into order quantity calculation
    - Logged for performance attribution

Performance Considerations:
    - Delta limit calculations cached
    - Volatility data precomputed
    - Minimal overhead per sizing call
    - Efficient cash lookups by rule

Override Function:
    override_calculate_position_size(**overrides):
        - Allows manual quantity calculation
        - Bypasses normal flow for testing
        - Requires: current_cash, option_price_at_time, delta, delta_limit
        - Returns: calculated quantity

Error Handling:
    - Invalid cash rule raises ValueError
    - Missing cash allocation uses default
    - Warnings on cash overwrite attempts
    - Zero/negative prices handled gracefully

Notes:
    - Sizing level acts as leverage multiplier
    - Cash rule always defaults to Rule 1 for live trading safety
    - Delta limits are per-leg for spreads (calculated separately)
    - Quantity always rounded down to integer
    - Minimum position size typically 1 contract
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
from trade.helpers.helper import setup_logger
from typing import TYPE_CHECKING
from EventDriven.riskmanager.utils import parse_signal_id
from trade.assets.helpers.utils import swap_ticker
import math
import pandas as pd
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import BDay
import os
from pathlib import Path
from ._utils import ZcoreScalar, default_delta_limit, zscore_rvol_delta_limit, delta_position_sizing, raise_none

if TYPE_CHECKING:
    from EventDriven.riskmanager.new_base import RiskManager
    from EventDriven.new_portfolio import OptionSignalPortfolio


logger = setup_logger("QuantTools.EventDriven.riskmanager.sizer")
BASE = Path(os.environ["WORK_DIR"]) / ".riskmanager_cache"


def override_calculate_position_size(**overrides) -> float:
    try:
        return delta_position_sizing(
            cash_available=overrides.pop("current_cash"),
            option_price_at_time=overrides.pop("option_price_at_time"),
            delta=overrides.pop("delta"),
            delta_limit=overrides.pop("delta_limit"),
        )
    except KeyError as e:
        raise ValueError(f"Missing required override parameter: {e}") from e


class BaseSizer(ABC):
    """
    Abstract base class for size calculation.

    This class provides a framework for calculating position sizes based on available cash and other parameters.

    Attributes:
    CASH_USE_RULES (dict): A dictionary defining the rules for cash base when calculating position sizes and greek limits.
    """

    __CASH_USE_RULES = {
        1: "USE_CASH_EVERY_NEW_POSITION_ID",
        2: "USE_CASH_EVERY_NEW_TICK",
        3: "USE_CASH_EVERY_NEW_SIGNAL_ID",
    }

    def __init__(self, pm: OptionSignalPortfolio = None, rm: RiskManager = None, sizing_lev=1.0, *args, **kwargs):
        """
        Initialize the BaseSizer with a PowerManager and ResourceManager.
        Args:
            pm (OptionSignalPortfolio): The PowerManager instance managing the portfolio.
            rm (RiskManager): The ResourceManager instance managing resources.
            sizing_lev (float): The sizing level for position sizing. Default is 1.0.
        """
        self.pm = pm
        self.rm = rm
        self._unavailable = (pm is None) and (rm is None)
        self.sizing_lev = sizing_lev
        self.signal_starting_cash = {}
        self.position_id_starting_cash = {}  ## This is helpful to track the starting cash for each position ID, especially when calculating limits
        self.ticker_starting_cash = pm.allocated_cash_map.copy() if not self._unavailable else {}
        self.cash_rule = self.__CASH_USE_RULES[1]
        self.re_update_on_roll = True  # Flag to control re-updating on roll
        self.delta_limit_log = {}

    @abstractmethod
    def update_delta_limit(self, *args, **kwargs):
        """
        Abstract method to calculate the delta limit for a given position ID and date.
        """
        pass

    @abstractmethod
    def calculate_position_size(self, *args, **kwargs):
        """
        Abstract method to calculate the position size for a given position ID and date.
        """
        pass

    @abstractmethod
    def pre_analyze_task(self, *args, **kwargs):
        """
        Abstract method for daily updates.
        """
        pass

    @abstractmethod
    def post_analyze_task(self, *args, **kwargs):
        """
        Abstract method for daily updates.
        """
        pass

    def get_cash(self, tick: str, signal_id: str, position_id: str) -> float:
        """
        Get the available cash for a given position ID and date based on the cash rule.
        This method determines the cash available for position sizing based on the specified cash rule.
        Args:
            tick (str): The ticker symbol for the asset.
            signal_id (str): The ID of the signal associated with the position.
            position_id (str): The ID of the position.
        """

        if self.cash_rule == self.__CASH_USE_RULES[1]:
            return self.position_id_starting_cash.get(position_id, self.pm.allocated_cash_map[tick])
        elif self.cash_rule == self.__CASH_USE_RULES[2]:
            return self.ticker_starting_cash[tick]
        elif self.cash_rule == self.__CASH_USE_RULES[3]:
            return self.signal_starting_cash.get(signal_id, self.pm.allocated_cash_map[tick])
        else:
            raise ValueError(f"Unknown cash rule: {self.cash_rule}")

    def set_cash_rule(self, rule: int) -> None:
        """
        Set the cash rule for position sizing.
        Args:
            rule (int): The cash rule to set. Must be one of the keys in CASH_USE_RULES.
        """
        logger.critical(
            "Cash rule will always be USE_CASH_EVERY_NEW_POSITION_ID because it is difficult to maintain live "
        )
        rule = 1
        if rule in self.__CASH_USE_RULES:
            self.cash_rule = self.__CASH_USE_RULES[rule]
        else:
            raise ValueError(f"Invalid cash rule: {rule}. Available rules: {self.__CASH_USE_RULES.keys()}")

    def register_signal_starting_cash(self, signal_id: str, cash: float) -> None:
        """
        Register the starting cash for a specific signal ID.
        Args:
            signal_id (str): The ID of the signal.
            cash (float): The starting cash amount for the signal.
        """
        if signal_id in self.signal_starting_cash:
            logger.warning(f"Signal ID {signal_id} already has starting cash registered. Overwriting previous value.")
            return
        self.signal_starting_cash[signal_id] = cash

    def register_position_id_starting_cash(self, position_id: str, cash: float) -> None:
        """
        Register the starting cash for a specific position ID.
        Args:
            position_id (str): The ID of the position.
            cash (float): The starting cash amount for the position.
        """
        if position_id in self.position_id_starting_cash:
            logger.warning(
                f"Position ID {position_id} already has starting cash registered. Overwriting previous value."
            )
            return
        self.position_id_starting_cash[position_id] = cash

    def log_daily_delta_limit(self, symbol: str, date: datetime, delta_limit: float) -> None:
        """
        Log the daily delta limit for a specific symbol and date.
        Args:
            symbol (str): The ticker symbol for the asset.
            date (datetime): The date for which the delta limit is logged.
            delta_limit (float): The delta limit value to log.
        """
        if symbol not in self.delta_limit_log:
            self.delta_limit_log[symbol] = {}
        self.delta_limit_log[symbol][date] = delta_limit

    @property
    def CASH_USE_RULES(self):
        """
        Get the available cash use rules.
        """
        return self.__CASH_USE_RULES

    @CASH_USE_RULES.setter
    def CASH_USE_RULES(self, value):
        """
        Set the available cash use rules.
        """
        raise AttributeError("CASH_USE_RULES is a read-only property.")

    def __repr__(self):
        """
        String representation of the BaseSizer class.
        """
        return f"{self.__class__.__name__}(sizing_lev={self.sizing_lev}, cash_rule={self.cash_rule})"


class DefaultSizer(BaseSizer):
    """
    Default implementation of the BaseSizer.
    Calculates position size based on delta limits and available cash.
    """

    def __init__(self, pm=None, rm=None, sizing_lev=1.0, *args, **kwargs):
        super().__init__(pm, rm, sizing_lev, *args, **kwargs)

    def get_daily_delta_limit(
        self, signal_id: str = None, position_id: str = None, date: str | datetime = None, **overrides
    ) -> float:
        """
        Returns the delta limit for a given signal ID and date.
        This method retrieves the delta limit for a specific signal ID and date,
        calculating it if it doesn't already exist in the risk manager's greek limits.

        Args:
            signal_id (str): The ID of the signal.
            position_id (str): The ID of the position.
            date (str|datetime): The date for which the delta limit is calculated.

        Returns:
            float: The delta limit for the specified signal ID and date.
        """

        if self._unavailable:
            try:
                raise_none(signal_id, "signal_id")
                raise_none(position_id, "position_id")
                current_cash = overrides.pop("current_cash")
                underlier_price_at_time = overrides.pop("underlier_price_at_time")
            except KeyError as e:
                raise ValueError(f"Missing required override parameter: {e}") from e
            self.register_position_id_starting_cash(signal_id, current_cash)
            self.register_signal_starting_cash(position_id, current_cash)
            delta = default_delta_limit(
                cash_available=current_cash, sizing_lev=self.sizing_lev, underlier_price_at_time=underlier_price_at_time
            )
            return delta

        else:
            logger.info(
                "DefaultSizer: Calculating Delta Limit for Signal ID: %s and Position ID: %s on Date: %s\n",
                signal_id,
                position_id,
                date,
            )
            id_details = parse_signal_id(signal_id)
            self.register_signal_starting_cash(
                signal_id, self.pm.allocated_cash_map[id_details["ticker"]]
            )  ## Register the starting cash for the signal
            self.register_position_id_starting_cash(
                position_id, self.pm.allocated_cash_map[id_details["ticker"]]
            )  ## Register the starting cash for the position ID

            logger.info("Updating Greek Limits for Signal ID: %s and Position ID: %s", signal_id, position_id)
            starting_cash = self.get_cash(
                id_details["ticker"], signal_id, position_id
            )  ## Get the cash available for the ticker based on the cash rule
            logger.info(
                f"Starting Cash for {id_details['ticker']} on {date}: {starting_cash} vs Current Cash: {self.pm.allocated_cash_map[id_details['ticker']]}"
            )
            delta_at_purchase = self.rm.position_data[position_id]["Delta"][
                date
            ]  ## This is the delta at the time of purchase
            s0_at_purchase = self.rm.position_data[
                position_id
            ][
                "s"
            ][
                date
            ]  ## This is the delta at the time of purchase## As always, we use the chain spot data to account for splits
            equivalent_delta_size = (
                (starting_cash * self.sizing_lev) / (s0_at_purchase * 100) if s0_at_purchase != 0 else 0
            )
            logger.info(
                f"Spot Price at Purchase: {s0_at_purchase} at time {date}  ## This is the delta at the time of purchase"
            )
            logger.info(f"Delta at Purchase: {delta_at_purchase}")
            logger.info(
                f"Equivalent Delta Size: {equivalent_delta_size}, with Cash Available: {starting_cash}, and Leverage: {self.sizing_lev}"
            )
            return equivalent_delta_size

    def update_delta_limit(
        self, signal_id: str = None, position_id: str = None, date: str | datetime = None, **overrides
    ) -> float:
        """
        Updates the limits associated with a signal
        ps: This should only be updated on first purchase of the signal
            Limits are saved in absolute values to account for both long and short positions

        Limit Calc: (allocated_cash * lev)/(S0 * 100)
        """
        if self._unavailable:
            raise_none(signal_id, "signal_id")
            raise_none(position_id, "position_id")
            return self.get_daily_delta_limit(**overrides, signal_id=signal_id, position_id=position_id, date=date)

        equivalent_delta_size = self.get_daily_delta_limit(
            signal_id, position_id, date
        )  ## Get the delta limit for the signal ID and date
        self.rm.greek_limits["delta"][signal_id] = (
            equivalent_delta_size  ## Update the delta limit in the risk manager's greek limits
        )
        return equivalent_delta_size

    def calculate_position_size(
        self,
        signal_id: str = None,
        position_id: str = None,
        opt_price: str = None,
        date: str | datetime = None,
        side: int = 1,
        **overrides,
    ) -> float:
        """
        Returns the quantity of the position that can be bought based on the sizing type
        Args:
            signal_id (str): The ID of the signal.
            position_id (str): The ID of the position.
            opt_price (float): The price of the option.
            date (str|datetime): The date for which the position size is calculated.
            side (int): The side of the position, 1 for long and -1 for short.
        Raises:
            ValueError: If side is not 1 or -1.
            NotImplementedError: If side is -1 (short position sizing not implemented).
        Note: This calculation only makes sense for long positions. No implementation for short positions yet.
        Returns:
            float: The quantity of the position that can be bought based on the sizing type.

        """
        if side not in [1, -1]:
            raise ValueError("side must be either 1 (long) or -1 (short).")
        if side == -1:
            raise NotImplementedError("Short position sizing is not implemented yet.")

        if self._unavailable:
            return override_calculate_position_size(**overrides)

        else:
            self.update_delta_limit(signal_id, position_id, date)  ## Always calculate the delta limit first
            logger.info(
                f"Calculating Quantity for Position ID: {position_id} and Signal ID: {signal_id} on Date: {date}"
            )
            if position_id not in self.rm.position_data:  ## If the position data isn't available, calculate the greeks
                self.rm.calculate_position_greeks(position_id, date)

            ## First get position info and ticker
            position_dict, _ = self.rm.parse_position_id(position_id)
            key = list(position_dict.keys())[0]
            ticker = swap_ticker(position_dict[key][0]["ticker"])

            ## Now calculate the max size cash can buy
            cash_available = self.pm.allocated_cash_map[ticker]
            purchase_date = pd.to_datetime(date)
            s0_at_purchase = self.rm.position_data[position_id]["s"][
                purchase_date
            ]  ## s -> chain spot, s0_close -> adjusted close
            logger.info(f"Spot Price at Purchase: {s0_at_purchase} at time {purchase_date}")
            logger.info(
                f"Cash Available: {cash_available}, Option Price: {opt_price}, Cash_Available/OptPRice: {(cash_available/(opt_price*100))}"
            )
            max_size_cash_can_buy = abs(
                math.floor(cash_available / (opt_price * 100))
            )  ## Assuming Allocated Cash map is already in 100s

            delta = self.rm.position_data[position_id]["Delta"][purchase_date]
            target_delta = self.rm.greek_limits["delta"][signal_id]
            logger.info(f"Target Delta: {target_delta}")
            delta_size = math.floor(target_delta / abs(delta))
            logger.info(f"Delta from Full Cash Spend: {max_size_cash_can_buy * delta}, Size: {max_size_cash_can_buy}")
            logger.info(f"Delta with Size Limit: {delta_size * delta}, Size: {delta_size}")
            return delta_size if abs(delta_size) <= abs(max_size_cash_can_buy) else max_size_cash_can_buy

    def daily_update(self):
        """
        Daily update method to refresh the position data and limits.
        """
        pass

    def pre_analyze_task(self, *args, **kwargs):
        """
        Abstract method for daily updates.
        """
        pass

    def post_analyze_task(self, *args, **kwargs):
        """
        Abstract method for daily updates.
        """
        pass


class ZscoreRVolSizer(BaseSizer):
    """
    Sizer that calculates position size based on the zscore of the realized volatility.
    """

    VOL_TYPES = ["mean", "weighted_mean", "window"]

    def __init__(
        self,
        pm: OptionSignalPortfolio,
        rm: RiskManager,
        sizing_lev: float = 1.0,
        rvol_window: int | tuple = None,
        rolling_window: int = 100,
        weights=(0.5, 0.3, 0.2),
        vol_type: str = "mean",
        *args,
        **kwargs,
    ):
        """
        Initialize the ZscoreRVolSizer with a PowerManager and ResourceManager.
        Args:
            pm (OptionSignalPortfolio): The PowerManager instance managing the portfolio.
            rm (RiskManager): The ResourceManager instance managing resources.
            sizing_lev (float): The sizing level for position sizing. Default is 1.0.
            rvol_window (int): The window size for realized volatility calculation. Default is 30 days.
            rolling_window (int): The window size for rolling calculations. Default is 100 days.
            weights (tuple): Weights for the weighted mean calculation. Default is (0.5, 0.3, 0.2). Corresponding to 5D, 20D, and 63D realized volatility respectively.
        """
        super().__init__(pm, rm, sizing_lev)

        if isinstance(weights, (list)):
            weights = tuple(weights)  ## Ensure weights is a tuple
        if isinstance(rvol_window, (list)):
            rvol_window = tuple(rvol_window)  ## Ensure rvol_window is a tuple if provided as a list

        rvol_window = self.__rvol_window_assert(
            vol_type, rvol_window
        )  ## Assert that the rvol_window is valid based on the vol_type
        assert vol_type in self.VOL_TYPES, f"vol_type must be one of {self.VOL_TYPES}, got {self.vol_type}"
        assert isinstance(weights, tuple) and len(weights) == 3, "weights must be a tuple of length 3"
        assert sum(weights) == 1, "weights must sum to 1"
        assert rolling_window > 0, "rolling_window must be a positive integer"

        ## Ensure underlier_list is provided if pm and rm are not provided
        if self._unavailable:
            assert (
                "underlier_list" in kwargs
            ), "underlier_list must be provided in kwargs when pm and rm are not provided."
            assert (
                isinstance(kwargs["underlier_list"], list) and len(kwargs["underlier_list"]) > 0
            ), "underlier_list must be a non-empty list."
            self.underlier_list = kwargs["underlier_list"]
        else:
            logger.critical(
                "ZscoreRVolSizer: underlier_list will be derived from the portfolio's allocated cash map keys."
            )
            self.underlier_list = list(self.pm.allocated_cash_map.keys())

        self.__rvol_window = rvol_window
        self.__rolling_window = rolling_window
        self.rvol_timeseries = {}
        self.z_i = {}
        self.vol_type = vol_type
        self.norm_constant = kwargs.get(
            "norm_constant", kwargs.get("norm_const", 1.0)
        )  ## Normalization constant for the scaler
        self.weights = weights  ## Weights for the weighted mean calculation
        self.scaler = ZcoreScalar(
            rvol_window=self.rvol_window,
            rolling_window=self.rolling_window,
            weights=self.weights,
            vol_type=self.vol_type,
            norm_constant=self.norm_constant,
            syms=self.underlier_list,
        )
        self._initialize = True

    def __setattr__(self, name, value):
        reload_scaler_attrs = ["rvol_window", "rolling_window", "weights", "vol_type", "norm_constant"]
        if name in reload_scaler_attrs and getattr(self, "_initialize", False):
            logger.info(f"Attribute {name} changed, reloading scalers.")
            self.scaler.__setattr__(name, value)
            self.scaler.reload()
            self.rvol_timeseries = {}  # Clear the cache if the window changes
        return super().__setattr__(name, value)

    def __rvol_window_assert(self, vol_type: str, rvol_window: int | tuple = None) -> int | tuple:
        """
        Assert that the rvol_window is a positive integer or a tuple of three integers.
        """
        if rvol_window is None:  ## If rvol_window is not provided, set it to default
            rvol_window = 30 if vol_type == "window" else (5, 20, 63)

        if isinstance(
            rvol_window, int
        ):  ## If rvol_window is an int, it must be a positive integer and vol_type must be 'window'
            assert vol_type == "window", "rvol_window must be an int only when vol_type is 'window'."
            assert rvol_window > 0, "rvol_window must be a positive integer."

        elif isinstance(
            rvol_window, tuple
        ):  ## If rvol_window is a tuple, it must be of length 3 and vol_type must be 'weighted_mean' or 'mean'
            assert (
                vol_type == "weighted_mean" or vol_type == "mean"
            ), "rvol_window must be a tuple only when vol_type is 'weighted_mean' or 'mean'."
            assert len(rvol_window) == 3, "rvol_window must be a tuple of length 3 for weighted mean calculation."

        return rvol_window

    @property
    def rvol_window(self):
        return self.__rvol_window

    @rvol_window.setter
    def rvol_window(self, value):
        self.__rvol_window_assert(self.vol_type, value)  ## Assert that the rvol_window is valid based on the vol_type

        if value != self.__rvol_window:  ## Clear the cache if the window changes
            logger.info(f"Setting realized volatility window to {value} days")
            self.rvol_timeseries = {}  # Clear the cache if the window changes
        self.__rvol_window = value

    @property
    def rolling_window(self):
        return self.__rolling_window

    @rolling_window.setter
    def rolling_window(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("rolling_window must be a positive integer.")
        self.__rolling_window = value

    def get_daily_delta_limit(
        self, signal_id: str = None, position_id: str = None, date: str | datetime = None, **overrides
    ) -> float:
        """
        Calculate the delta limit based on the percentile of the realized volatility.
        This method is called to get the delta limit for a given signal and position.
        Args:
            signal_id (str): The ID of the signal.
            position_id (str): The ID of the position.
            date (str|datetime): The date for which the delta limit is calculated.
        Overrides:
            current_cash (float): The current cash available for the position.
            underlier_price_at_time (float): The price of the underlier at the time of calculation.

        Returns:
            float: The scaled delta size based on the realized volatility scaler.

        Limit Calc: (allocated_cash * lev)/(S0 * 100) * 1/(1+Zscore(rvol(30), window))
        """
        id_details = parse_signal_id(signal_id)
        symbol = swap_ticker(id_details["ticker"])

        ##NEED TO:
        ## Load the rvol_timeseries for the symbol if not already loaded
        if symbol not in self.scaler.scalers:
            self.scaler.load_scalers([symbol])

        ## Get the scaler for the symbol and date
        scaler = self.scaler.get_scaler_on_date(symbol, date)

        if self._unavailable:
            ## Calculate the equivalent delta size as normal
            try:
                current_cash = overrides.pop("current_cash")
                underlier_price_at_time = overrides.pop("underlier_price_at_time")
            except KeyError as e:
                raise ValueError(f"Missing required override parameter: {e}") from e

            equivalent_delta_size = zscore_rvol_delta_limit(
                cash_available=current_cash,
                sizing_lev=self.sizing_lev,
                underlier_price_at_time=underlier_price_at_time,
                _scaler=scaler,
            )
            return equivalent_delta_size

        else:
            logger.info(
                f"ZscoreRVolSizer: Calculating Delta Limit for Signal ID: {signal_id} and Position ID: {position_id} on Date: {date}\n"
            )
            id_details = parse_signal_id(signal_id)
            starting_cash = self.get_cash(id_details["ticker"], signal_id, position_id)
            logger.info(
                f"Starting Cash for {id_details['ticker']} on {date}: {starting_cash} vs Current Cash: {self.pm.allocated_cash_map[id_details['ticker']]}"
            )
            delta_at_purchase = self.rm.position_data[position_id]["Delta"][date]
            s0_at_purchase = self.rm.position_data[position_id]["s"][date]
            equivalent_delta_size = (
                ((starting_cash * self.sizing_lev) / (s0_at_purchase * 100)) if s0_at_purchase != 0 else 0
            )
            scaled_delta_size = equivalent_delta_size * scaler
            logger.info(f"Scaler for {symbol} on {date}: {scaler}")
            logger.info(
                f"Spot Price at Purchase: {s0_at_purchase} at time {date}  ## This is the delta at the time of purchase"
            )
            logger.info(f"Delta at Purchase: {delta_at_purchase}")
            logger.info(
                f"Equivalent Delta Size: {equivalent_delta_size}, with Cash Available: {starting_cash}, Leverage: {self.sizing_lev} and Scaler: {scaler}"
            )
            logger.info(f"Scaled Delta Size: {scaled_delta_size}")
            return scaled_delta_size

    def update_delta_limit(self, signal_id: str, position_id: str, date: str | datetime) -> float:
        """
        Calculate the delta limit based on the percentile of the realized volatility.
        This method is called to update the delta limit for a given signal and position.
        Args:
            signal_id (str): The ID of the signal.
            position_id (str): The ID of the position.
            date (str|datetime): The date for which the delta limit is calculated.

        This method checks if the delta limit for the signal ID is already updated. If not, it calculates the delta limit based on the realized volatility and updates the risk manager's greek limits.
        """

        id_details = parse_signal_id(signal_id)
        symbol = swap_ticker(id_details["ticker"])
        self.register_signal_starting_cash(
            signal_id, self.pm.allocated_cash_map[symbol]
        )  ## Register the starting cash for the signal
        self.register_position_id_starting_cash(
            position_id, self.pm.allocated_cash_map[symbol]
        )  ## Register the starting cash for the position ID
        logger.info(f"Updating Greek Limits for Signal ID: {signal_id} and Position ID: {position_id}")
        scaled_delta_size = self.get_daily_delta_limit(signal_id, position_id, date)
        self.rm.greek_limits["delta"][signal_id] = abs(scaled_delta_size)
        return scaled_delta_size

    def calculate_position_size(
        self,
        signal_id: str = None,
        position_id: str = None,
        opt_price: float = None,
        date: str | datetime = None,
        side: int = 1,
        **overrides,
    ) -> float:
        """
        Calculate the position size based on the percentile of the realized volatility.
        This method calculates the position size for a given signal ID and position ID based on the available cash and the delta limit.
        Args:
            signal_id (str): The ID of the signal.
            position_id (str): The ID of the position.
            opt_price (float): The price of the option.
            date (str|datetime): The date for which the position size is calculated.
            side (int): The side of the position, 1 for long and -1 for short. Default is 1.
        Overrides:
            current_cash (float): The current cash available for the position.
            option_price_at_time (float): The price of the option at the time of calculation.
            delta (float): The delta of the position at the time of calculation.
            delta_limit (float): The delta limit for the position.

        Returns:
            float: The quantity of the position that can be bought based on the sizing type.

        Note: This calculation only makes sense for long positions. No implementation for short positions yet.
        """
        if side not in [1, -1]:
            raise ValueError("side must be either 1 (long) or -1 (short).")
        if side == -1:
            raise NotImplementedError("Short position sizing is not implemented yet.")

        if self._unavailable:
            return override_calculate_position_size(**overrides)

        self.update_delta_limit(signal_id, position_id, date)  ## Always calculate the delta limit first
        logger.info(f"Calculating Quantity for Position ID: {position_id} and Signal ID: {signal_id} on Date: {date}")
        if position_id not in self.rm.position_data:  ## If the position data isn't available, calculate the greeks
            self.rm.calculate_position_greeks(position_id, date)

        ## First get position info and ticker
        position_dict, _ = self.rm.parse_position_id(position_id)
        key = list(position_dict.keys())[0]
        ticker = swap_ticker(position_dict[key][0]["ticker"])

        ## Now calculate the max size cash can buy
        cash_available = self.pm.allocated_cash_map[ticker]
        purchase_date = pd.to_datetime(date)
        s0_at_purchase = self.rm.position_data[position_id]["s"][
            purchase_date
        ]  ## s -> chain spot, s0_close -> adjusted close
        logger.info(f"Spot Price at Purchase: {s0_at_purchase} at time {purchase_date}")
        logger.info(
            f"Cash Available: {cash_available}, Option Price: {opt_price}, Cash_Available/OptPRice: {(cash_available/(opt_price*100))}"
        )
        max_size_cash_can_buy = abs(
            math.floor(cash_available / (opt_price * 100))
        )  ## Assuming Allocated Cash map is already in 100s

        delta = self.rm.position_data[position_id]["Delta"][purchase_date]
        target_delta = self.rm.greek_limits["delta"][signal_id]
        logger.info(f"Target Delta: {target_delta}")
        delta_size = math.floor(target_delta / abs(delta))
        logger.info(f"Delta from Full Cash Spend: {max_size_cash_can_buy * delta}, Size: {max_size_cash_can_buy}")
        logger.info(f"Delta with Size Limit: {delta_size * delta}, Size: {delta_size}")
        return max(delta_size if abs(delta_size) <= abs(max_size_cash_can_buy) else max_size_cash_can_buy, 1)

    def calculate_scaler(self, symbol: str) -> None:
        """
        Calculate the scaler for the realized volatility timeseries.
        """
        raise DeprecationWarning("This method is deprecated. Use the ZcoreScalar class for scaler calculations.")

        if symbol not in self.rvol_timeseries:
            self.load_rvol_timeseries(symbol)

        rvol = self.rvol_timeseries[symbol]
        rolling_mean = rvol.rolling(window=self.rolling_window).mean()
        rolling_std = rvol.rolling(window=self.rolling_window).std()
        z_i = (rvol - rolling_mean) / rolling_std
        scaler = self.norm_constant / (1 + z_i.abs())
        self.z_i[symbol] = z_i
        self.scaler[symbol] = scaler

    def load_rvol_timeseries(self, symbol: str) -> None:
        """
        Load the realized volatility timeseries for a given symbol.
        """

        raise DeprecationWarning("This method is deprecated. Use the ZcoreScalar class for scaler calculations.")

        logger.info(
            f"Loading realized volatility timeseries for {symbol} with vol_type: {self.vol_type} and rvol_window: {self.rvol_window}"
        )

        def weighted_mean(series: pd.Series) -> pd.Series:
            """
            Calculate the weighted mean of a series.
            """
            logger.info(
                f"Calculating weighted mean for {symbol} with weights: {self.weights} and rvol_window: {self.rvol_window}"
            )
            w1, w2, w3 = self.weights
            win1, win2, win3 = self.rvol_window
            rvol_series1 = self.pm.get_underlier_data(symbol).rvol(
                ts_start=pd.to_datetime(self.rm.start_date) - relativedelta(years=1),
                ts_end=pd.to_datetime(self.rm.end_date) + BDay(1),
                window=win1,
                log=True,
            )[f"rvol_{win1}D"]
            rvol_series2 = self.pm.get_underlier_data(symbol).rvol(
                ts_start=pd.to_datetime(self.rm.start_date) - relativedelta(years=1),
                ts_end=pd.to_datetime(self.rm.end_date) + BDay(1),
                window=win2,
                log=False,
            )[f"rvol_{win2}D"]
            rvol_series3 = self.pm.get_underlier_data(symbol).rvol(
                ts_start=pd.to_datetime(self.rm.start_date) - relativedelta(years=1),
                ts_end=pd.to_datetime(self.rm.end_date) + BDay(1),
                window=win3,
                log=False,
            )[f"rvol_{win3}D"]
            rvol_series = pd.concat([rvol_series1 * w1, rvol_series2 * w2, rvol_series3 * w3], axis=1).sum(axis=1)
            return rvol_series.rename(symbol)

        def mean(series: pd.Series) -> pd.Series:
            """
            Calculate the weighted mean of a series.
            """
            logger.info(f"Calculating mean for {symbol} with rvol_window: {self.rvol_window}")
            win1, win2, win3 = self.rvol_window
            rvol_series1 = self.pm.get_underlier_data(symbol).rvol(
                ts_start=pd.to_datetime(self.rm.start_date) - relativedelta(years=1),
                ts_end=pd.to_datetime(self.rm.end_date) + BDay(1),
                window=win1,
                log=True,
            )[f"rvol_{win1}D"]
            rvol_series2 = self.pm.get_underlier_data(symbol).rvol(
                ts_start=pd.to_datetime(self.rm.start_date) - relativedelta(years=1),
                ts_end=pd.to_datetime(self.rm.end_date) + BDay(1),
                window=win2,
                log=False,
            )[f"rvol_{win2}D"]
            rvol_series3 = self.pm.get_underlier_data(symbol).rvol(
                ts_start=pd.to_datetime(self.rm.start_date) - relativedelta(years=1),
                ts_end=pd.to_datetime(self.rm.end_date) + BDay(1),
                window=win3,
                log=False,
            )[f"rvol_{win3}D"]
            rvol_series = pd.concat([rvol_series1, rvol_series2, rvol_series3], axis=1).mean(axis=1)
            return rvol_series.rename(symbol)

        def window(series: pd.Series) -> pd.Series:
            """
            Calculate the windowed mean of a series.
            """
            logger.info(f"Calculating windowed mean for {symbol} with rvol_window: {self.rvol_window}")
            rvol_series1 = self.pm.get_underlier_data(symbol).rvol(
                ts_start=pd.to_datetime(self.rm.start_date) - relativedelta(years=1),
                ts_end=pd.to_datetime(self.rm.end_date) + BDay(1),
                window=self.rvol_window,
                log=True,
            )[f"rvol_{self.rvol_window}D"]
            return rvol_series1.rename(symbol)

        if symbol not in self.rvol_timeseries:
            if self.vol_type == "mean":
                rvol_series = mean(pd.Series())
            elif self.vol_type == "weighted_mean":
                rvol_series = weighted_mean(pd.Series())
            elif self.vol_type == "window":
                rvol_series = window(pd.Series())
            else:
                raise ValueError(f"Unknown vol_type: {self.vol_type}. Available types: {self.VOL_TYPES}")
            self.rvol_timeseries[symbol] = rvol_series

    def daily_update(self):
        ### Consider implementing:
        ### - Increase & Reduce position delta based on scaler
        ### - Reduce only using min(limit, today_delta_limit)
        pass

    def pre_analyze_task(self, *args, **kwargs):
        """
        Abstract method for daily updates.
        """
        pass

    def post_analyze_task(self, *args, **kwargs):
        """
        Abstract method for daily updates.
        """
        pass
