"""Position Sizing Calculation Utilities and Helper Functions.

This module provides core calculation functions for position sizing strategies,
including fixed delta limits, volatility-adjusted sizing, and z-score-based
scaling. These functions support various sizer implementations by providing
consistent mathematical foundations.

Key Functions:
    default_delta_limit:
        - Calculates base delta limit from available cash
        - Uses fixed sizing leverage multiplier
        - Returns equivalent delta exposure
        - Foundation for DefaultSizer

    zscore_rvol_delta_limit:
        - Calculates volatility-adjusted delta limit
        - Scales by realized volatility z-score
        - Increases limits in low-vol, decreases in high-vol
        - Used by ZscoreRVolSizer

    calculate_position_quantity:
        - Converts delta limit to contract quantity
        - Accounts for option delta per contract
        - Ensures integer quantities
        - Respects minimum quantity constraints

    compute_rvol_zscore:
        - Calculates realized volatility z-score
        - Uses rolling window statistics
        - Normalizes current vol vs historical
        - Returns scaler for limit adjustment

Delta Limit Calculations:
    Equivalent Delta Size:
        - Represents directional exposure in delta-one terms
        - If buying $10,000 of stock (delta=1): 10,000/price = contracts
        - Sizing leverage allows exceeding cash allocation
        - Example: $10k cash, 2x leverage, $150 stock = 133 delta limit

    Formula (Default):
        delta_limit = (cash * sizing_lev) / (underlier_price * 100)

        Components:
        - cash: Available capital for position
        - sizing_lev: Leverage multiplier (1.0 = no leverage)
        - underlier_price: Current stock price
        - 100: Option contract multiplier

    Formula (Z-score Adjusted):
        delta_limit = (cash * sizing_lev * scaler) / (underlier_price * 100)

        Additional:
        - scaler: Volatility-based adjustment (0.5 to 2.0 typical)
        - Higher vol → lower scaler → smaller positions
        - Lower vol → higher scaler → larger positions

Volatility Z-Score Calculation:
    1. Calculate realized volatility (rolling std of returns)
    2. Compute z-score vs historical distribution:
       z = (current_rvol - mean_rvol) / std_rvol
    3. Convert z-score to scaler:
       scaler = f(z)  # Mapping function, often inverse
    4. Apply bounds (typically 0.5 to 2.0)

    Interpretation:
        z > 1: High volatility period → reduce position size
        z = 0: Normal volatility → standard position size
        z < -1: Low volatility period → increase position size

Position Quantity Calculation:
    Given:
    - delta_limit: Maximum allowable delta
    - option_delta: Delta per contract (e.g., 0.35)
    - structure_delta: Net delta of multi-leg structure

    Formula:
        quantity = floor(delta_limit / abs(structure_delta * 100))

    Constraints:
        - Must be positive integer
        - Minimum 1 contract (if any position)
        - Never exceed delta_limit
        - Account for spread structure deltas

Sizing Leverage Examples:
    1.0x (Conservative):
        - Use 100% of available cash as limit
        - No leverage applied
        - Lower risk, smaller positions

    2.0x (Moderate):
        - Use 200% of cash as delta limit
        - Moderate leverage
        - Allows larger positions in good setups

    3.0x (Aggressive):
        - Use 300% of cash as delta limit
        - High leverage
        - Requires careful risk management

Volatility Regimes:
    Low Volatility (z-score < -1):
        - Scaler = 1.5 to 2.0
        - Increase position sizes
        - Markets are calmer
        - Risk of complacency

    Normal Volatility (-1 < z < 1):
        - Scaler = 0.8 to 1.2
        - Standard position sizing
        - Most common regime
        - No special adjustments

    High Volatility (z-score > 1):
        - Scaler = 0.5 to 0.8
        - Reduce position sizes
        - Markets are turbulent
        - Protect capital

Usage:
    # Basic delta limit
    delta_limit = default_delta_limit(
        cash_available=10000.0,
        sizing_lev=2.0,
        underlier_price_at_time=150.0
    )
    # Result: (10000 * 2.0) / (150 * 100) = 1.33 delta

    # Volatility-adjusted limit
    rvol_scaler = compute_rvol_zscore(underlier='AAPL', date=date)
    delta_limit = zscore_rvol_delta_limit(
        cash_available=10000.0,
        sizing_lev=2.0,
        underlier_price_at_time=150.0,
        _scaler=rvol_scaler
    )

    # Convert to quantity
    quantity = calculate_position_quantity(
        delta_limit=1.33,
        structure_delta=0.35,  # Per contract
        min_quantity=1
    )

Integration:
    - DefaultSizer calls default_delta_limit()
    - ZscoreRVolSizer calls zscore_rvol_delta_limit()
    - All sizers use calculate_position_quantity()
    - RVolCalculator provides compute_rvol_zscore()

Constants:
    Y2_LAGGED_START_DATE: Historical data start for vol calculations
    CONTRACT_MULTIPLIER: Standard option contract size (100)
    MIN_SCALER: Lower bound for vol adjustment (0.5)
    MAX_SCALER: Upper bound for vol adjustment (2.0)

Notes:
    - Division by zero protected (returns 0 if price=0)
    - All quantities rounded down (floor) for safety
    - Negative quantities converted to absolute values
    - Leverage can be fractional (e.g., 1.5x)
    - Volatility lookback periods configurable
"""

from datetime import datetime
import math
from dataclasses import dataclass, field
from typing import Literal, ClassVar
import pandas as pd
from EventDriven.riskmanager.utils import logger
from ..._vars import Y2_LAGGED_START_DATE
from ..market_data import get_timeseries_obj


def default_delta_limit(
    cash_available: float,
    sizing_lev: float,
    underlier_price_at_time: float,
):
    """
    This function calculates the equivalent delta size based on the cash available * sizing leverage.
    Equivalent delta size is the amount of delta exposure that can be taken on based on the cash available if buying Delta 1.
    Sizing leverage allows for increasing the amount of delta exposure that can be taken on.
    args:
    -------
    cash_available: Cash available for trading
    sizing_lev: Sizing leverage. This is a multiplier that increases the equivalent delta size
    underlier_price_at_time: Price of the underlier at the time of calculation
    returns:
    ---------
    equivalent_delta_size: Equivalent delta size based on cash available and sizing leverage
    """
    equivalent_delta_size = (
        ((cash_available * sizing_lev) / (underlier_price_at_time * 100)) if underlier_price_at_time != 0 else 0
    )
    return equivalent_delta_size


def zscore_rvol_delta_limit(
    cash_available: float, sizing_lev: float, underlier_price_at_time: float, _scaler: float
) -> float:
    """
    This function calculates the equivalent delta size based on the cash available * sizing leverage and a z-score scaler.
    Equivalent delta size is the amount of delta exposure that can be taken on based on the cash available if buying Delta 1.
    Sizing leverage allows for increasing the amount of delta exposure that can be taken on.
    The z-score scaler adjusts the equivalent delta size based on the realized volatility of the underlier.
    args:
    --------
    cash_available: Cash available for trading
    sizing_lev: Sizing leverage. This is a multiplier that increases the equivalent delta size
    underlier_price_at_time: Price of the underlier at the time of calculation
    scaler: Z-score scaler based on the realized volatility of the underlier
    returns:
    ----------
    equivalent_delta_size: Equivalent delta size based on cash available, sizing leverage and z-score scaler
    """

    equiv = default_delta_limit(
        cash_available=cash_available, sizing_lev=sizing_lev, underlier_price_at_time=underlier_price_at_time
    )
    return equiv * _scaler


def delta_position_sizing(
    cash_available: float,
    option_price_at_time: float,
    delta: float,
    delta_limit: float,
) -> int:
    """
    Calculate the position size based on delta and cash available.
    Args:
        cash_available (float): The cash available for trading.
        option_price_at_time (float): The price of the option at the time of calculation.
        delta (float): The delta of the option.
        delta_limit (float): The maximum delta exposure allowed.
    Returns:
        int: The calculated position size."""
    ## TODO: Add docstring
    ## TODO: Raise error if delta is 0 or cash_available is <= 0
    if delta == 0 or math.isnan(delta) or cash_available <= 0 or option_price_at_time <= 0:
        logger.critical(
            f"Delta is 0/NaN or cash_available is <= 0 or option_price_at_time <= 0. delta: {delta}, cash_available: {cash_available}, option_price_at_time: {option_price_at_time}. This is intended to be long only sizing. Returning 0."
        )
        return 0
    try:
        delta_size = math.floor(delta_limit / abs(delta))
    except (ValueError, ZeroDivisionError) as e:
        logger.critical(
            f"Error calculating delta_size: {e}. delta: {delta}, delta_limit: {delta_limit}. Returning size 0."
        )
        raise e
    max_size_cash_can_buy = abs(math.floor(cash_available / (option_price_at_time * 100)))
    # size = max(delta_size if abs(delta_size) <= abs(max_size_cash_can_buy) else max_size_cash_can_buy, 1)

    ## Opting to return 0 if size is 0
    size = delta_size if abs(delta_size) <= abs(max_size_cash_can_buy) else max_size_cash_can_buy
    return size


def raise_none(param, name):
    if param is None:
        raise ValueError(f"{name} cannot be None")


def realized_vol(series: pd.Series, window: int) -> pd.Series:
    return series.pct_change().rolling(window).std() * (252**0.5)


def weighted_realized_vol(series: pd.Series, windows: tuple, weights: tuple) -> pd.Series:
    vols = [realized_vol(series, w) for w in windows]
    weighted_vol = sum(w * v for w, v in zip(weights, vols))
    return weighted_vol


def mean_realized_vol(series: pd.Series, windows: tuple) -> pd.Series:
    vols = [realized_vol(series, w) for w in windows]
    mean_vol = sum(vols) / len(vols)
    return mean_vol


def scaler(realized_vol_series: pd.Series, factor: float, window: int) -> pd.Series:
    rolling_mean = realized_vol_series.rolling(window=window).mean()
    rolling_std = realized_vol_series.rolling(window=window).std()
    z = (realized_vol_series - rolling_mean) / rolling_std
    return factor / (1 + z.abs())


@dataclass
class ZcoreScalar:
    rvol_window: int | tuple
    rolling_window: int
    weights: tuple
    vol_type: Literal["mean", "weighted", "window"]
    norm_constant: int = 1.0
    rvol_timeseries: dict = field(default_factory=dict)
    VOL_TYPES: ClassVar[set] = {"mean", "weighted_mean", "window"}  ## TODO: Align ZscoreRvolSizer to this
    syms: list = field(default_factory=list)
    interval: str = "1d"
    scalers: dict = field(default_factory=dict)

    def __post_init__(self):
        ## Ensure all lists are tuples

        if self.interval != "1d":
            raise NotImplementedError("Not allowed to change interval yet.")

        if isinstance(self.weights, list):
            self.weights = tuple(self.weights)

        if isinstance(self.rvol_window, list):
            self.rvol_window = tuple(self.rvol_window)

        ## Assert rvol_window is valid
        self.rvol_window = self.__rvol_window_assert(self.vol_type, self.rvol_window)
        assert self.vol_type in self.VOL_TYPES, f"vol_type must be one of {self.VOL_TYPES}, got {self.vol_type}"
        assert isinstance(self.weights, tuple) and len(self.weights) == 3, "weights must be a tuple of length 3"
        assert sum(self.weights) == 1, "weights must sum to 1"
        assert self.rolling_window > 0, "rolling_window must be a positive integer"

    def reload(self):
        """
        Reload the scalers for all symbols in self.syms
        """
        self.scalers = {}

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

    def __repr__(self) -> str:
        return f"ZcoreScalar(rvol_window={self.rvol_window}, syms={self.syms})"

    def load_scalers(self, syms: list = None, force=False) -> None:
        """
        Load the z-score scalers for the specified symbols.
        args:
        -------
        syms: List of symbols to load. If None, load all symbols in self.syms
        force: If True, reload scalers even if they already exist
        Note: syms must be in self.syms. No loading unassociated syms
        Raises:
        -------
        ValueError: If any symbol in syms is not in self.syms
        TypeError: If syms is not a list
        Returns:
        -------
        None: Values are stored in self.scalers
        """

        ## Get timeseries object
        timeseries = get_timeseries_obj()

        ## If syms is None, use the existing syms
        ## This is to avoid reloading timeseries if already loaded
        if syms is None:
            syms = self.syms

        if not isinstance(syms, list):
            raise TypeError(f"syms must be a list, got {type(syms)}")

        ## Raise error for syms not in self.syms. No loading unassociated syms
        unknown_syms = [s for s in syms if s not in self.syms]
        if unknown_syms:
            raise ValueError(f"Unknown syms: {unknown_syms}. Please add them to self.syms first.")

        ## Ensure syms is added to self.syms
        if not force:
            in_syms_bool = all(s in self.scalers.keys() for s in syms)
            syms = list(set(syms) - set(self.scalers.keys()))  ## Only load syms that are not already in self.scalers
        else:
            in_syms_bool = False

        ## If all syms are already in self.syms, do nothing
        ## Add new syms to self.syms
        if not in_syms_bool:
            self.syms.extend(syms)

        ## Load timeseries for each symbol and calculate the z-score scaler
        for sym in syms:
            timeseries.load_timeseries(
                sym=sym, start_date=Y2_LAGGED_START_DATE, end_date=datetime.now(), interval=self.interval
            )
            ts = timeseries.get_timeseries(sym=sym, interval=self.interval).spot["close"]

            if self.vol_type == "window":
                func = lambda x: realized_vol(x, self.rvol_window)
            elif self.vol_type == "weighted_mean":
                func = lambda x: weighted_realized_vol(x, self.rvol_window, self.weights)
            elif self.vol_type == "mean":
                func = lambda x: mean_realized_vol(x, self.rvol_window)
            else:
                raise ValueError(f"Unknown vol_type: {self.vol_type}")

            def _callable(sym, func=func, ts=ts):
                self.rvol_timeseries[sym] = func(ts)
                return scaler(self.rvol_timeseries[sym], self.norm_constant, self.rolling_window)

            self.scalers[sym] = _callable(sym)

    def append_syms(self, syms: list | str, ignore_load: bool = False):
        """
        Add syms to list of syms

        args:
        -------

        syms: List of syms, or a single sym
        ignore_load: ignores the load function if true
        """

        if isinstance(syms, str):
            if syms not in self.syms:
                self.syms.append(syms)
            syms = [syms]

        elif isinstance(syms, list):
            syms = [s for s in syms if s not in self.syms]
            self.syms.extend(syms)

        else:
            raise TypeError(f"Unknown format: {type(syms)}, expected [list, str]")

        if not ignore_load:
            self.load_scalers(syms=syms)

    def get_scaler(self, sym: str) -> pd.Series:
        """
        Get the z-score scaler for a specific symbol.

        args:
        -------
        sym: Symbol to get the scaler for

        returns:
        --------
        pd.Series: Z-score scaler for the symbol
        """
        if sym not in self.scalers:
            self.append_syms(sym)

        return self.scalers[sym]

    def get_scaler_on_date(self, sym: str, date: pd.Timestamp | str | datetime) -> float:
        """
        Get the z-score scaler for a specific symbol on a specific date.
        """
        if isinstance(date, str):
            date = pd.Timestamp(date).date()
        elif isinstance(date, datetime):
            date = date.date()

        scaler_ts = self.get_scaler(sym)
        date = pd.to_datetime(date).strftime("%Y-%m-%d")
        if date not in scaler_ts.index:
            raise ValueError(f"Date {date} not found in scaler_ts index for symbol {sym}.")

        return scaler_ts.loc[date]
