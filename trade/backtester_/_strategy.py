from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Tuple
import inspect
import pandas as pd
from trade.backtester_.data import PTDataset
from typing import List
from dataclasses import dataclass
import numpy as np
from plotly.subplots import make_subplots
from EventDriven.types import PositionEffect, SignalID
import plotly.graph_objects as go
from pandas.tseries.offsets import BDay  # noqa
from trade.helpers.helper import change_to_last_busday  # noqa
from ._types import Side, SideInt  # noqa
from trade.backtester_.indicators import (
    compute_atr_loss,
    update_atr_trail_long,
    update_atr_trail_short,
)


@dataclass
class Indicator:
    name: str
    series: pd.Series
    overlay: bool = False
    color: Optional[str] = "red"
    values: Optional[np.ndarray] = None

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise TypeError("Indicator name must be a string.")
        if not isinstance(self.series, pd.Series):
            raise TypeError("Indicator series must be a pandas Series.")
        if not isinstance(self.overlay, bool):
            raise TypeError("Indicator overlay must be a boolean.")
        if self.color is not None and not isinstance(self.color, str):
            raise TypeError("Indicator color must be a string or None.")
        if self.values is not None and not isinstance(self.values, np.ndarray):
            raise TypeError("Indicator values must be a numpy array or None.")
        if self.values is None:
            self.values = self.series.to_numpy(copy=False)


@dataclass
class TradeDecision:
    ok: bool
    side: int
    pos_effect: Optional[PositionEffect] = None
    signal_id: Optional[SignalID] = None
    sideint: Optional[int] = None
    side_enum: Optional[Side] = None

    def __post_init__(self):
        if not isinstance(self.ok, (bool, np.bool_)):
            raise TypeError("TradeDecision.ok must be a boolean.")
        if not isinstance(self.side, int):
            raise TypeError("TradeDecision.side must be an integer.")
        if self.side not in (1, -1, 0):
            raise ValueError("TradeDecision.side must be 1 (long), -1 (short), or 0 (no position).")
        if (self.signal_id is not None and not isinstance(self.signal_id, (SignalID, str))) and self.signal_id != "N/A":
            if isinstance(self.signal_id, str) and self.signal_id != "N/A":
                try:
                    self.signal_id = SignalID.parse(self.signal_id)
                except Exception as e:
                    raise ValueError(f"Invalid signal_id string: {self.signal_id}") from e
            raise TypeError(
                "TradeDecision.signal_id must be of type SignalID, 'N/A', or None. Received type: {}".format(
                    type(self.signal_id)
                )
            )
        if self.pos_effect is not None and not isinstance(self.pos_effect, PositionEffect):
            raise TypeError("TradeDecision.pos_effect must be of type PositionEffect or None.")

        ## Enforcing needed information:
        if self.ok:
            if self.side == 0:
                raise ValueError("If ok is True, side cannot be 0 (no position).")
            if self.pos_effect is None:
                raise ValueError("If ok is True, pos_effect must be provided.")
            if self.signal_id is None:
                raise ValueError(
                    "If ok is True, signal_id must be provided. If this is a close with no prior signal, set signal_id to N/A"
                )

            ## If Open signal_id must be parseable, if Close signal_id must be parseable or N/A
            if self.pos_effect == PositionEffect.OPEN:
                if self.signal_id == "N/A":
                    raise ValueError("If pos_effect is OPEN, signal_id cannot be 'N/A'. It must be a valid SignalID.")
                try:
                    SignalID.parse(self.signal_id)
                except Exception as e:
                    raise ValueError(f"Invalid signal_id: {self.signal_id}") from e
            elif self.pos_effect == PositionEffect.CLOSE:
                if self.signal_id != "N/A":
                    try:
                        SignalID.parse(self.signal_id)
                    except Exception as e:
                        raise ValueError(f"Invalid signal_id: {self.signal_id}") from e

        self.ok = bool(self.ok)
        self.sideint = int(self.side)
        self.side_enum = Side.LONG if self.sideint == 1 else Side.SHORT if self.sideint == -1 else None

    def __bool__(self):
        return self.ok

    def __repr__(self):
        return (
            f"TradeDecision(ok={self.ok}, side={self.side}, signal_id={self.signal_id}, pos_effect={self.pos_effect})"
        )


@dataclass
class PositionInfo:
    entry_date: Optional[pd.Timestamp] = None
    entry_price: Optional[float] = None
    side: Optional[SideInt] = None
    signal_id: Optional[SignalID] = None

    def __post_init__(self):
        ## If any of the fields are set, they must all be set (for simplicity of logic elsewhere)
        fields = [self.entry_date, self.entry_price, self.side, self.signal_id]
        if any(f is not None for f in fields) and not all(f is not None for f in fields):
            raise ValueError("If any of entry_date, entry_price, side, or signal_id is set, they must all be set.")

    def __bool__(self):
        return (
            self.entry_date is not None
            and self.entry_price is not None
            and self.side is not None
            and self.signal_id is not None
        )


class StrategyBase(ABC):
    """
    Abstract base class for trading strategies with built-in backtest parameter validation.

    This class provides a framework for building trading strategies with automatic parameter
    validation and efficient data access using numpy arrays. It enforces that all subclasses
    declare their required parameters via the `bt_params` class attribute.

    Subclasses MUST define:
        bt_params: Dict[str, Any] = {"param_name": default_or_REQUIRED, ...}

    Key Features:
    - Automatic parameter validation at class definition time
    - Efficient data access via numpy arrays for OHLCV data
    - Built-in indicator management system
    - Position tracking and state management with support for long/short positions
    - Simulation and visualization capabilities

    Data Structures:
    - TradeDecision: Dataclass returned by should_open() and should_close() containing:
      - ok (bool): Whether the trade decision is valid
      - side (int): Position side (1 for long, -1 for short)
    - PositionInfo: Dataclass tracking current position details:
      - entry_date (pd.Timestamp): Date position was opened
      - entry_price (float): Price at which position was opened
      - side (SideInt): Position side (SideInt.BUY or SideInt.SELL)
    - Side/SideInt: Enum types for position direction (BUY=1, SELL=-1)

    Methods to Override (Required):
    - setup(): Initialize indicators and strategy-specific state
    - is_open_signal(): Define logic for opening signals
    - is_close_signal(): Define logic for closing signals
    - open_action(): Execute actions when opening a position
    - close_action(): Execute actions when closing a position

    Methods to Override (Optional):
    - should_trade(): Customize trading eligibility logic. Ideally checks market regime, time filters, etc.
    - should_open(): Customize position opening logic, returns TradeDecision. Ideally this takes into account both
      - should_trade() and is_open_signal()
    - should_close(): Customize position closing logic, returns TradeDecision. Ideally this takes into account both
      - is_close_signal() and position status
    - info_on_date(): Get strategy state at a specific date/index
    - have_position(): Check if a position is currently open
    - reset_strategy_state(): Reset position and stop-loss state

    Built-in Methods (Do Not Override):
    - simulate(): Runs the backtest simulation
    - plot_strategy_indicators(): Visualizes strategy performance
    - plot_signals(): Visualizes buy/sell signals
    - add_indicator(): Add indicators to the strategy
    - get_indicator(): Retrieve indicator values

    Attributes:
    - data (PTDataset): Market data container
    - position_open (bool): Current position status
    - position_side (SideInt): Current position direction (BUY or SELL)
    - position_info (PositionInfo): Current position details (entry date, price, side)
    - stop (Optional[float]): Stop-loss price level
    - indicators (Dict[str, Any]): Dictionary of strategy indicators
    - close, open, high, low, volume, dates: Numpy array properties for efficient data access
    """

    def __init_subclass__(cls, **kwargs):
        """
        Validates that subclasses properly define bt_params and __init__ signature.

        This hook is automatically called when a subclass is created. It ensures:
        1. bt_params exists and is a dictionary
        2. All keys in bt_params correspond to parameters in __init__ (unless **kwargs is used)

        Args:
            **kwargs: Additional keyword arguments passed to parent __init_subclass__

        Raises:
            TypeError: If bt_params is missing, not a dict, or contains keys not in __init__
        """
        super().__init_subclass__(**kwargs)
        must_have_in_init = [
            "data",
            "start_trading_date",
            "ticker",
            "tplusn",
        ]
        is_abstract = inspect.isabstract(cls)

        # Don't enforce on the abstract base itself
        if cls is StrategyBase or is_abstract:
            return

        # 1) bt_params must exist and be a dict
        if not hasattr(cls, "bt_params"):
            raise TypeError(f"{cls.__name__} must define class attribute bt_params (dict).")

        if not isinstance(cls.bt_params, dict):
            raise TypeError(f"{cls.__name__}.bt_params must be a dict of param_name -> default/REQUIRED.")

        # 2) every key in bt_params must be accepted by __init__
        sig = inspect.signature(cls.__init__)
        params = sig.parameters

        # Enforce that required args are in __init__
        for req in must_have_in_init:
            if req not in params:
                raise TypeError(f"{cls.__name__}.__init__ must accept parameter '{req}'.")

        # Enfore params are in bt_params
        for p in params.values():
            if p.name in must_have_in_init + ["self", "kwargs", "args"]:
                continue
            if p.name not in cls.bt_params:
                raise TypeError(
                    f"{cls.__name__}.__init__ has parameter '{p.name}' which is not listed in {cls.__name__}.bt_params."
                )

        # If __init__ has **kwargs, accept anything; still enforce bt_params existence/type.
        # has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        # if has_kwargs:
        #     return

        for k in cls.bt_params.keys():
            if k not in params and k not in must_have_in_init:
                raise TypeError(
                    f"{cls.__name__}.bt_params includes '{k}', but {cls.__name__}.__init__ "
                    f"does not accept '{k}' (and has no **kwargs)."
                )

    def __init__(
        self,
        data: PTDataset,
        start_trading_date: Optional[str] = None,
        ticker: Optional[str] = None,
        tplusn: Optional[int | float] = 1,
        **kwargs,
    ):
        """
        Initializes the strategy with data and parameters.
        Parameters:
        - data: PTDataset containing the market data.
        - start_trading_date: Optional start date for trading (YYYY-MM-DD).
        - kwargs: Additional parameters defined in bt_params.
        - ticker: Optional ticker symbol for the strategy.
        - tplusn: Optional time offset parameter.

        Please always call super().__init__() in subclass __init__.
        """
        # --- Core state ---
        self.data: PTDataset = data
        self.start_date = pd.Timestamp(start_trading_date) if start_trading_date else None
        self.ticker = ticker
        self.tplusn = tplusn

        self.position_info: Optional[PositionInfo] = PositionInfo()
        self.stop: Optional[float] = None
        self.indicators: Dict[str, Indicator] = {}

        # Cache index + numpy views for speed and consistent date handling
        self._df = self.data.data.copy()  # expects a DataFrame-like
        self._index = pd.DatetimeIndex(self._df.index)
        self._n = len(self._index)

        # Map: pd.Timestamp -> int (fast get_loc alternative if you want O(1) dict)
        # Note: we store Timestamps from the same DatetimeIndex to avoid dtype mismatch.
        self._dates_map: Dict[pd.Timestamp, int] = {ts: i for i, ts in enumerate(self._index)}

        # Numpy arrays (views when possible)
        self._sanitize_data()
        self._close = self._df["close"].to_numpy(copy=False)
        self._open = self._df["open"].to_numpy(copy=False)
        self._high = self._df["high"].to_numpy(copy=False)
        self._low = self._df["low"].to_numpy(copy=False)
        self._volume = self._df["volume"].to_numpy(copy=False)
        self._dates = self._index.to_numpy(copy=False)

        # Let subclass set up indicators, etc.
        self.setup()

    @property
    def position_open(self) -> bool:
        """Returns True if a position is currently open."""
        return self.position_info is not None and bool(self.position_info)

    @property
    def position_side(self) -> Optional[SideInt]:
        """Returns the side of the current position (BUY=1, SELL=-1) or None if no position."""
        return self.position_info.side if self.position_info and self.position_info.side is not None else None

    @position_side.setter
    def position_side(self, value: SideInt):
        raise AttributeError(
            "position_side is a read-only property. To change position side, use open_action() and close_action() methods which handle state updates and validations."
        )

    @position_open.setter
    def position_open(self, value: bool):
        raise AttributeError(
            "position_open is a read-only property. To change position status, use open_action() and close_action() methods which handle state updates and validations."
        )

    def additional_on_entry_info(self, *, date: pd.Timestamp = None, index: Optional[int] = None) -> Dict[str, Any]:
        """
        Override this method in your strategy subclass to provide additional information to be stored in PositionInfo when a position is opened.

        This can be useful for tracking custom state or metrics related to the position that are not covered by the default entry_date, entry_price, side, and signal_id fields.

        Returns:
            Dict[str, Any]: A dictionary of additional information to include in PositionInfo. This will be merged with the default fields when a position is opened.
        """
        return {}

    def additional_on_exit_info(self, *, date: pd.Timestamp = None, index: Optional[int] = None) -> Dict[str, Any]:
        """
        Override this method in your strategy subclass to provide additional information to be stored or logged when a position is closed.

        This can be useful for tracking custom state or metrics related to the position exit that are not covered by the default fields.

        Returns:
            Dict[str, Any]: A dictionary of additional information to include in logs or records when a position is closed.
        """
        return {}

    def get_bt_params(self) -> Dict[str, Any]:
        """
        Return the current backtest parameter values for this strategy instance.

        The returned mapping uses the keys declared on the subclass ``bt_params``
        contract and resolves each value from ``self`` via ``getattr`` so the result
        reflects the current instance state. If any declared parameter is missing,
        this raises an error immediately.

        Returns:
            Dict[str, Any]: Current backtest parameter values keyed by ``bt_params``.

        Raises:
            TypeError: If ``bt_params`` is not a dictionary.
            AttributeError: If any declared backtest parameter is missing on the instance.
        """
        bt_params = getattr(type(self), "bt_params", None)
        if not isinstance(bt_params, dict):
            raise TypeError(f"{type(self).__name__}.bt_params must be a dict.")

        current_params: Dict[str, Any] = {}
        missing_keys: List[str] = []
        for key in bt_params.keys():
            if not hasattr(self, key):
                missing_keys.append(key)
                continue
            current_params[key] = getattr(self, key)

        if missing_keys:
            raise AttributeError(f"{type(self).__name__} is missing backtest parameter attribute(s): {missing_keys}.")

        return current_params

    def _resolve(self, *, date: pd.Timestamp = None, index: int = None) -> Tuple[int, pd.Timestamp]:
        """
        Resolve date or index into a validated (index, timestamp) tuple.

        This internal method ensures consistent handling of date/index parameters throughout
        the strategy. It validates the input and returns both the integer index and
        corresponding timestamp from the strategy's data.

        Args:
            date (pd.Timestamp, optional): Date to resolve to an index. Must exist in data.
            index (int, optional): Integer index to resolve to a timestamp. Must be in range.

        Returns:
            Tuple[int, pd.Timestamp]: (validated_index, corresponding_timestamp)

        Raises:
            TypeError: If neither date nor index is provided
            IndexError: If index is out of range
            KeyError: If date is not found in the dataset index

        Note:
            Exactly one of date or index must be provided.
        """
        if index is None and date is None:
            raise TypeError("Provide either 'date' or 'index'.")

        if index is not None:
            if index < 0 or index >= self._n:
                raise IndexError(f"index out of range: {index}")
            ts = self._index[index]
            return index, ts

        # date provided
        ts = pd.Timestamp(date)
        idx = self._dates_map.get(ts)
        if idx is None:
            # If not exact, you can decide whether to allow nearest/ffill/etc.
            raise KeyError(f"date {ts} not found in dataset index.")
        return idx, ts

    def _sanitize_data(self) -> None:
        """
        Validates and normalizes the input data.

        This internal method performs data quality checks:
        1. Ensures no missing values exist in the dataset
        2. Normalizes column names to lowercase
        3. Verifies required OHLCV columns are present

        Raises:
            ValueError: If data contains missing values
            AssertionError: If required columns (open, high, low, close, volume) are missing

        Note:
            This method is automatically called during initialization.
        """
        if self._df.isnull().values.any():
            raise ValueError("Input data contains missing values. Please clean the data before using the strategy.")

        if self._df.index.duplicated().any():
            raise ValueError("Input data contains duplicated index values. Please ensure the index is unique.")

        if self._df.duplicated().any().any():
            cols_with_dupes = self._df.columns[self._df.T.duplicated()].tolist()
            raise ValueError(
                "Input data contains duplicated rows. Please ensure all rows are unique."
                f" Duplicated columns: {cols_with_dupes}, Tick: {self.ticker}"
            )

        self._df.columns = self._df.columns.str.lower()
        assert set(self._df.columns).issuperset({"open", "high", "low", "close", "volume"}), (
            "Data must contain open, high, low, close, volume columns."
        )

    @abstractmethod
    def setup(self) -> None:
        """
        Initialize indicators and strategy-specific state.

        This method is called automatically during __init__ after data is loaded and
        validated. Use it to:
        - Calculate and add indicators using add_indicator()
        - Initialize any strategy-specific attributes
        - Perform any one-time setup calculations

        Must be implemented by subclasses.

        Example:
            def setup(self):
                sma_20 = self._df['close'].rolling(20).mean()
                self.add_indicator('SMA_20', sma_20, overlay=True, color='blue')
        """
        raise NotImplementedError("Subclasses must implement setup() method.")

    @abstractmethod
    def is_open_signal(self, *, date: pd.Timestamp = None, index: int = None) -> bool:
        """
        Determine if conditions are met to open a new position.

        This method contains the core logic for entry signals. It should return True
        when your strategy conditions indicate a buy signal.

        Must be implemented by subclasses.

        Args:
            date (pd.Timestamp, optional): Date to check for signal
            index (int, optional): Integer index to check for signal

        Returns:
            bool: True if signal to open position is present, False otherwise

        Note:
            Provide exactly one of date or index. Access data using self.close[idx],
            self.indicators, etc.

        Example:
            def is_open_signal(self, *, date=None, index=None):
                idx, _ = self._resolve(date=date, index=index)
                sma = self.get_indicator('SMA_20').values.iloc[idx]
                return self.close[idx] > sma
        """
        raise NotImplementedError("Subclasses must implement is_open_signal() method.")

    @abstractmethod
    def is_close_signal(self, *, date: pd.Timestamp = None, index: int = None) -> bool:
        """
        Determine if conditions are met to close the current position.

        This method contains the core logic for exit signals. It should return True
        when your strategy conditions indicate a sell signal.

        Must be implemented by subclasses.

        Args:
            date (pd.Timestamp, optional): Date to check for signal
            index (int, optional): Integer index to check for signal

        Returns:
            bool: True if signal to close position is present, False otherwise

        Note:
            Provide exactly one of date or index. This is only called when position_open
            is True. Access data using self.close[idx], self.stop, etc.

        Example:
            def is_close_signal(self, *, date=None, index=None):
                idx, _ = self._resolve(date=date, index=index)
                sma = self.get_indicator('SMA_20').values.iloc[idx]
                return self.close[idx] < sma or self.close[idx] < self.stop
        """
        raise NotImplementedError("Subclasses must implement is_close_signal() method.")

    @abstractmethod
    def open_action(
        self,
        *,
        signal_id: Optional[str] = None,
        entry_price: Optional[float] = None,
        side: Optional[int] = None,
        date: pd.Timestamp = None,
        index: int = None,
    ) -> None:
        """
        Execute actions when opening a new position.

        This method is called after is_open_signal() returns True and should_open()
        confirms the trade should execute. Use it to:
        - Set self.position_open = True
        - Set stop-loss levels (self.stop)
        - Record entry price or other state

        Must be implemented by subclasses.

        Args:
            date (pd.Timestamp, optional): Date of position opening
            index (int, optional): Integer index of position opening

        Note:
            Provide exactly one of date or index.

        Example:
            def open_action(self, *, date=None, index=None, signal_id=None, entry_price=None, side=None):
                idx, _ = self._resolve(date=date, index=index)
                super().open_action(
                    date=date,
                    index=index,
                    signal_id=signal_id,
                    entry_price=entry_price,
                    side=side,
                )
                self.stop = self.close[idx] * 0.95  # 5% stop-loss
        """
        assert signal_id is not None, "signal_id must be provided for open_action"
        assert side is not None, "side must be provided for open_action"
        assert entry_price is not None, "entry_price must be provided for open_action"
        _, date = self._resolve(date=date, index=index)
        self.position_info = PositionInfo(entry_date=date, entry_price=entry_price, side=side, signal_id=signal_id)

    @abstractmethod
    def close_action(self, *, date: pd.Timestamp = None, index: int = None) -> None:
        """
        Execute actions when closing the current position.

        This method is called after is_close_signal() returns True and should_close()
        confirms the trade should execute. Use it to:
        - Set self.position_open = False
        - Clear stop-loss (self.stop = None)
        - Clean up any position-specific state

        Must be implemented by subclasses.

        Args:
            date (pd.Timestamp, optional): Date of position closing
            index (int, optional): Integer index of position closing

        Note:
            Provide exactly one of date or index.

        Example:
            def close_action(self, *, date=None, index=None):
                self.position_open = False
                self.stop = None
        """
        self.remove_position_info()

    # --- NumPy-first data access ---
    @property
    def close(self) -> np.ndarray:
        """
        Get close prices as a numpy array for efficient access.

        Returns:
            np.ndarray: Array of closing prices indexed by bar position
        """
        return self._close

    @property
    def open(self) -> np.ndarray:
        """
        Get open prices as a numpy array for efficient access.

        Returns:
            np.ndarray: Array of opening prices indexed by bar position
        """
        return self._open

    @property
    def high(self) -> np.ndarray:
        """
        Get high prices as a numpy array for efficient access.

        Returns:
            np.ndarray: Array of high prices indexed by bar position
        """
        return self._high

    @property
    def low(self) -> np.ndarray:
        """
        Get low prices as a numpy array for efficient access.

        Returns:
            np.ndarray: Array of low prices indexed by bar position
        """
        return self._low

    @property
    def volume(self) -> np.ndarray:
        """
        Get trading volumes as a numpy array for efficient access.

        Returns:
            np.ndarray: Array of trading volumes indexed by bar position
        """
        return self._volume

    @property
    def dates(self) -> np.ndarray:
        """
        Get dates as a numpy array for efficient access.

        Returns:
            np.ndarray: Array of pandas Timestamps indexed by bar position
        """
        return self._dates

    def reset(self):
        """
        Reset the strategy to its initial state.

        This method can be called to clear any open positions and reset indicators
        before running a new simulation. It sets position_info to a new instance and stop to None.
        """
        self.stop = None
        self.position_info = PositionInfo()

    def reset_strategy_state(self):
        """
        Reset the strategy's position and stop-loss state.

        This method is automatically called at the end of simulate() to clean up state.
        Can also be called manually if needed to reset the strategy between simulations.

        Resets:
        - position_info: Set to a new instance of PositionInfo
        - stop: Set to None
        """
        self.stop = None
        self.position_info = PositionInfo()

    def info_on_date(self, *, date: pd.Timestamp = None, index: int = None) -> Dict[str, Any]:
        """
        Get a snapshot of strategy state and data at a specific date or index.

        Useful for debugging and analyzing strategy behavior at specific points in time.

        Args:
            date (pd.Timestamp, optional): Date to query
            index (int, optional): Integer index to query

        Returns:
            Dict[str, Any]: Dictionary containing:
                - date: The timestamp
                - close: Close price at that point
                - indicators: Dict of all indicator values
                - position_open: Whether position is open
                - stop: Current stop-loss level

        Note:
            Provide exactly one of date or index.
        """
        idx, ts = self._resolve(date=date, index=index)

        # Indicators: assume Indicator.values is a pd.Series (or array-like)
        ind_snapshot = {}
        for name, ind in self.indicators.items():
            values = getattr(ind, "values", ind)
            if isinstance(values, pd.Series):
                ind_snapshot[name] = values.iloc[idx]
            else:
                # fallback if it's numpy array-like
                ind_snapshot[name] = values[idx]

        info = {
            "date": ts,
            "close": float(self._close[idx]),
            "is_open_signal": self.is_open_signal(date=ts),
            "is_close_signal": self.is_close_signal(date=ts),
            "position_open": self.position_open,
            "position_side": self.position_side,
            "position_info": self.position_info,
            "stop": self.stop,
        }
        info["indicators"] = ind_snapshot
        return info

    # defaults
    def should_trade(self, *, date: pd.Timestamp = None, index: int = None) -> bool:
        """
        Determine if trading is allowed at the specified date/index.

        Default implementation checks if the date is on or after start_trading_date.
        Can be overridden to implement more complex trading eligibility logic
        (e.g., market regime filters, time-of-day restrictions).

        Args:
            date (pd.Timestamp, optional): Date to check
            index (int, optional): Integer index to check

        Returns:
            bool: True if trading is allowed, False otherwise

        Note:
            Provide exactly one of date or index.
        """
        _, ts = self._resolve(date=date, index=index)
        if self.start_date is None:
            return True
        return ts >= self.start_date

    def should_open(self, *, date: pd.Timestamp = None, index: int = None) -> TradeDecision:
        """
        Determine if a position should be opened at the specified date/index.

        Default implementation combines three conditions:
        1. Trading is allowed (should_trade returns True)
        2. No position is currently open
        3. Open signal is present (is_open_signal returns True)

        Can be overridden to add additional logic (e.g., position sizing constraints,
        cooldown periods, maximum daily trades).

        Args:
            date (pd.Timestamp, optional): Date to check
            index (int, optional): Integer index to check

        Returns:
            bool: True if position should be opened, False otherwise

        Note:
            Provide exactly one of date or index.
        """
        raise NotImplementedError("Subclasses must implement should_open() method to return a TradeDecision object.")

    def should_close(self, *, date: pd.Timestamp = None, index: int = None) -> TradeDecision:
        """
        Determine if the current position should be closed at the specified date/index.

        Default implementation checks:
        1. A position is currently open
        2. Close signal is present (is_close_signal returns True)

        Can be overridden to add additional logic (e.g., time-based exits,
        profit targets).

        Args:
            date (pd.Timestamp, optional): Date to check
            index (int, optional): Integer index to check

        Returns:
            bool: True if position should be closed, False otherwise

        Note:
            Provide exactly one of date or index.
        """
        raise NotImplementedError("Subclasses must implement should_close() method to return a TradeDecision object.")

    def add_indicator(self, name: str, series: pd.Series, overlay: bool = False, color: Optional[str] = "red") -> None:
        """
        Add a technical indicator to the strategy for tracking and visualization.

        Indicators can be overlaid on the price chart or displayed in separate subplots.

        Args:
            name (str): Unique name for the indicator
            series (pd.Series): Indicator values indexed by date
            overlay (bool, optional): If True, plot on main price chart. If False,
                create separate subplot. Defaults to False.
            color (Optional[str], optional): Color for plotting. Defaults to "red".

        Example:
            sma_20 = self._df['close'].rolling(20).mean()
            self.add_indicator('SMA_20', sma_20, overlay=True, color='blue')
        """
        # assumes you have an Indicator class somewhere
        self.indicators[name] = Indicator(
            name=name, series=series, overlay=overlay, color=color, values=series.to_numpy(copy=False)
        )

    def get_indicator(self, name: str) -> Any:
        """
        Retrieve an indicator by name.

        Args:
            name (str): Name of the indicator to retrieve

        Returns:
            Indicator: The indicator object with name, values, overlay, and color attributes

        Raises:
            ValueError: If indicator with given name is not found

        Example:
            sma = self.get_indicator('SMA_20')
            current_sma = sma.values.iloc[idx]
        """
        i = self.indicators.get(name)
        if i is None:
            raise ValueError(f"Indicator '{name}' not found in strategy indicators.")
        return i

    def have_position(self) -> bool:
        """
        Check if a position is currently open.

        Returns:
            bool: True if position is open, False otherwise
        """
        return bool(self.position_info)

    def set_position_info(
        self,
        *,
        entry_date: Optional[pd.Timestamp] = None,
        entry_price: Optional[float] = None,
        side: Optional[SideInt] = None,
        signal_id: Optional[SignalID] = None,
    ) -> None:
        """
        Set the current position information.

        Args:
            entry_date (pd.Timestamp, optional): Date when the position was opened
            entry_price (float, optional): Price at which the position was opened
            side (SideInt, optional): Position side (SideInt.BUY or SideInt.SELL)
            signal_id (SignalID, optional): Identifier for the signal that triggered the position

        Note:
            This method is a convenient way to update all position info attributes at once.
            You can also set these attributes individually if needed.
        """
        signal_id = SignalID(signal_id) if signal_id is not None else None
        self.position_info = PositionInfo(
            entry_date=entry_date,
            entry_price=entry_price,
            side=side,
            signal_id=signal_id,
        )

    def remove_position_info(self) -> None:
        """
        Clear the current position information.

        This is typically called when closing a position to reset the state.
        """
        self.position_info = PositionInfo()

    def _record_close_trade(
        self,
        *,
        trades: List[Dict[str, Any]],
        ts: pd.Timestamp,
        exit_price: float,
        equity: float,
        index: int,
    ) -> None:
        """Append a close trade from live ``position_info`` and clear position state.

        Args:
            trades: Trade log mutated in place.
            ts: Execution timestamp.
            exit_price: Exit price for the close.
            equity: Equity after bar return, before state reset.
            index: Bar index passed to ``close_action``.
        """
        position_info = self.position_info
        entry_price = position_info.entry_price if position_info else None
        side = position_info.side if position_info else None

        if entry_price is not None and side is not None and float(entry_price) != 0.0:
            return_pct = ((exit_price - float(entry_price)) / float(entry_price)) * float(side)
        else:
            return_pct = 0.0

        trades.append(
            {
                "date": ts,
                "action": "close",
                "price": exit_price,
                "equity": equity,
                "return_pct": return_pct,
                "entry_price": entry_price,
                "side": side,
                "position_info": position_info,
                "signal_id": position_info.signal_id if position_info else None,
                **self.additional_on_exit_info(index=index),
            }
        )
        self.close_action(index=index)

    def simulate(
        self, finalize: bool = True, enforce_open_on_signal: bool = False, enforce_close_on_signal: bool = False
    ) -> Tuple[List[Dict[str, Any]], pd.Series]:
        """
        Run a backtest simulation of the strategy across all available data.

        The simulation iterates through each bar, applies returns for open positions,
        checks for signals, and executes trades on close prices. Equity is tracked
        throughout.

        Args:
            finalize (bool, optional): If True, close any open position at the end
                of the simulation. Defaults to True.

            enforce_open_on_signal (bool, optional): If True, when executing a scheduled open trade (t+n), re-check that the open signal is still valid at execution time. Defaults to True.
            enforce_close_on_signal (bool, optional): If True, when executing a scheduled close trade (t+n), re-check that the close signal is still valid at execution time. Defaults to False.

        Returns:
            Tuple[List[Dict[str, Any]], pd.Series]:
                - trades: List of trade dictionaries with keys:
                    - date: Trade execution date
                    - action: 'open' or 'close'
                    - price: Execution price
                    - equity: Equity at that point
                    - return_pct: Return % (close trades only)
                    - entry_price: Entry price (close trades only)
                - equity_series: Pandas Series of equity curve indexed by date

        Note:
            - Strategy state is automatically reset after simulation
            - Trades execute on close prices
            - Equity compounds bar-to-bar with ``1 + side * bar_return`` while a
              position is open from the prior bar
            - Initial equity is 1.0 (100%)
        """
        n = self._n
        close = self._close
        dates = self._index  # pd.DatetimeIndex for consistent timestamps
        tn_int = int(self.tplusn) if self.tplusn is not None else 0

        trades: List[Dict[str, Any]] = []
        equity = np.empty(n, dtype=float)
        equity[0] = 1.0

        eq = 1.0

        # pending executions: map execution_index -> list of ops ("open"/"close")
        pending: Dict[int, List[Dict[str, Any]]] = {}

        # We start from i=0; interval returns apply from i-1 -> i if in position at i-1
        for i in range(n):
            ts = dates[i]
            current_price = float(close[i])
            in_pos_prev = self.position_open

            # 1) Compound bar return when position was open at the prior bar close
            if i > 0 and in_pos_prev:
                prev_price = float(close[i - 1])
                side = self.position_side
                if prev_price != 0.0 and side is not None:
                    bar_ret = (current_price / prev_price) - 1.0
                    eq *= 1.0 + float(side) * bar_ret

            # 2) First, process any scheduled executions for this index
            for op in pending.pop(i, []):
                if op.get("action") == "open":
                    ## Optionally enforce that the signal is still valid at execution time (e.g., if tplusn > 0, the market conditions may have changed)
                    enforce_open = self.should_open(index=i).ok if enforce_open_on_signal else True

                    # only open if not already in a position
                    if not self.position_open and enforce_open:
                        self.open_action(
                            index=i, side=op.get("side"), signal_id=op.get("signal_id"), entry_price=current_price
                        )
                        trades.append(
                            {
                                "date": ts,
                                "action": "open",
                                "price": current_price,
                                "equity": eq,
                                "signal_id": op.get("signal_id"),
                                **self.additional_on_entry_info(index=i),
                            }
                        )

                elif op.get("action") == "close":
                    ## Optionally enforce that the close signal is still valid at execution time (e.g., if tplusn > 0, the market conditions may have changed)
                    enforce_close = self.should_close(index=i).ok if enforce_close_on_signal else True
                    if self.position_open and enforce_close:
                        self._record_close_trade(
                            trades=trades,
                            ts=ts,
                            exit_price=current_price,
                            equity=eq,
                            index=i,
                        )

            # 3) Check for new signals at t and schedule (or execute immediately if tn==0)
            open_decision = self.should_open(index=i)
            close_decision = self.should_close(index=i)
            if open_decision.ok:
                exec_idx = i if tn_int == 0 else min(i + tn_int, n - 1)
                if tn_int == 0:
                    # immediate execution
                    if not self.position_open:
                        self.open_action(
                            index=exec_idx,
                            side=open_decision.side,
                            signal_id=open_decision.signal_id,
                            entry_price=current_price,
                        )
                        trades.append(
                            {
                                "date": ts,
                                "action": "open",
                                "price": current_price,
                                "equity": eq,
                                "signal_id": open_decision.signal_id,
                                **self.additional_on_entry_info(index=i),
                            }
                        )
                else:
                    pending.setdefault(exec_idx, []).append(
                        {
                            "action": "open",
                            "signal_index": i,
                            "side": open_decision.side,
                            "signal_id": open_decision.signal_id,
                        }
                    )

            elif close_decision.ok:
                exec_idx = i if tn_int == 0 else min(i + tn_int, n - 1)
                if tn_int == 0:
                    if self.position_open:
                        self._record_close_trade(
                            trades=trades,
                            ts=ts,
                            exit_price=current_price,
                            equity=eq,
                            index=exec_idx,
                        )
                else:
                    pending.setdefault(exec_idx, []).append({"action": "close", "signal_index": i})

            equity[i] = eq

        # After loop, optionally finalize: if position still open close at last bar
        if finalize:
            # There may be pending executions scheduled for the last bar; they've already executed
            if self.position_open:
                current_price = float(close[-1])
                self._record_close_trade(
                    trades=trades,
                    ts=dates[-1],
                    exit_price=current_price,
                    equity=eq,
                    index=n - 1,
                )

        self.reset_strategy_state()
        equity_series = pd.Series(equity, index=dates, name="equity")

        ## Build trades into a DataFrame for easier analysis (optional)

        return trades, equity_series

    def _convert_trades_to_df(self, trades: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert the list of trade dictionaries into a pandas DataFrame for easier analysis.

        Args:
            trades (List[Dict[str, Any]]): List of trade dictionaries as returned by simulate()
        Returns:
            pd.DataFrame: DataFrame with columns corresponding to trade dictionary keys, indexed by trade date
        """
        if not trades:
            return pd.DataFrame()  # return empty DataFrame if no trades

        ## convert list of dicts into list of two sequential dicts for open/close pairs, then build DataFrame
        trade_records = []
        random_date = trades[0]["date"] if trades else None
        entry_info_keys = self.additional_on_entry_info(date=random_date).keys()
        exit_info_keys = self.additional_on_exit_info(date=random_date).keys()
        for i in range(0, len(trades), 2):
            open_trade = trades[i]
            close_trade = trades[i + 1] if i + 1 < len(trades) else None
            record = {
                "entry_date": open_trade["date"],
                "entry_price": open_trade["price"],
                "entry_equity": open_trade["equity"],
                "return_pct": None,
                "side": open_trade.get("side"),
                "signal_id": open_trade.get("signal_id"),
            }
            if close_trade and close_trade["action"] == "close":
                record.update(
                    {
                        "exit_date": close_trade["date"],
                        "exit_price": close_trade["price"],
                        "exit_equity": close_trade["equity"],
                        "return_pct": close_trade["return_pct"],
                        "side": close_trade.get("side"),
                        "position_info": close_trade.get("position_info"),
                    }
                )
            if entry_info_keys:
                record.update({f"entry_{k}": open_trade.get(k) for k in entry_info_keys})
            if exit_info_keys and close_trade:
                record.update({f"exit_{k}": close_trade.get(k) for k in exit_info_keys})
            trade_records.append(record)
        return pd.DataFrame(trade_records).set_index("entry_date")

    def plot_strategy_indicators(self, log_scale: bool = True, add_signal_marker: bool = True) -> go.Figure:
        """
        Create an interactive Plotly visualization of the strategy.

        Generates a multi-panel chart with:
        1. Main panel: Candlestick price chart with overlaid indicators
        2. Sub-panels: One for each non-overlay indicator
        3. Optional: Buy/sell signal markers on the main chart

        Args:
            log_scale (bool, optional): If True, use logarithmic scale for price axis.
                Useful for visualizing percentage moves. Defaults to True.
            add_signal_marker (bool, optional): If True, add triangle markers showing
                buy (down arrow) and sell (up arrow) signals. Defaults to True.

        Returns:
            go.Figure: Interactive Plotly figure object that can be displayed or saved

        Note:
            - Only displays data from start_trading_date onwards
            - Weekends are hidden for cleaner visualization
            - Overlay indicators appear on the main price chart
            - Non-overlay indicators get their own subplots below
            - Calling this method runs a full simulation to determine signal markers

        Example:
            fig = strategy.plot_strategy_indicators(log_scale=True)
            fig.show()
        """

        ## The plotting function should be defined here
        ## Ultimately, it will be a subplot where:
        ## 1. THe main plot is the price with indicators overlayed (only those with overlay=True)
        ## 2. Additional subplots for non-overlay indicators
        ## Therefore: subplots rows will be 1 + number of non-overlay indicators
        ## Ratio will be [4] + [1]*num_non_overlay_indicators
        ## 3. There will be `add_signal_marker` bool. Which if True, will add markers for open/close signals on the main plot

        indicators = self.indicators
        num_non_overlay = sum(1 for ind in indicators.values() if not ind.overlay)
        fig = make_subplots(
            rows=1 + num_non_overlay,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.6] + [0.2] * num_non_overlay,
        )
        df = self._df.copy()
        # df = df.loc[self.start_date :]
        ## Main plot
        fig.add_trace(
            go.Candlestick(
                x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"
            ),
            row=1,
            col=1,
        )

        if add_signal_marker:
            trades, _ = self.simulate(finalize=False)
            plotted_open = False
            plotted_close = False
            for trade in trades:
                if trade["action"] == "open":
                    showlegend = not plotted_open
                    plotted_open = True
                    fig.add_trace(
                        go.Scatter(
                            x=[trade["date"]],
                            y=[trade["price"] * 1.05],
                            mode="markers",
                            marker=dict(symbol="triangle-down", color="black", size=10),
                            name="Buy Signal",
                            showlegend=showlegend,
                            hovertemplate=f"Buy Signal<br>Date: {trade['date']}<br>Price: {trade['price']:.2f}<extra></extra>",
                        ),
                        row=1,
                        col=1,
                    )
                elif trade["action"] == "close":
                    showlegend = not plotted_close
                    plotted_close = True
                    fig.add_trace(
                        go.Scatter(
                            x=[trade["date"]],
                            y=[trade["price"] * 0.95],
                            mode="markers",
                            marker=dict(symbol="triangle-up", color="black", size=10),
                            name="Sell Signal",
                            showlegend=showlegend,
                            hovertemplate=(
                                f"Sell Signal<br>Date: {trade['date']}<br>"
                                f"Price: {trade['price']:.2f}<br>"
                                f"Return: {trade['return_pct']:.2%}<extra></extra>"
                            ),
                        ),
                        row=1,
                        col=1,
                    )
        ## Indicators
        for ind_name, indicator in self.indicators.items():
            ind_values = indicator.series[df.index]
            if indicator.overlay:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=ind_values, mode="lines", name=ind_name, line=dict(color=indicator.color or "red")
                    ),
                    row=1,
                    col=1,
                )
            else:
                row_idx = 2 + sum(
                    1
                    for ind in list(self.indicators.values())[: list(self.indicators.keys()).index(ind_name)]
                    if not ind.overlay
                )
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=ind_values, mode="lines", name=ind_name, line=dict(color=indicator.color or "red")
                    ),
                    row=row_idx,
                    col=1,
                )

        fig.update_layout(
            height=300 * (1 + num_non_overlay),
            title_text=f"Strategy Indicators for {self.__class__.__name__}",
            showlegend=True,
            width=1000,
        )
        fig.update_layout(xaxis=dict(rangeslider=dict(visible=False)))
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])  # hide weekends
        fig.update_yaxes(title_text="Price", row=1, col=1, type="log" if log_scale else "linear")

        return fig

    def plot_signals(self, log_scale: bool = True) -> go.Figure:
        """
        Create an interactive Plotly visualization of buy/sell signals on the price chart.

        Generates a candlestick chart with triangle markers indicating buy (down arrow)
        and sell (up arrow) signals based on the strategy's logic.

        Args:
            log_scale (bool, optional): If True, use logarithmic scale for price axis.
                Useful for visualizing percentage moves. Defaults to True.

        Returns:
            go.Figure: Interactive Plotly figure object that can be displayed or saved

        Note:
            - Only displays data from start_trading_date onwards
            - Weekends are hidden for cleaner visualization
            - Calling this method runs a full simulation to determine signal markers

        Example:
            fig = strategy.plot_signals(log_scale=True)
            fig.show()
        """
        open_sigs = [self.is_open_signal(index=i) for i in range(self._n)]
        close_sigs = [self.is_close_signal(index=i) for i in range(self._n)]
        combined = [1 if open_sigs[i] else -1 if close_sigs[i] else 0 for i in range(self._n)]
        signal_series = pd.Series(combined, index=self._index, name="signals")
        signal_series = signal_series.loc[self.start_date :]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=signal_series.index,
                y=signal_series.values,
                mode="lines+markers",
                name="Signals",
                line=dict(color="blue"),
                hovertemplate=("Date: %{x|%Y-%m-%d}<br>Signal: %{y}<extra></extra>"),
            )
        )
        name = self.__class__.__name__ if self.ticker is None else f"{self.ticker} - {self.__class__.__name__}"
        fig.update_layout(
            height=550,
            width=1000,
            title_text=f"Strategy Signals for {name}",
            showlegend=True,
        )
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])  # hide weekends
        return fig


class ATRTrailingStrategyBase(StrategyBase, ABC):
    """
    StrategyBase class that implements ATR trailing stop logic. Inherit from this and implement your entry/exit signals.
    You can use compute_atr_loss and update_atr_trail_long/short in your logic.
    """

    bt_params: Dict[str, Any] = {}

    def __init__(
        self,
        data: PTDataset,
        atr_period: int = 20,
        atr_factor: float = 3.0,
        trail_type: str = "m",
        average_type: str = "w",
        start_trading_date: Optional[str] = None,
        ticker: Optional[str] = None,
        is_long: bool = True,
        is_short: bool = False,
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)
        self.atr_period = atr_period
        self.atr_factor = atr_factor
        self.trail_type = trail_type
        self.average_type = average_type
        self.is_long = is_long
        self.is_short = is_short
        self.loss_series: Optional[pd.Series] = None

    def setup(self) -> None:
        # Compute the ATR-based loss series once during setup
        self.loss_series = compute_atr_loss(
            self.data.data,
            atr_period=self.atr_period,
            atr_factor=self.atr_factor,
            trail_type=self.trail_type,
            average_type=self.average_type,
        )

    def open_action(self, *, signal_id=None, entry_price=None, side=None, date=None, index=None):
        idx, _ = self._resolve(date=date, index=index)

        if self.is_long:
            self.stop = update_atr_trail_long(
                close=float(self.close[idx]),
                loss=float(self.loss_series[idx]),
                prev_trail=self.stop,
                reset=False,
            )
        elif self.is_short:
            self.stop = update_atr_trail_short(
                close=float(self.close[idx]),
                loss=float(self.loss_series[idx]),
                prev_trail=self.stop,
                reset=False,
            )

    def close_action(self, *, date=None, index=None):
        self.stop = None

    def is_close_signal(self, *, date=None, index=None) -> bool:
        if self.is_long:
            self.stop = update_atr_trail_long(
                close=float(self.close[index]),
                loss=float(self.loss_series[index]),
                prev_trail=self.stop,
                reset=False,
            )
        elif self.is_short:
            self.stop = update_atr_trail_short(
                close=float(self.close[index]),
                loss=float(self.loss_series[index]),
                prev_trail=self.stop,
                reset=False,
            )
