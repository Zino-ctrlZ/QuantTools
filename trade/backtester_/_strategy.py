
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Tuple
import inspect
import pandas as pd
from trade.backtester_.backtester_ import PTDataset
from typing import List
from dataclasses import dataclass
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from ._types import Side, SideInt # noqa

@dataclass
class Indicator:
    name: str
    values: pd.Series
    overlay: bool = False
    color: Optional[str] = "red"

@dataclass
class TradeDecision:
    ok: bool
    side: int

    def __post_init__(self):
        if not isinstance(self.ok, (bool, np.bool_)):
            raise TypeError("TradeDecision.ok must be a boolean.")
        if not isinstance(self.side, int):
            raise TypeError("TradeDecision.side must be an integer.")
        
        self.ok = bool(self.ok)
            

    def __bool__(self):
        return self.ok
    
    def __repr__(self):
        return f"TradeDecision(ok={self.ok}, side={self.side})"

@dataclass
class PositionInfo:
    entry_date: Optional[pd.Timestamp] = None
    entry_price: Optional[float] = None
    side: Optional[SideInt] = None

    def __bool__(self):
        return self.entry_date is not None and self.entry_price is not None and self.side is not None
    



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

    bt_params: Dict[str, Any] = {}

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
        ]

        # Don't enforce on the abstract base itself
        if cls is StrategyBase:
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

        # If __init__ has **kwargs, accept anything; still enforce bt_params existence/type.
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if has_kwargs:
            return

        for k in cls.bt_params.keys():
            if k not in params:
                raise TypeError(
                    f"{cls.__name__}.bt_params includes '{k}', but {cls.__name__}.__init__ "
                    f"does not accept '{k}' (and has no **kwargs)."
                )
            


    def __init__(self, 
                 data: PTDataset, 
                 start_trading_date: Optional[str] = None, 
                 ticker: Optional[str] = None,
                 **kwargs):
        """
        Initializes the strategy with data and parameters.
        Parameters:
        - data: PTDataset containing the market data.
        - start_trading_date: Optional start date for trading (YYYY-MM-DD).
        - kwargs: Additional parameters defined in bt_params.

        Please always call super().__init__() in subclass __init__.
        """
        # --- Core state ---
        self.data: PTDataset = data
        self.start_date = pd.Timestamp(start_trading_date) if start_trading_date else None
        self.ticker = ticker

        self.position_open: bool = False
        self.position_side: Optional[SideInt] = SideInt.BUY
        self.position_info: PositionInfo = PositionInfo()
        self.stop: Optional[float] = None
        self.indicators: Dict[str, Any] = {}

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
            raise ValueError("Input data contains duplicated rows. Please ensure all rows are unique."
                             f" Duplicated columns: {cols_with_dupes}, Tick: {self.ticker}")

        self._df.columns = self._df.columns.str.lower()
        assert set(self._df.columns).issuperset(
            {"open", "high", "low", "close", "volume"}
        ), "Data must contain open, high, low, close, volume columns."

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
    def open_action(self, *, date: pd.Timestamp = None, index: int = None) -> None:
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
            def open_action(self, *, date=None, index=None):
                idx, _ = self._resolve(date=date, index=index)
                self.position_open = True
                self.stop = self.close[idx] * 0.95  # 5% stop-loss
        """
        raise NotImplementedError("Subclasses must implement open_action() method.")

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
        pass

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

    def reset_strategy_state(self):
        """
        Reset the strategy's position and stop-loss state.

        This method is automatically called at the end of simulate() to clean up state.
        Can also be called manually if needed to reset the strategy between simulations.

        Resets:
        - position_open: Set to False
        - stop: Set to None
        """
        self.position_open = False
        self.stop = None

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
        
        return TradeDecision(
            ok=(
                self.should_trade(date=date, index=index)
                and (not self.position_open)
                and self.is_open_signal(date=date, index=index)
            ),
            side=1
        )

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
        return TradeDecision(
            ok=(
                self.position_open
                and self.is_close_signal(date=date, index=index)
            ),
            side=-1
        )

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
        self.indicators[name] = Indicator(name=name, values=series, overlay=overlay, color=color)

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
        return self.position_open

    def simulate(self, finalize: bool = True) -> Tuple[List[Dict[str, Any]], pd.Series]:
        """
        Run a backtest simulation of the strategy across all available data.

        The simulation iterates through each bar, applies returns for open positions,
        checks for signals, and executes trades on close prices. Equity is tracked
        throughout.

        Args:
            finalize (bool, optional): If True, close any open position at the end
                of the simulation. Defaults to True.

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
            - Returns are calculated bar-to-bar when position is open
            - Initial equity is 1.0 (100%)
        """
        n = self._n
        close = self._close
        dates = self._index  # pd.DatetimeIndex for consistent timestamps

        trades = []
        equity = np.empty(n, dtype=float)
        equity[0] = 1.0
        entry_price: Optional[float] = None

        eq = 1.0

        # We start from i=0; interval returns apply from i-1 -> i if in position at i-1
        for i in range(n):
            ts = dates[i]
            current_price = float(close[i])
            in_pos_prev = self.position_open
            position_side = self.position_side

            # 1) Apply return for the interval prev->current based on prior position state
            if i > 0 and in_pos_prev:
                prev_price = float(close[i - 1])
                if prev_price != 0.0:
                    ratio = current_price / prev_price
                    eq *= ratio ** position_side  # adjust for short/long

            # 2) Decide actions using today's bar
            if self.should_open(index=i):
                self.open_action(index=i)
                entry_price = current_price
                trades.append({"date": ts, "action": "open", "price": current_price, "equity": eq})

            elif self.should_close(index=i):
                self.close_action(index=i)
                return_pct = (current_price - entry_price) / entry_price if entry_price is not None else 0.0
                return_pct *= position_side  # adjust for short/long
                trades.append(
                    {
                        "date": ts,
                        "action": "close",
                        "price": current_price,
                        "equity": eq,
                        "return_pct": return_pct,
                        "entry_price": entry_price,
                    }
                )

            equity[i] = eq

        if finalize:
            if self.position_open:
                # Close any open position at the last price
                current_price = float(close[-1])
                self.close_action(index=n - 1)
                return_pct = (current_price - entry_price) / entry_price if entry_price is not None else 0.0
                return_pct *= self.position_side  # adjust for short/long
                trades.append(
                    {
                        "date": dates[-1],
                        "action": "close",
                        "price": current_price,
                        "equity": eq,
                        "return_pct": return_pct,
                        "entry_price": entry_price,
                    }
                )

        self.reset_strategy_state()
        equity_series = pd.Series(equity, index=dates, name="equity")
        return trades, equity_series

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
        df = df.loc[self.start_date :]
        ## Main plot
        fig.add_trace(
            go.Candlestick(
                x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"
            ),
            row=1,
            col=1,
        )

        if add_signal_marker:
            trades, _ = self.simulate()
            plotted_open = False
            plotted_close = False
            for trade in trades:
                if trade["action"] == "open":
                    showlegend = not plotted_open
                    plotted_open = True
                    fig.add_trace(
                        go.Scatter(
                            x=[trade["date"]],
                            y=[trade["price"] * 1.2],
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
                            y=[trade["price"] * 0.8],
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
            ind_values = indicator.values.loc[df.index]
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
        combined = [1 if open_sigs[i]
                    else -1 if close_sigs[i]
                    else 0 for i in range(self._n)]
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
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>Signal: %{y}<extra></extra>"
                )
            )
        )
        name = self.__class__.__name__ if self.ticker is None else f"{self.ticker} - {self.__class__.__name__}"
        fig.update_layout(
            height=550,
            width=1500,
            title_text=f"Strategy Signals for {name}",
            showlegend=True,
        )
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])  # hide weekends
        return fig