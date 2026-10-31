"""Hold OHLCV data, indicators, and entry rules for one strategy's ticker.

Lifted verbatim from ``_strategy.py`` so data loading, date resolution,
indicators, and ``RuleFilter`` live in one reusable place. ``_strategy.py`` re-exports
``Indicator``, ``Bar``, ``Rule`` and ``RuleFilter``, so existing imports and
pickles that name ``trade.backtester_._strategy`` keep working.

Core Classes:
    Bar: Read-only access to one bar's ticker and indicator values.
    RuleFilter: Collection of boolean rules evaluated for one bar.
    MarketDataMixin: Data load and cleanup, ``_resolve``, indicators, array views.

Core Dataclasses:
    Indicator: Registered indicator metadata and values.

Risk/Assumptions:
    ``Bar`` and ``RuleFilter`` only use ``_resolve``, ``ticker`` and
    ``indicators``, so they work on anything with this mixin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from trade.backtester_.data import PTDataset


@dataclass
class Indicator:
    """One registered indicator: its series, plot settings, and a numpy view of its values."""

    name: str
    series: pd.Series
    overlay: bool = False
    color: Optional[str] = "red"
    values: Optional[np.ndarray] = None
    ## When False, make_bt_wrapper skips self.I(...) registration so leading NaNs
    ## cannot delay backtesting.py's first Strategy.next() via indicator warmup.
    plot_in_backtester: bool = True

    def __post_init__(self) -> None:
        """Validate field types and fill ``values`` from ``series``.

        Raises:
            TypeError: If a field has the wrong type.
        """
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
        if not isinstance(self.plot_in_backtester, bool):
            raise TypeError("Indicator plot_in_backtester must be a boolean.")
        if self.values is None:
            self.values = self.series.to_numpy(copy=False)


class Bar:
    """Expose a strategy's ticker and indicator values at one resolved bar."""

    def __init__(self, strategy: "MarketDataMixin", index: int, date: pd.Timestamp) -> None:
        """Initialize a resolved strategy bar.

        Args:
            strategy: Strategy that owns the ticker and indicators.
            index: Resolved integer position in the strategy data.
            date: Timestamp corresponding to ``index``.
        """
        self._strategy = strategy
        self.index = index
        self.date = date
        self.ticker = strategy.ticker

    def ind(self, name: str) -> Any:
        """Return an indicator's value at this bar.

        Args:
            name: Registered indicator name.

        Returns:
            Indicator value at this bar.

        Raises:
            KeyError: If the strategy has no indicator registered as ``name``.
        """
        return self._strategy.indicators[name].values[self.index]


Rule = Callable[[Bar], bool]


class RuleFilter:
    """Store rules and require every rule to pass for a strategy bar."""

    def __init__(self, strategy: "MarketDataMixin") -> None:
        """Initialize an empty rule collection for a strategy.

        Args:
            strategy: Strategy whose bars the rules evaluate.
        """
        self._strategy = strategy
        self._rules: List[Rule] = []

    def add(self, rule: Rule) -> None:
        """Register a rule.

        Args:
            rule: Callable receiving a resolved ``Bar`` and returning a boolean.
        """
        self._rules.append(rule)

    def check(
        self,
        *,
        date: Optional[pd.Timestamp] = None,
        index: Optional[int] = None,
    ) -> bool:
        """Return whether every registered rule passes.

        An empty filter passes by default, preserving existing strategy behavior.

        Args:
            date: Date to resolve when ``index`` is omitted.
            index: Bar position to resolve when ``date`` is omitted.

        Returns:
            ``True`` when all registered rules pass.
        """
        resolved_index, resolved_date = self._strategy._resolve(date=date, index=index)
        bar = Bar(self._strategy, resolved_index, resolved_date)
        return all(bool(rule(bar)) for rule in self._rules)

    def first_failing(self, *, date: Optional[pd.Timestamp] = None, index: Optional[int] = None) -> Optional[Rule]:
        """Return the first rule that fails at this bar, or None when all pass.

        Same evaluation as ``check``, but says which rule blocked, for veto reasons.

        Args:
            date: Date to resolve when ``index`` is omitted.
            index: Bar position to resolve when ``date`` is omitted.
        """
        resolved_index, resolved_date = self._strategy._resolve(date=date, index=index)
        bar = Bar(self._strategy, resolved_index, resolved_date)
        return next((rule for rule in self._rules if not bool(rule(bar))), None)

    def __len__(self) -> int:
        """Return how many rules are registered."""
        return len(self._rules)


class MarketDataMixin:
    """OHLCV data, date resolution, indicators, and ``rules``, as StrategyBase has them today."""

    def _init_market_data(self, data: PTDataset) -> None:
        """Load ``data`` exactly as ``StrategyBase.__init__`` did.

        Callers set ``self.ticker`` first because ``Bar`` reads it. Rules exist
        before ``setup`` so subclasses can register them there.

        Args:
            data: Dataset with OHLCV columns (any capitalization).
        """
        self.data: PTDataset = data
        self.indicators: Dict[str, Indicator] = {}
        ## Rules exist before setup so subclasses can register them during setup.
        self.rules = RuleFilter(self)

        # Cache index + numpy views for speed and consistent date handling
        self._df = self.data.data.copy()  # expects a DataFrame-like
        self._index = pd.DatetimeIndex(self._df.index)
        self._n = len(self._index)
        ## Indexing a DatetimeIndex builds a new Timestamp on every call, and _resolve runs
        ## several times per bar, so keep plain Timestamps in a list for O(1) lookups.
        self._ts_list: List[pd.Timestamp] = list(self._index)
        self._dates_map: Dict[pd.Timestamp, int] = {ts: i for i, ts in enumerate(self._ts_list)}

        self._sanitize_data()
        self._close = self._df["close"].to_numpy(copy=False)
        self._open = self._df["open"].to_numpy(copy=False)
        self._high = self._df["high"].to_numpy(copy=False)
        self._low = self._df["low"].to_numpy(copy=False)
        self._volume = self._df["volume"].to_numpy(copy=False)
        self._dates = self._index.to_numpy(copy=False)

    def _resolve(self, *, date: pd.Timestamp = None, index: int = None) -> Tuple[int, pd.Timestamp]:
        """Resolve date or index into a validated (index, timestamp) tuple.

        Args:
            date: Date to resolve to an index. Must exist in data.
            index: Integer index to resolve to a timestamp. Must be in range.

        Returns:
            ``(validated_index, corresponding_timestamp)``.

        Raises:
            TypeError: If neither date nor index is provided.
            IndexError: If index is out of range.
            KeyError: If date is not found in the dataset index.
        """
        if index is None and date is None:
            raise TypeError("Provide either 'date' or 'index'.")

        if index is not None:
            if index < 0 or index >= self._n:
                raise IndexError(f"index out of range: {index}")
            return index, self._ts_list[index]

        ts = pd.Timestamp(date)
        idx = self._dates_map.get(ts)
        if idx is None:
            raise KeyError(f"date {ts} not found in dataset index.")
        return idx, ts

    def _sanitize_data(self) -> None:
        """Validate and normalize the input data.

        Raises:
            ValueError: If data contains missing values or duplicated index values.
            AssertionError: If required columns (open, high, low, close, volume) are missing.
        """
        if self._df.isnull().values.any():
            raise ValueError("Input data contains missing values. Please clean the data before using the strategy.")

        if self._df.index.duplicated().any():
            raise ValueError("Input data contains duplicated index values. Please ensure the index is unique.")

        self._df.columns = self._df.columns.str.lower()
        assert set(self._df.columns).issuperset({"open", "high", "low", "close", "volume"}), (
            "Data must contain open, high, low, close, volume columns."
        )

    def add_indicator(
        self,
        name: str,
        series: pd.Series,
        overlay: bool = False,
        color: Optional[str] = "red",
        plot_in_backtester: bool = True,
    ) -> None:
        """Add a technical indicator for tracking and visualization.

        Args:
            name: Unique name for the indicator.
            series: Indicator values indexed by date.
            overlay: If True, plot on the main price chart.
            color: Plot color.
            plot_in_backtester: If True, register with backtesting.py ``Strategy.I``
                when ``plot_indicators=True``. Set False for sparse or late-starting
                features so leading NaNs do not delay the first ``Strategy.next``.

        Example:
            sma_20 = self._df['close'].rolling(20).mean()
            self.add_indicator('SMA_20', sma_20, overlay=True, color='blue')
        """
        self.indicators[name] = Indicator(
            name=name,
            series=series,
            overlay=overlay,
            color=color,
            values=series.to_numpy(copy=False),
            plot_in_backtester=plot_in_backtester,
        )

    def get_indicator(self, name: str) -> Any:
        """Retrieve an indicator by name.

        Raises:
            ValueError: If indicator with given name is not found.
        """
        i = self.indicators.get(name)
        if i is None:
            raise ValueError(f"Indicator '{name}' not found in strategy indicators.")
        return i

    # --- NumPy-first data access ---
    @property
    def close(self) -> np.ndarray:
        """Return closing prices as a numpy array indexed by bar position."""
        return self._close

    @property
    def open(self) -> np.ndarray:
        """Return opening prices as a numpy array indexed by bar position."""
        return self._open

    @property
    def high(self) -> np.ndarray:
        """Return high prices as a numpy array indexed by bar position."""
        return self._high

    @property
    def low(self) -> np.ndarray:
        """Return low prices as a numpy array indexed by bar position."""
        return self._low

    @property
    def volume(self) -> np.ndarray:
        """Return trading volumes as a numpy array indexed by bar position."""
        return self._volume

    @property
    def dates(self) -> np.ndarray:
        """Return dates as a numpy array indexed by bar position."""
        return self._dates
