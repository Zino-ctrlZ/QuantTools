from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Type, List
import pandas as pd
from trade.assets.Stock import DATE_HINT
from trade.backtester_._strategy import StrategyBase, TradeDecision
from trade.backtester_.backtester_ import PTDataset

def _setup_strategy(
    strategy_class: Type[StrategyBase],
    data: PTDataset,
    start_date: str,
    ticker: str,
    params: Dict[str, Any],
    tplusn: Optional[int] = 0
) -> StrategyBase:
    """
    Utility function to initialize a strategy instance with the given parameters.

    Args:
        strategy_class (Type[StrategyBase]): The StrategyBase subclass to instantiate
        data (PTDataset): The dataset for the strategy
        start_date (str): The starting date for the strategy in 'YYYY-MM-DD' format
        ticker (str): The ticker symbol for the strategy
        params (Dict[str, Any]): Additional parameters to pass to the strategy's __init__

    Returns:
        StrategyBase: An initialized instance of the specified strategy class
    """
    default_params = strategy_class.bt_params

    ## Add default params to params if key is missing
    for key, value in default_params.items():
        if key not in params:
            params[key] = value
    return strategy_class(
        data=data, start_trading_date=start_date, ticker=ticker, tplusn=tplusn, **params
)

@dataclass
class SimulationResults:
    """
    Container for multi-asset strategy simulation results.

    Attributes:
        trades (Dict[str, List[Dict[str, Any]]]): Dictionary mapping tickers to their list of trades
        equity (Dict[str, pd.Series]): Dictionary mapping tickers to their equity curves
    """

    trades: Dict[str, List[Dict[str, Any]]]
    equity: Dict[str, pd.Series]

    def __repr__(self) -> str:
        """String representation showing summary statistics."""
        tickers = list(self.trades.keys())
        total_trades = sum(len(t) for t in self.trades.values()) / 2 # Assuming each trade has an open and close action
        return f"SimulationResults(" f"tickers={tickers}, " f"total_trades={total_trades})"
    
@dataclass
class MultiAssetSignals:
    """
    Container for multi-asset strategy signals on a given date.

    Attributes:
        date (str): Current date in 'YYYY-MM-DD' format
        open_signals (Dict[str, TradeDecision]): Dictionary mapping tickers to their open trade decisions
        close_signals (Dict[str, TradeDecision]): Dictionary mapping tickers to their close trade decisions
    """
    
    date: str
    open_signals: Dict[str, TradeDecision]
    close_signals: Dict[str, TradeDecision]

    def __repr__(self) -> str:
        """String representation showing summary of signals."""
        return (
            f"MultiAssetSignals(date='{self.date}', "
            f"open_signals={self.open_signals}, "
            f"close_signals={self.close_signals})"
        )


## Consider making this a subclass of StrategyBase
@dataclass
class MultiAssetStrategy:
    """
    Multi-asset strategy container for managing multiple StrategyBase instances across different tickers.

    This class allows you to run the same strategy with different parameters on multiple
    assets simultaneously. Each ticker gets its own strategy instance with custom parameters.

    Attributes:
        name (str): Name of the multi-asset strategy setup
        start_date (str): Starting date for all strategies (YYYY-MM-DD format)
        params (Dict[str, Dict[str, Any]]): Dictionary mapping tickers to their strategy parameters
        strategy_class (Type[StrategyBase]): The StrategyBase subclass to instantiate for each ticker
        data (Dict[str, PTDataset]): Dictionary mapping tickers to their PTDataset instances
        strategies (Dict[str, StrategyBase]): Dictionary of initialized strategy instances (auto-populated)

    Example:
        ```python
        params = {
            "AAPL": {"band_length": 190.0, "band_deviation": 0.25},
            "NVDA": {"band_length": 190.0, "band_deviation": 0.1}
        }

        data = {
            "AAPL": PTDataset(aapl_data),
            "NVDA": PTDataset(nvda_data)
        }

        multi_strat = MultiAssetStrategy(
            name="BollingerBands_Multi",
            start_date="2020-01-01",
            params=params,
            strategy_class=BollingerBandStrategy,
            data=data
        )

        # Run all simulations
        results = multi_strat.simulate_all()

        # Get combined equity curve
        combined_equity = multi_strat.get_combined_equity()
        ```
    """

    name: str
    start_date: str
    params: Dict[str, Dict[str, Any]]
    strategy_class: Type[StrategyBase]
    data: Dict[str, PTDataset]
    asset_strategies: Dict[str, StrategyBase] = field(default_factory=dict, init=False)
    current_open_positions: Dict[str, bool] = field(default_factory=dict, init=False)
    strategy_settings: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False)
    tplusn: Optional[int] = 1

    def __post_init__(self):
        """
        Initialize individual strategy instances for each ticker.

        Creates a StrategyBase instance for each ticker using the provided data
        and parameters. Parameters are passed as kwargs to each strategy's __init__.

        Raises:
            ValueError: If data is not provided for a ticker in params
        """
        for ticker, ticker_params in self.params.items():
            if ticker not in self.data:
                raise ValueError(f"Data not provided for ticker: {ticker}")
            
            self.asset_strategies[ticker] = _setup_strategy(
                strategy_class=self.strategy_class,
                data=self.data[ticker],
                start_date=self.start_date,
                ticker=ticker,
                tplusn=self.tplusn,
                params=ticker_params
            )
            self.current_open_positions[ticker] = False
            self.strategy_settings[ticker] = ticker_params

            # default_params = self.strategy_class.bt_params

            # ## Add default params to ticker_params if key is missing
            # for key, value in default_params.items():
            #     if key not in ticker_params:
            #         ticker_params[key] = value

            # # Initialize strategy with data, start_date, and ticker-specific params
            # self.asset_strategies[ticker] = self.strategy_class(
            #     data=self.data[ticker], start_trading_date=self.start_date, ticker=ticker, **ticker_params
            # )
            # self.current_open_positions[ticker] = False
            # self.strategy_settings[ticker] = ticker_params

    def reset_strategies(self):
        """
        Reset all strategy instances to their initial state.

        This can be useful if you want to re-run simulations without creating a new MultiAssetStrategy instance.
        """
        for ticker, strategy in self.asset_strategies.items():
            strategy.reset()
            self.current_open_positions[ticker] = False

    def simulate_all(self, finalize: bool = True) -> SimulationResults:
        """
        Run simulations for all asset_strategies.

        Args:
            finalize (bool, optional): Whether to close open positions at the end. Defaults to True.

        Returns:
            SimulationResults: Container with trades and equity dictionaries for all tickers
        """
        trades_dict = {}
        equity_dict = {}

        for ticker, strategy in self.asset_strategies.items():
            trades, equity = strategy.simulate(finalize=finalize)
            trades_dict[ticker] = trades
            equity_dict[ticker] = equity

        return SimulationResults(trades=trades_dict, equity=equity_dict)

    def get_strategy(self, ticker: str) -> StrategyBase:
        """
        Get the strategy instance for a specific ticker.

        Args:
            ticker (str): Ticker symbol

        Returns:
            StrategyBase: Strategy instance for the ticker

        Raises:
            KeyError: If ticker not found in asset_strategies
        """
        if ticker not in self.asset_strategies:
            raise KeyError(f"Strategy for ticker '{ticker}' not found")
        return self.asset_strategies[ticker]

    def get_combined_equity(self) -> pd.Series:
        """
        Calculate the combined equity curve across all asset_strategies.

        Assumes equal weighting across all asset_strategies and combines their
        equity curves into a single portfolio equity curve.

        Returns:
            pd.Series: Combined equity curve indexed by date, with equal weighting

        Raises:
            ValueError: If no asset_strategies exist
        """
        results = self.simulate_all()

        if not results.equity:
            raise ValueError("No asset_strategies to combine")

        # Combine with equal weighting
        combined = pd.concat(results.equity.values(), axis=1)
        combined.columns = list(results.equity.keys())

        # Average across all asset_strategies
        return combined  # .mean(axis=1)

    def get_all_trades(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all trades from all asset_strategies.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary mapping tickers to their list of trades
        """
        results = self.simulate_all()
        return results.trades

    def plot_all_asset_strategies(self, log_scale: bool = True, add_signal_marker: bool = True) -> Dict[str, Any]:
        """
        Generate plots for all asset_strategies.

        Args:
            log_scale (bool, optional): Whether to use logarithmic scale. Defaults to True.
            add_signal_marker (bool, optional): Whether to add buy/sell markers. Defaults to True.

        Returns:
            Dict[str, go.Figure]: Dictionary mapping tickers to their Plotly figure objects
        """
        figures = {}
        for ticker, strategy in self.asset_strategies.items():
            figures[ticker] = strategy.plot_strategy_indicators(
                log_scale=log_scale, add_signal_marker=add_signal_marker
            )
        return figures

    def should_open(self, ticker: str, current_date: str) -> TradeDecision:
        """
        Check if the strategy for a given ticker signals to open a position on the current date.

        Args:
            ticker (str): Ticker symbol
            current_date (str): Current date in 'YYYY-MM-DD' format

        Returns:
            bool: True if the strategy signals to open a position, False otherwise
        """
        strategy = self.get_strategy(ticker)
        return strategy.should_open(date=current_date)

    def should_close(self, ticker: str, current_date: str) -> TradeDecision:
        """
        Check if the strategy for a given ticker signals to close a position on the current date.

        Args:
            ticker (str): Ticker symbol
            current_date (str): Current date in 'YYYY-MM-DD' format
        Returns:
            bool: True if the strategy signals to close a position, False otherwise
        """
        strategy = self.get_strategy(ticker)
        return strategy.should_close(date=current_date)
    
    def should_close_all(self, current_date: str) -> Dict[str, TradeDecision]:
        """
        Check if any of the strategies signal to close positions on the current date.

        Args:
            current_date (str): Current date in 'YYYY-MM-DD' format 
        Returns:
            Dict[str, TradeDecision]: Dictionary mapping tickers to their close decisions
        """        
        decisions = {}
        for ticker, strategy in self.asset_strategies.items():
            decisions[ticker] = strategy.should_close(date=current_date)
        return decisions
    
    def should_open_all(self, current_date: str) -> Dict[str, TradeDecision]:
        """
        Check if any of the strategies signal to open positions on the current date.

        Args:
            current_date (str): Current date in 'YYYY-MM-DD' format
        Returns:
            Dict[str, TradeDecision]: Dictionary mapping tickers to their open decisions
        """
        decisions = {}
        for ticker, strategy in self.asset_strategies.items():
            decisions[ticker] = strategy.should_open(date=current_date)
        return decisions
    
    def generate_signals_on_date(self, current_date: str, filter_actionable: bool = False) -> MultiAssetSignals:
        """
        Generate a dictionary of signals for all tickers on the current date.

        Args:
            current_date (str): Current date in 'YYYY-MM-DD' format
        Returns:
            MultiAssetSignals: Object containing open and close signals for all tickers
        """
        opens = {}
        closes = {}
        for ticker, strategy in self.asset_strategies.items():
            opens[ticker] = strategy.should_open(date=current_date)
            closes[ticker] = strategy.should_close(date=current_date)
        if filter_actionable:
            opens = {ticker: decision for ticker, decision in opens.items() if decision.ok}
            closes = {ticker: decision for ticker, decision in closes.items() if decision.ok}
        return MultiAssetSignals(date=current_date, open_signals=opens, close_signals=closes)

    
    def set_position(self, 
                     ticker: str, 
                     signal_id: str,
                     current_date: DATE_HINT,
                     side: int,
                     entry_price: Optional[float] = 0.0):
        """
        Set the position for a given ticker and signal ID.

        Args:
            ticker (str): Ticker symbol
            signal_id (str): Signal ID for the position
            have_position (bool): Whether the position is open or not
        """
        strategy = self.get_strategy(ticker)
        strategy.set_position_info(
            entry_date=current_date,
            entry_price=entry_price,
            side=side,
            signal_id=signal_id
        )
        self.current_open_positions[ticker] = True

    def open_action(self,
                    ticker: str,
                    signal_id: str,
                    current_date: DATE_HINT,
                    side: int,
                    entry_price: Optional[float] = 0.0):
        """
        Get the open action for the strategy of a given ticker on the current date.

        Args:
            ticker (str): Ticker symbol
            signal_id (str): Signal ID for the position
            current_date (DATE_HINT): Current date for the action
            side (int): Trade side (1 for long, -1 for short)
            entry_price (Optional[float]): Entry price for the position
        """        
        strategy = self.get_strategy(ticker)
        strategy.open_action(
            date=current_date,
            signal_id=signal_id,
            side=side,
            entry_price=entry_price
        )
        self.current_open_positions[ticker] = True


    
    def unset_position(self, ticker: str):
        """
        Clear the position information for a given ticker.

        This is typically called when closing a position to reset the state.

        Args:
            ticker (str): Ticker symbol
        """
        strategy = self.get_strategy(ticker)
        strategy.remove_position_info()
        self.current_open_positions[ticker] = False

    def close_action(self, ticker: str, current_date: str):
        """
        Get the close action for the strategy of a given ticker on the current date.

        Args:
            ticker (str): Ticker symbol
            current_date (str): Current date in 'YYYY-MM-DD' format
        """
        strategy = self.get_strategy(ticker)
        self.current_open_positions[ticker] = False
        return strategy.close_action(date = current_date, index = None)

    def info_on_date(self, ticker: str, current_date: str) -> Dict[str, Any]:
        """
        Get strategy info for a given ticker on the current date.

        Args:
            ticker (str): Ticker symbol
            current_date (str): Current date in 'YYYY-MM-DD' format
        Returns:
            Dict[str, Any]: Strategy info dictionary
        """
        strategy = self.get_strategy(ticker)
        return strategy.info_on_date(date=current_date)

    def __repr__(self) -> str:
        """String representation showing key information."""
        return (
            f"MultiAssetStrategy(name='{self.name}', "
            f"tickers={list(self.asset_strategies.keys())}, "
            f"start_date='{self.start_date}')"
        )
