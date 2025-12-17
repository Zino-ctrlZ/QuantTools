from dataclasses import dataclass, field
from typing import Dict, Any, Type, List
import pandas as pd
from trade.backtester_._strategy import StrategyBase
from trade.backtester_.backtester_ import PTDataset


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
        total_trades = sum(len(t) for t in self.trades.values())
        return (
            f"SimulationResults("
            f"tickers={tickers}, "
            f"total_trades={total_trades})"
        )


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
            
            # Initialize strategy with data, start_date, and ticker-specific params
            self.asset_strategies[ticker] = self.strategy_class(
                data=self.data[ticker],
                start_trading_date=self.start_date,
                ticker=ticker,
                **ticker_params
            )
    
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
        return combined#.mean(axis=1)
    
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
                log_scale=log_scale,
                add_signal_marker=add_signal_marker
            )
        return figures
    
    def __repr__(self) -> str:
        """String representation showing key information."""
        return (
            f"MultiAssetStrategy(name='{self.name}', "
            f"tickers={list(self.asset_strategies.keys())}, "
            f"start_date='{self.start_date}')"
        )
