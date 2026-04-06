"""Risk Manager Core Implementation and Orchestration.

This module implements the RiskManager class, which serves as the central orchestrator
for all risk management operations in the EventDriven backtesting framework. It
coordinates order generation, position analysis, Greek-based limits, and market
data management for options trading strategies.

Core Class:
    RiskManager: Main risk management orchestrator

Key Responsibilities:
    - Order generation from strategy signals
    - Position analysis and action recommendations
    - Greek-based risk limit enforcement
    - Dynamic position sizing
    - Market data caching and retrieval
    - Corporate action handling
    - Position rolling management

Architecture:
    The RiskManager delegates to specialized components:
        - OrderPicker: Intelligent order selection from chains
        - BacktestTimeseries: Market data management
        - PositionAnalyzer: Position evaluation via cogs
        - LimitsAndSizingCog: Risk limits and sizing

Order Generation Flow:
    1. Receive signal from strategy
    2. Load market data for signal date
    3. Apply risk limits and sizing
    4. Search option chain for suitable contracts
    5. Build order structure (single, spread, etc.)
    6. Validate order against constraints
    7. Return order or failure result

Position Analysis Flow:
    1. Load current positions from portfolio
    2. Retrieve market data for analysis date
    3. Run all enabled cogs (limits, signals, etc.)
    4. Collect action recommendations
    5. Reconcile conflicting recommendations
    6. Return final strategy changes

Risk Limit Enforcement:
    Greek Limits:
        - Delta: Directional exposure per position
        - Gamma: Convexity exposure monitoring
        - Vega: Volatility exposure tracking
        - Theta: Time decay management

    Portfolio Limits:
        - Maximum leverage constraints
        - Sector concentration limits
        - Cash usage rules
        - Position count limits

    Sizing Rules:
        - Per-position capital allocation
        - Signal-based allocation
        - Ticker-based allocation
        - Dynamic volatility adjustment

Corporate Action Handling:
    Stock Splits:
        - Trade IDs NOT updated during backtest
        - Strikes adjusted automatically in data
        - Contract quantities adjusted
        - Position continuity maintained

    Dividends:
        - Special dividend tracking
        - Regular dividend timeseries
        - Early exercise impact analysis
        - Ex-date handling

Data Processing:
    Zero Value Handling:
        - Greeks with zero values forward-filled
        - Midpoint prices forward-filled
        - Prevents calculation errors
        - Maintains data continuity

    Missing Data:
        - Forward-fill for temporary gaps
        - Error on extended missing data
        - Logging of all fill operations

Caching Strategy:
    Multiple Cache Locations:
        - Order cache: Successful order searches
        - Position cache: Position-specific data
        - Market cache: Spot, chain, Greeks
        - Persistent cache: Long-term storage
        - Temporary cache: Session-only data

    Cache Keys:
        - Composite keys prevent collisions
        - Include symbol, date, strategy params
        - Version tagging for schema changes

    TODO: Consolidate cache locations for clarity

Configuration:
    Required Config Parameters:
        - symbol_list: Trading universe
        - start_date/end_date: Backtest window
        - initial_capital: Starting capital
        - option_settings: Strategy configuration
        - risk_settings: Limit parameters

    Optional Parameters:
        - cache_settings: Cache control
        - execution_settings: Order execution
        - analysis_settings: Position analysis

Integration Points:
    Strategy Integration:
        - Receives signals from Strategy class
        - Returns orders for execution
        - Provides position analysis results

    Portfolio Integration:
        - Queries current positions
        - Submits orders for execution
        - Updates position tracking

    Data Integration:
        - Interfaces with DataManager
        - Handles multiple data sources
        - Manages data caching

Performance Optimization:
    - Aggressive caching of all data
    - Memoization of expensive calculations
    - Lazy loading of market data
    - Parallel processing where possible
    - Efficient DataFrame operations

Usage:
    # Initialize RiskManager
    rm = RiskManager(
        config=risk_manager_config,
        initial_capital=100000
    )

    # Generate order from signal
    order_result = rm.generate_order(
        signal=strategy_signal,
        current_date=analysis_date,
        portfolio_state=current_portfolio
    )

    # Analyze existing positions
    analysis_results = rm.analyze_positions(
        positions=current_positions,
        analysis_date=date,
        market_data=market_snapshot
    )

Error Handling:
    - Failed order searches logged with details
    - Position analysis errors don't crash system
    - Data retrieval failures have fallbacks
    - All errors include context for debugging

Logging:
    - Order generation attempts
    - Position analysis decisions
    - Limit breaches
    - Data cache hits/misses
    - Corporate action adjustments

Notes:
    - All dates handled as pandas Timestamps
    - Prices in dollars, Greeks in standard units
    - Quantities always integers (contracts)
    - Thread-safe via proper locking

Known Limitations:
    - Trade IDs not updated for splits (by design)
    - Cache management could be simplified
    - Limited support for exotic option strategies
    - Assumes liquid markets for execution
"""

## NOTE:
## 1) If a split happens during a backtest window, the trade id won't be updated. The dataframe will simply be uploaded with a the split adjusted strike.
## 2) All Greeks &  Midpoint with Zero values will be FFWD'ed
## 3) Do something about all these caches locations. I don't like it. It's confusing

from .utils import (
    get_timeseries_start_end,
    get_persistent_cache,
)
from EventDriven._vars import get_use_temp_cache
from trade.helpers.helper import CustomCache, is_USholiday
import numpy as np
import pandas as pd
from typing import List, Optional
from datetime import datetime
from trade.helpers.Logging import setup_logger
from EventDriven.riskmanager.picker.order_picker import OrderPicker, order_failed
from EventDriven.riskmanager.market_timeseries import BacktestTimeseries
from EventDriven.riskmanager.position.analyzer import PositionAnalyzer
from EventDriven.riskmanager.position.cogs.limits import LimitsAndSizingCog
from EventDriven.dataclasses.orders import OrderRequest
from EventDriven.dataclasses.states import NewPositionState, PositionAnalysisContext, StrategyChangeMeta
from EventDriven.types import ResultsEnum, Order
from EventDriven.configs.core import RiskManagerConfig
from EventDriven._vars import CONTRACT_MULTIPLIER, load_riskmanager_cache
from pprint import pprint

logger = setup_logger("EventDriven.riskmanager.new_base", stream_log_level="WARNING")


class RiskManager:
    """
    Manages portfolio risk and executes options trading strategies with position sizing and Greek-based limits.

    The RiskManager orchestrates the entire risk management lifecycle: analyzing positions, generating orders
    based on strategy signals, enforcing risk limits, and managing position rolls. It integrates multiple
    specialized components to handle market data, position analytics, and order selection.

    Key Responsibilities:
    - Position analysis and order generation based on risk limits and signals
    - Greek-based risk management (delta, gamma, vega, theta)
    - Position rolling based on moneyness and DTE thresholds
    - Market data caching and option pricing across backtests
    - Corporate action handling (splits, dividends)

    Core Components:
        order_picker (OrderPicker): Selects optimal orders from available options based on moneyness,
            DTE, liquidity, and strategy-specific criteria. Filters option chains and ranks candidates.

        timeseries (BacktestTimeseries): Manages and caches all market data including spot prices,
            option chains, Greeks, dividends, and risk-free rates. Handles data loading and caching
            strategies for efficient backtesting.

        position_analyzer (PositionAnalyzer): Analyzes current positions against risk limits and signals
            to determine required actions (entry, exit, roll, adjust). Calculates portfolio Greeks and
            sizing requirements.

        limits_cog (LimitsAndSizingCog): Enforces risk limits and calculates position sizes based on
            available capital, leverage, and Greek constraints. Validates orders against risk parameters.

    Core Attributes:
        symbol_list (List[str]): List of tradable symbols in the universe
        start_date/end_date (str|datetime): Backtest window boundaries
        initial_capital (float): Starting portfolio capital for the backtest
        config (RiskManagerConfig): Configuration object containing all risk management parameters

    """

    def __init__(
        self,
        symbol_list: List[str],
        bkt_start: str | datetime,
        bkt_end: str | datetime,
        initial_capital: float,
        *args,
        **kwargs,
    ):
        """
        Initialize the RiskManager with core backtest parameters and component setup.

        Parameters
        ----------
        symbol_list : List[str]
            Trading symbols to manage in the portfolio
        bkt_start : str | datetime
            Backtest start date
        bkt_end : str | datetime
            Backtest end date
        initial_capital : float
            Starting portfolio capital
        **kwargs : dict, optional
            Additional configuration passed to RiskManagerConfig

        Notes
        -----
        - Initializes OrderPicker, BacktestTimeseries, PositionAnalyzer, and LimitsAndSizingCog
        - Loads configuration from RiskManagerConfig with overrides from kwargs
        - Sets up caching infrastructure for market data and position analytics
        """
        ## Backtest Window
        self.start_date = bkt_start
        self.end_date = bkt_end

        ## Configs
        self.config = RiskManagerConfig()

        ## Get start and end dates for timeseries loading
        start, end = get_timeseries_start_end()
        self.ts_start = start
        self.ts_end = end

        ## Other Core Attributes
        self.symbol_list = symbol_list

        ## Core Classes
        self.order_picker: OrderPicker = OrderPicker(start, end)
        self.market_data: BacktestTimeseries = BacktestTimeseries(_start=self.ts_start, _end=self.ts_end)
        self.position_analyzer: PositionAnalyzer = PositionAnalyzer()
        lmt_cog: LimitsAndSizingCog = LimitsAndSizingCog(underlier_list=self.symbol_list)
        self.position_analyzer.add_cog(lmt_cog)

        ## Misc Attributes
        self.initial_capital = initial_capital
        self.symbol_list = symbol_list

        ## Initialize on disk caches. These will be cleared on exit.

        ##Order Cache
        # self.order_cache = load_riskmanager_cache(target="order_cache", create_on_missing=True, clear_on_exit=True)
        self.order_cache = {}
        if len(self.order_cache.values()) > 0:
            logger.info(f"Order cache loaded with {len(self.order_cache.values())} orders")

        ##Position Analysis Cache
        self.analysis_cache = load_riskmanager_cache(
            target="position_analysis", create_on_missing=True, clear_on_exit=True
        )
        if len(self.analysis_cache.values()) > 0:
            logger.info(f"Position analysis cache loaded with {len(self.analysis_cache.values())} analyses")

        ##Order Request Cache
        self.order_request_cache = load_riskmanager_cache(
            target="order_request_cache", create_on_missing=True, clear_on_exit=True
        )
        if len(self.order_request_cache.values()) > 0:
            logger.info(f"Order request cache loaded with {len(self.order_request_cache.values())} requests")

    def clear_caches(self):
        """
        Clears all caches used by the RiskManager.
        """
        if get_use_temp_cache():
            self.market_data.options_cache.clear()
            self.market_data.position_data_cache.clear()
            get_persistent_cache().clear()  ## Ensures any caching with `.memoize` is cleared as well.
        else:
            logger.critical("USE_TEMP_CACHE set to False. Cache will not be cleared")

    def _liquity_multiplier(self, pct_ratio: float, good: Optional[float] = None, bad: Optional[float] = None) -> float:
        """
        Calculate a liquidity multiplier based on the bid-ask spread percentage.

        Parameters
        ----------
        pct_ratio : float
            The percentage ratio of the bid-ask spread to the mid price.
        good : float, optional
            The multiplier value when the spread is very good (default is 1.0).
        bad : float, optional
            The multiplier value when the spread is very bad (default is 0.5).

        Returns
        -------
        float
            A liquidity multiplier between `bad` and `good` based on the `pct_ratio`.

        Notes
        -----
        - The function uses an exponential decay to assign higher multipliers to better spreads.
        - If `good` and `bad` are not provided, it defaults to a range of [0.5, 1.0].
        - The `pct_ratio` is expected to be a positive value representing the spread as a percentage of the mid price.
        """
        if good is None:
            good = self.order_picker._scoring_config.target_spread_pct
        if bad is None:
            bad = self.order_picker._scoring_config.pct_spread_max

        if pct_ratio <= good:
            return 1.0
        elif pct_ratio >= bad:
            return 0.25
        else:
            x = (pct_ratio - good) / (bad - good)  # Normalize to [0, 1]
            decay = np.exp(-5 * x)  # Exponential decay factor
            multiplier = round(0.25 + 0.75 * decay, 4)  # Scale to [0.25, 1.0]
            return multiplier
    
    def _quantiy_liquidity_adjustment(self, quantity: int, pct_ratio: float) -> int:
        """
        Adjust the order quantity based on the liquidity of the option, as measured by the bid-ask spread percentage.

        Parameters
        ----------
        quantity : int
            The original order quantity before adjustment.
        pct_ratio : float
            The percentage ratio of the bid-ask spread to the mid price.

        Returns
        -------
        int
            The adjusted order quantity after applying the liquidity multiplier.

        Notes
        -----
        - The function calculates a liquidity multiplier using `_liquity_multiplier` and applies it to the original quantity.
        - The resulting quantity is rounded to the nearest integer and cannot be negative.
        - This adjustment helps to reduce position size for options with poor liquidity (wide spreads).
        """
        multiplier = self._liquity_multiplier(pct_ratio)
        adjusted_quantity = int(round(quantity * multiplier))
        q = max(adjusted_quantity, 1)  # Ensure quantity is at least 1
        return q

    def get_order(self, req: OrderRequest) -> NewPositionState:
        """
        Generates an order based on the provided OrderRequest dataclass and returns
        a NewPositionState after processing through the PositionAnalyzer. The order will be
        generated based on the parameters specified in the OrderRequest as well as configs
        defined in the RiskManager & OrderPicker.

        To learn more about the OrderRequest dataclass, refer to EventDriven.dataclasses.orders.OrderRequest.
        To learn more about the NewPositionState dataclass, refer to EventDriven.dataclasses.states.NewPositionState.
        To learn more about the configs used in order generation, refer to EventDriven.configs.core

        Args:
            req (OrderRequest): The order request containing parameters for order generation.
        Returns:
            NewPositionState: The state of the new position after analysis.
        """
        if is_USholiday(req.date):
            logger.info(f"Date {req.date} is a US Holiday, skipping order generation")
            return {"result": ResultsEnum.IS_HOLIDAY.value, "data": None}

        ## Investigate if tick cash is scaled
        if not req.is_tick_cash_scaled:
            req.tick_cash = (
                req.tick_cash * CONTRACT_MULTIPLIER
            )  ## Scale tick cash to account for options contract multiplier
            req.is_tick_cash_scaled = True
        ## Generate Data
        self.market_data.market_timeseries.load_timeseries(sym=req.symbol)
        at_index = self.market_data.market_timeseries.get_at_index(sym=req.symbol, index=req.date)
        spot = at_index.spot["close"]
        chain_spot = at_index.chain_spot["close"]

        ## Update request with data
        req.spot = spot
        req.chain_spot = chain_spot
        req.delta_lmt = self.position_analyzer.get_delta_limit(
            tick_cash=req.tick_cash,
            chain_spot=chain_spot,
            date=req.date,
            ticker=req.symbol
        )

        ## Get order
        print(f"Generating order for request: {req}")
        if self.config.cache_order_requests:
            logger.info(f"Caching order request for signal ID: {req.signal_id}")
            self.order_request_cache[req.signal_id] = req
        order = self.order_picker.get_order(request=req).to_dict()
        logger.info(f"Order generated: {order}")

        ## Process order

        if not order_failed(order):
            print(f"\nOrder Received:\n")
            pprint(order)
            position_id = order["data"]["trade_id"]
        else:
            print(f"\nOrder Failed: {order}\n")
            logger.info(f"Signal ID: {req.signal_id}, Unable to produce order, returning None")
            failed_order_state = NewPositionState(
                trade_id="None",
                symbol=req.symbol,
                order=Order.from_dict(order),
                request=req,
                at_time_data=None,
                undl_at_time_data=None,
                limits=None,
            )
            return failed_order_state

        ## Get position data
        logger.info(f"Retrieving position data for position ID: {position_id}")
        self.market_data.calculate_option_data(position_id=position_id, date=req.date)

        ## Update Close price
        at_time_data = self.market_data.get_at_time_position_data(position_id=position_id, date=req.date)
        if at_time_data is not None:
            close_price = at_time_data.close
            logger.info(
                f"At time data found for position ID: {position_id} on date: {req.date}, updating close price to {close_price}"
            )
            order["data"]["close"] = close_price
        else:
            logger.critical(
                f"At time data NOT found for position ID: {position_id} on date: {req.date}, close price remains unchanged"
            )

        ## Create NewPositionState and run through position_analyzer
        logger.info(f"Quantity before analysis: {order['data']['quantity']}")
        undl_at_time_data = self.market_data.market_timeseries.get_at_index(sym=req.symbol, index=req.date)
        new_pos_state = NewPositionState(
            trade_id=position_id,
            order=Order.from_dict(order),
            request=req,
            symbol=req.symbol,
            at_time_data=at_time_data,
            undl_at_time_data=undl_at_time_data,
        )
        logger.info(f"Processing new position state through position_analyzer: {new_pos_state}")
        updated_pos_state = self.position_analyzer.on_new_position(new_position_state=new_pos_state)
        logger.info(f"Quantity after analysis: {updated_pos_state.order.data['quantity']}")

        if self.config.cache_orders:
            logger.info(f"Caching order for position ID: {position_id}")
            self.order_cache[position_id] = updated_pos_state

        q = updated_pos_state.order.data["quantity"]
        if q == 0:
            logger.warning(f"Final calculated position size is 0 for order {position_id}. Order will not be placed.")
            updated_pos_state.order["result"] = ResultsEnum.POSITION_SIZE_ZERO.value

        order = updated_pos_state.order.to_dict()

        ## Update order for liquidity
        if not order_failed(order) and q > 0:
            pct_ratio = updated_pos_state.order["metrics"]["spread_pct_ratio"]
            q = self._quantiy_liquidity_adjustment(q, pct_ratio)
            updated_pos_state.order.data["quantity"] = q
            logger.info(f"Order quantity after liquidity adjustment: {q} for position ID: {position_id}")
        return updated_pos_state

    def analyze_position(self, context: PositionAnalysisContext) -> StrategyChangeMeta:
        """
        Analyze the given portfolio context using the PositionAnalyzer.

        Args:
            context (PositionAnalysisContext): The context containing portfolio information for analysis.

        Returns:
            CogActions: The actions determined by the analysis.
        """
        analysis = self.position_analyzer.analyze(context)
        if self.config.cache_position_analysis:
            logger.info(f"Caching position analysis for date: {context.date}")
            dt = pd.to_datetime(context.date).strftime("%Y-%m-%d")
            self.analysis_cache[dt] = analysis
        return analysis

    def append_option_data(
        self,
        option_id: str = None,
        position_data: pd.DataFrame = None,
        data_pack: dict | CustomCache = None,
    ):
        """
        Append option data to the processed_position_data cache.
        Parameters:
        position_id: str: ID of the position
        position_data: pd.DataFrame: DataFrame containing the position data
        data_pack: dict|CustomCache: Data pack containing the position data
        """
        if option_id:
            assert position_data is not None, "position_data must be provided if option_id is given"
            self.market_data.options_cache[option_id] = position_data

        elif isinstance(data_pack, (CustomCache, dict)):
            for k, v in data_pack.items():
                self.market_data.options_cache[k] = v

        else:
            raise ValueError("Either option_id or data_pack must be provided to append_option_data")

    def append_position_data(
        self,
        position_id: str = None,
        position_data: pd.DataFrame = None,
        data_pack: dict | CustomCache = None,
    ):
        """
        Append position data to the position_data cache.
        Parameters:
        position_id: str: ID of the position
        position_data: pd.DataFrame: DataFrame containing the position data
        data_pack: dict|CustomCache: Data pack containing the position data
        """
        if position_id:
            assert position_data is not None, "position_data must be provided if position_id is given"
            self.market_data.position_data_cache[position_id] = position_data

        elif data_pack:
            assert isinstance(data_pack, (dict, CustomCache)), "data_pack must be a dict or CustomCache"
            for k, v in data_pack.items():
                self.market_data.position_data_cache[k] = v

        else:
            raise ValueError("Either position_id or data_pack must be provided to append_position_data")
