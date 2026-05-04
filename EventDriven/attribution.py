"""Position attribution workflows for EventDriven backtests.

Provides quantity normalization, option attribution loading, and position-level
PnL decomposition with trade-aware adjustments for fills, commissions, and
slippage across a trade lifecycle.

Core Dataclasses:
    QuantityTimeSeries: Daily quantity state and execution metadata.
    BacktestPositionAttribution: Attribution output for a single trade.

Core Functions:
    _get_trade_quantity_time_series: Builds daily quantity and cost series.
    create_position_attribution: Loads and combines leg-level attribution.
    compute_position_attribution: Applies quantity and trade adjustments.
    compute_backtest_position_attribution: End-to-end portfolio integration.

Processing Flow:
    1. Build trade quantity time series from buy/sell ledgers.
    2. Load and aggregate leg-level option attribution by date.
    3. Scale greek attribution by position quantity.
    4. Apply open/close trade PnL adjustments and transaction costs.
    5. Return normalized daily attribution components.

Usage:
    >>> analyzer = PositionAttributionAnalyzer(portfolio)
    >>> result = analyzer.analyze_trade(trade_id)
    >>> daily_attr = result.attribution
"""


from trade.helpers.helper import change_to_last_busday, to_datetime
from pandas.tseries.offsets import BDay
from typing import Callable, Union
import pandas as pd
from dataclasses import dataclass
from functools import partial
from trade.datamanager.utils.date import DATE_HINT
from EventDriven.types import TradeID, SignalID
from trade.helpers.helper_types import FrozenValidated
from EventDriven.trade import Trade
from trade.assets.calculate.xmultiply_attr_v2 import load_option_pnl_data, OptionPnlPayload
from trade.assets.calculate.xmultiply_attr import load_option_pnl_data as load_option_pnl_data_v1
from trade.helpers.Logging import setup_logger
from EventDriven.riskmanager.market_timeseries import BacktestTimeseries
from EventDriven.new_portfolio import OptionSignalPortfolio
from typing import Tuple, Dict
from tqdm import tqdm
from trade.optionlib.config.defaults import OPTION_TIMESERIES_START_DATE

logger = setup_logger("EventDriven.attribution")


@dataclass(frozen=True)
class QuantityTimeSeries(FrozenValidated):
    """Immutable time series of position quantity and associated trade metadata.

    Stores daily cumulative quantity, quantity changes, execution prices, commissions,
    slippage, and the trade entry/exit dates for a single trade leg.

    Attributes:
        tick: The ticker symbol of the asset.
        trade_id: Unique identifier for the trade.
        signal_id: Identifier of the signal that generated the trade.
        daily_qty: Cumulative daily quantity indexed by business day.
        quantity_change: Daily quantity change indexed by business day.
        exec_price: Execution price per unit indexed by business day.
        commission: Per-unit commission cost; defaults to a zero series.
        slippage: Per-unit slippage cost; defaults to a zero series.
        trade_entry: First fill date; defaults to the earliest date in daily_qty.
        trade_exit: Last fill date; defaults to the latest date in daily_qty.
    """

    tick: str
    trade_id: TradeID
    signal_id: SignalID
    daily_qty: pd.Series
    quantity_change: pd.Series
    exec_price: pd.Series
    commission: pd.Series = None
    slippage: pd.Series = None
    trade_entry: DATE_HINT = None
    trade_exit: DATE_HINT = None

    def __post_init__(self):
        # Ensure that daily_qty, quantity_change, and exec_price have the same index
        if not (
            self.daily_qty.index.equals(self.quantity_change.index)
            and self.daily_qty.index.equals(self.exec_price.index)
        ):
            raise ValueError("daily_qty, quantity_change, and exec_price must have the same index")

        if self.commission is None:
            self.commission = pd.Series(0, index=self.daily_qty.index)
        if self.slippage is None:
            self.slippage = pd.Series(0, index=self.daily_qty.index)
        if self.trade_entry is None:
            self.trade_entry = self.daily_qty.index.min()
        if self.trade_exit is None:
            self.trade_exit = self.daily_qty.index.max()

    def __repr__(self) -> str:
        return f"QuantityTimeSeries(tick={self.tick}, trade_id={self.trade_id})"


@dataclass(frozen=True)
class BacktestPositionAttribution(FrozenValidated):
    """Container for the computed attribution of a single backtest position.

    Attributes:
        trade_id: Unique identifier for the trade.
        signal_id: Identifier of the signal that generated the trade.
        qty: The QuantityTimeSeries used to compute the attribution.
        attribution: DataFrame of daily attribution components (see
            :func:`compute_position_attribution` for column definitions).
    """

    trade_id: Union[TradeID, str]
    signal_id: Union[SignalID, str]
    qty: QuantityTimeSeries
    attribution: pd.DataFrame

    def __repr__(self) -> str:
        return f"BacktestPositionAttribution(trade_id={self.trade_id}, signal_id={self.signal_id})"


def _get_trade_quantity_time_series(
    trade_id: str,
    trade_obj: Trade,
) -> QuantityTimeSeries:
    """Extract daily quantity and quantity change time series for a given trade.

    Args:
        trade_id: The unique identifier for the trade.
        trade_obj: The Trade object containing the buy and sell ledgers.

    Returns:
        A QuantityTimeSeries containing the daily quantity, quantity change,
        execution price, commission, and slippage time series.
    """

    ## Sample trade
    sym = trade_obj.symbol
    individual_trades = trade_obj.buy_ledger.ledger + trade_obj.sell_ledger.ledger
    individual_trades_df = pd.DataFrame(individual_trades)

    ## Monitor if this addition is correct
    individual_trades_df["quantity"] = individual_trades_df.apply(
        lambda row: (
            row["quantity"] if row["direction"] == "BUY" else -abs(row["quantity"])
        ),
        axis=1,
    )

    ## Format the individual trades DataFrame for analysis
    cols = [
        "datetime",
        "quantity",
        "price",
        "per_unit_slippage",
        "per_unit_commission",
        "per_unit_market_value",
        "direction",
    ]
    new_col = [
        "fill_ts",
        "qty_change",
        "fill_price",
        "per_unit_slippage",
        "per_unit_commission",
        "per_unit_market_value",
        "direction",
    ]
    individual_trades_df = individual_trades_df[cols]
    individual_trades_df.columns = new_col

    ## Aggregate trades table
    def _aggregate_trade_group(group):
        total_qty = group["qty_change"].sum()

        if total_qty == 0:
            weighted_fill_price = 0
        else:
            weighted_fill_price = (group["fill_price"] * group["qty_change"]).sum() / total_qty

        return pd.Series({
            "qty_change": total_qty,
            "per_unit_slippage": group["per_unit_slippage"].sum(),
            "per_unit_commission": group["per_unit_commission"].sum(),
            "per_unit_market_value": group["per_unit_market_value"].sum(),
            "direction": group["direction"].iloc[0],
            "fill_price": weighted_fill_price,
        })

    individual_trades_df = (
        individual_trades_df
        .groupby("fill_ts", group_keys=False)
        .apply(_aggregate_trade_group)
        .sort_index()
        .reset_index()
    )
    

    individual_trades_df["qty_change"] = individual_trades_df.apply(
        lambda row: row["qty_change"] if row["direction"] == "BUY" else -abs(row["qty_change"]), axis=1
    )
    trade_entry = individual_trades_df["fill_ts"].min()
    trade_exit = individual_trades_df["fill_ts"].max()



    ## Between entry and exit, extract daily quantity and quantiy change
    date_range = pd.date_range(start=trade_entry, end=trade_exit, freq="B")
    qty_frame = individual_trades_df.set_index("fill_ts").reindex(date_range).fillna(0)
    qty_frame["qty_change"] = qty_frame["qty_change"].fillna(0)
    qty_frame["cumulative_qty"] = qty_frame["qty_change"].cumsum()

    return QuantityTimeSeries(
        tick=sym,
        trade_id=trade_id,
        signal_id=trade_obj.signal_id,
        daily_qty=qty_frame["cumulative_qty"],
        quantity_change=qty_frame["qty_change"],
        ## Scale everything to per-unit
        exec_price=qty_frame["per_unit_market_value"] / 100,
        commission=abs(qty_frame["per_unit_commission"].fillna(0) / 100),
        slippage=abs(qty_frame["per_unit_slippage"].fillna(0) / 100),
        trade_entry=trade_entry,
        trade_exit=trade_exit,
    )


def create_position_attribution(
    trade_id: TradeID, 
    entry_date: DATE_HINT, 
    exit_date: DATE_HINT, 
    v1: bool = False,
    portfolio: OptionSignalPortfolio = None,
) -> pd.DataFrame:
    """Create a position attribution DataFrame for a given trade ID.

    Extracts the relevant option legs, loads market data, and calculates the
    attribution for the position over the specified date range.

    Args:
        trade_id: The TradeID for which to create the position attribution.
        entry_date: The entry date of the trade (padded back 3 days for data loading).
        exit_date: The exit date of the trade (padded forward 3 days for data loading).
        v1: If True, uses the v1 attribution loader. Defaults to False.

    Returns:
        A DataFrame containing the position attribution for the given trade ID.
    """
    def _get_payload(opttick: str) -> OptionPnlPayload:
        """Helper function to load the option PnL payload with risk data for a given option tick."""
        if v1: 
            return None
        else:
            pay_load = OptionPnlPayload(
                opttick=opttick,
                date=to_datetime(entry_date),
            )
            opt_data = portfolio.risk_manager.market_data.generate_option_data_for_trade(opttick=opttick, check_date=entry_date)
            pay_load.vol = opt_data["vol"]

            greeks = opt_data[["Delta", "Gamma", "Vega", "Theta", "Rho", "Volga"]]
            greeks.columns = ["delta", "gamma", "vega", "theta", "rho", "volga"]
            option_spot = opt_data["Midpoint"]
            pay_load.greeks = greeks
            pay_load.spot = option_spot
            return pay_load
    legs = trade_id.legs
    attribution_frames = []
    entry_padding = max(pd.to_datetime(entry_date) - pd.Timedelta(days=3), to_datetime(OPTION_TIMESERIES_START_DATE))
    exit_padding = pd.to_datetime(exit_date) + pd.Timedelta(days=3)
    for direction, opttick in legs:
        if v1:
            attribution = load_option_pnl_data_v1(yesterday=entry_padding, today=exit_padding, opttick=opttick)
        else:
            payload = _get_payload(opttick)
            payload.date = to_datetime(exit_padding)
            attribution = load_option_pnl_data(yesterday=entry_padding, today=exit_padding, opttick=opttick, payload=payload)
        if direction == "S":
            attribution.attribution *= -1
        attribution_frames.append(attribution.attribution)
    combined_attribution = sum(attribution_frames)
    return combined_attribution


def _get_position_price(market_data: BacktestTimeseries, _id: TradeID, date: DATE_HINT, force: bool = False) -> float:
    """Get the position price for a given TradeID and date from the market data.

    Args:
        market_data: The BacktestTimeseries containing the market data for the backtest.
        _id: The TradeID for which to get the position price.
        date: The date for which to get the position price.
        force: If True, forces recalculation of the position price even if cached.

    Returns:
        The position price for the given TradeID and date.
    """
    return market_data.get_at_time_position_data(position_id=_id, date=date).get_price()


def compute_position_attribution(
    trade_id: TradeID,
    attribution: pd.DataFrame,
    qty_ts: QuantityTimeSeries,
    get_position_price_func: Callable[[TradeID, DATE_HINT, bool], float],
) -> pd.DataFrame:
    """Compute position attribution adjusted for quantity changes and execution prices.

    Iterates over attribution dates, checks for quantity changes, and adjusts
    attribution components accordingly.

    Args:
        trade_id: The TradeID for which to compute the position attribution.
        attribution: The DataFrame containing the initial attribution for the position.
        qty_ts: The QuantityTimeSeries containing the daily quantity, quantity changes,
            execution prices, and costs for the position.
        get_position_price_func: Callable with signature
            ``(trade_id: TradeID, date: DATE_HINT, force: bool) -> float`` used to
            fetch the mark price for the position on a given date.

    Returns:
        DataFrame with the following columns:

        - ``opt_dod_change``: Day-over-day change in option value from the attribution data.
        - ``opt_plus_adj``: Sum of ``opt_dod_change`` and ``trade_pnl_adjustment``.
        - ``total_pnl``: Total PnL after all adjustments.
        - ``unexplained_pnl``: Residual PnL not explained by greeks or trade adjustments.
        - ``trade_pnl_adjustment``: PnL adjustment for quantity-change days; zeroed on
          full open/close to avoid double counting.
        - ``commission_cost``: Commission cost for the quantity change on that day.
        - ``slippage_cost``: Slippage cost for the quantity change on that day.
        - ``delta_pnl``, ``gamma_pnl``, ``vega_pnl``, ``theta_pnl``, ``volga_pnl``,
          ``vanna_pnl``: Greek PnL components scaled by daily quantity.
    """

    ## Extract series from qty_ts for easier access
    daily_qty = qty_ts.daily_qty
    quantity_change = qty_ts.quantity_change

    ## Exec price is per unit market value
    exec_price = qty_ts.exec_price
    attribution = attribution.copy()
    commission = qty_ts.commission
    slippage = qty_ts.slippage

    ## Ensure attribution has necessary columns, if not create them with default values
    if "commission_cost" not in attribution.columns:
        attribution["commission_cost"] = 0#commission.fillna(0)
    if "slippage_cost" not in attribution.columns:
        attribution["slippage_cost"] = 0#slippage.fillna(0)
    if "trade_pnl_adjustment" not in attribution.columns:
        attribution["trade_pnl_adjustment"] = 0.0
    if "total_pnl" not in attribution.columns:
        attribution["total_pnl"] = attribution["opt_dod_change"] * daily_qty

    def _compute_pnl_for_change(date, qty) -> Tuple[float, float, float]:
        """Compute trade PnL for an open or close event on the given date.

        Args:
            date: The date of the quantity change.
            qty: The signed quantity change (positive for open, negative for close).

        Returns:
            Tuple of ``(pnl, entry_price, close_price)``.
        """
        if qty > 0:
            # OPEN: entry is execution price on this date, close is current position price
            entry_p = abs(exec_price.loc[date])  # + slippage.loc[date] + commission.loc[date]
            close_p = get_position_price_func(_id=trade_id, date=date, force=True)
        else:
            # CLOSE: entry is previous day's position price, close is execution price on this date
            prev_date = change_to_last_busday(date - BDay(1))
            entry_p = get_position_price_func(_id=trade_id, date=prev_date, force=True)
            close_p = abs(exec_price.loc[date])  # - slippage.loc[date] - commission.loc[date]
        pnl = (close_p - entry_p) * abs(qty)
        return pnl, entry_p, close_p

    # iterate over attribution dates (stable, less overhead than iterrows)
    for date in attribution.index:
        # get quantities (use .get so missing dates default to 0)
        qty_change = quantity_change.get(date, 0)
        today_qty = daily_qty.get(date, 0)

        # scale attribution to today's quantity
        attribution.loc[date, :] = attribution.loc[date, :] * today_qty

        # if no position at all today, zero all components and continue
        if today_qty == 0 and qty_change == 0:
            attribution.loc[date, :] = 0
            continue

        # if no change in quantity on this date, nothing else to do
        if qty_change == 0:
            continue

        # there is a quantity change: compute prev qty and flags
        prev_qty = today_qty - qty_change
        fully_closed = today_qty == 0
        just_opened = prev_qty == 0

        # compute pnl for the open/close event
        trade_pnl, entry_p, close_p = _compute_pnl_for_change(date, qty_change)
        commission_cost = commission.get(date, 0) * abs(qty_change)
        slippage_cost = slippage.get(date, 0) * abs(qty_change)
        # trade_pnl -= commission_cost + slippage_cost  # Decide whether to include costs in the trade PnL or keep them separate for attribution purposes

        # if fully closed or just opened, zero other components on that date
        if fully_closed or just_opened:
            attribution.loc[date, :] = 0

        # apply adjustments
        attribution.loc[date, "trade_pnl_adjustment"] += trade_pnl
        attribution.loc[date, "commission_cost"] -= commission_cost
        attribution.loc[date, "slippage_cost"] -= slippage_cost
        attribution.loc[date, "total_pnl"] += trade_pnl - commission_cost - slippage_cost
        logger.info(
            f"Date: {date.date()}, Qty: {qty_change}, Entry: {entry_p}, Close: {close_p}, PnL: {trade_pnl}, PrevQty: {prev_qty}, Commission: {commission_cost}, Slippage: {slippage_cost}"
        )
    attribution["opt_plus_adj"] = (
        attribution["opt_dod_change"]
        + attribution["trade_pnl_adjustment"]
        + attribution["commission_cost"]
        + attribution["slippage_cost"]
    )
    attribution = attribution[
        [
            "opt_dod_change",
            "opt_plus_adj",
            "total_pnl",
            "unexplained_pnl",
            "trade_pnl_adjustment",
            "commission_cost",
            "slippage_cost",
            "delta_pnl",
            "gamma_pnl",
            "vega_pnl",
            "theta_pnl",
            "volga_pnl",
            "vanna_pnl",
            "rho_pnl",
        ]
    ]

    return attribution


def compute_backtest_position_attribution(
    portfolio: OptionSignalPortfolio,
    trade_id: TradeID,
    signal_id: SignalID,
) -> BacktestPositionAttribution:
    """Compute position attribution for a given TradeID within a backtest portfolio.

    Retrieves the necessary trade and market data, creates the initial attribution,
    and computes the adjusted position attribution.

    Args:
        portfolio: The OptionSignalPortfolio containing the trades and market data.
        trade_id: The TradeID for which to compute the position attribution.

    Returns:
        A BacktestPositionAttribution containing the adjusted position attribution
        for the given TradeID.

    Raises:
        ValueError: If trade_id is not found in portfolio.trades_map.
    """
    # Retrieve the trade object from the portfolio using the trade_id
    trade_obj: Trade = portfolio._get_trade_object(trade_id, signal_id)
    if not trade_obj:
        raise ValueError(f"TradeID {trade_id} not found in portfolio trades_map")

    # Extract quantity time series for the trade
    qty_ts = _get_trade_quantity_time_series(trade_id, trade_obj)

    # Create initial attribution for the position)
    trade_entry = qty_ts.trade_entry
    trade_exit = qty_ts.trade_exit
    attr = create_position_attribution(trade_id=trade_id, entry_date=trade_entry, exit_date=trade_exit, v1=False, portfolio=portfolio)
    attr = attr.loc[trade_entry:trade_exit]

    # Make partial function for getting position price with market data from the portfolio's risk manager
    get_price_func = partial(_get_position_price, market_data=portfolio.risk_manager.market_data, force=True)

    # Compute the adjusted position attribution based on the quantity time series and execution prices
    computed_attr = compute_position_attribution(
        trade_id=trade_id, attribution=attr, qty_ts=qty_ts, get_position_price_func=get_price_func
    )
    return BacktestPositionAttribution(
        trade_id=trade_id, signal_id=trade_obj.signal_id, qty=qty_ts, attribution=computed_attr
    )


class PositionAttributionAnalyzer:
    """Analyzes position-level attribution for all trades in a backtest portfolio.

    Computes and caches BacktestPositionAttribution for each trade, and provides
    utilities to aggregate results into DataFrames grouped by signal or trade.
    """

    def __init__(self, portfolio: OptionSignalPortfolio):
        self.portfolio = portfolio
        self.attribution_cache: Dict[Tuple[TradeID, SignalID], BacktestPositionAttribution] = {}

    def analyze_trade(self, trade_id: TradeID, signal_id: SignalID, force: bool = False) -> BacktestPositionAttribution:
        """Analyze a specific trade by computing its position attribution.

        Args:
            trade_id: The TradeID of the trade to analyze.
            signal_id: The SignalID associated with the trade.
            force: If True, forces re-computation even if the result is cached.

        Returns:
            A BacktestPositionAttribution containing the attribution analysis
            for the specified trade.
        """
        trade_key = self.portfolio._get_trade_key(trade_id, signal_id)
        if trade_key not in self.attribution_cache or force:
            self.attribution_cache[trade_key] = compute_backtest_position_attribution(
                self.portfolio, trade_id, signal_id
            )
        return self.attribution_cache[trade_key]

    def analyze_all_trades(self, force: bool = False) -> Dict[Tuple[TradeID, SignalID], BacktestPositionAttribution]:
        """Analyze all trades in the portfolio by computing their position attributions.

        Args:
            force: If True, forces re-computation even if results are cached.

        Returns:
            A dictionary mapping each (TradeID, SignalID) tuple to its BacktestPositionAttribution.
        """
        for trade_key, trade_obj in tqdm(self.portfolio.trades_map.items(), desc="Analyzing trades"):
            if trade_key not in self.attribution_cache or force:
                self.attribution_cache[trade_key] = compute_backtest_position_attribution(
                    self.portfolio, trade_obj.trade_id, trade_obj.signal_id
                )
        return self.attribution_cache

    def convert_attribution_to_df(self, groupby: str = "signal_id", ignore_missing: bool = False) -> pd.DataFrame:
        """Convert cached attributions to a grouped summary DataFrame.

        Args:
            groupby: Aggregation mode. Must be ``"signal_id"``, ``"trade_id"``,
                or ``"daily"``.
            ignore_missing: If True, skips trades without computed attributions.
                If False, raises an error for any missing trades.

        Returns:
            A DataFrame with attribution columns summed and scaled by 100, grouped
            by the specified column.

        Raises:
            ValueError: If no attributions have been computed yet.
            ValueError: If ``ignore_missing=False`` and any trades are missing attributions.
            AssertionError: If ``groupby`` is not ``"signal_id"``, ``"trade_id"``,
                or ``"daily"``.
        """
        assert groupby in ["signal_id", "trade_id", "daily"], (
            "groupby must be one of 'signal_id', 'trade_id', or 'daily'"
        )
        if not self.attribution_cache:
            raise ValueError("No attributions computed yet. Please run analyze_all_trades first.")
        if not ignore_missing:
            missing_trades = [
                trade_key for trade_key in self.portfolio.trades_map.keys() if trade_key not in self.attribution_cache
            ]
            if missing_trades:
                raise ValueError(f"Missing attributions for TradeKeys: {missing_trades}")
        records = []
        for attr in self.attribution_cache.values():
            df = attr.attribution.copy()
            df["trade_id"] = attr.trade_id
            df["signal_id"] = attr.signal_id
            records.append(df)
        combined_df = pd.concat(records)
        if groupby == "signal_id":
            return combined_df.drop(columns=["trade_id"]).groupby("signal_id").sum() * 100
        if groupby == "trade_id":
            return combined_df.drop(columns=["signal_id"]).groupby("trade_id").sum() * 100

        # Daily aggregation drops both IDs and sums across all trades/signals per date.
        daily_df = combined_df.drop(columns=["signal_id", "trade_id"])
        return daily_df.groupby(daily_df.index).sum() * 100
