"""Position attribution workflows for EventDriven backtests.

Provides quantity normalization, option attribution loading, and position-level
PnL decomposition with trade-aware adjustments for fills, commissions, and
slippage across a trade lifecycle. Includes a static (no-Trade) path that
uses caller-provided per-leg market timeseries instead of portfolio RM data.

Core Dataclasses:
    QuantityTimeSeries: Daily quantity state and execution metadata.
    BacktestPositionAttribution: Attribution output for a single trade/position.
    LegTimeseries: One option leg market frame for the static path.
    StaticPosition: Static hold specification (legs, window, costs).

Core Classes:
    PositionAttributionAnalyzer: Portfolio/Trade path.
    StaticPositionAttributionAnalyzer: Static path (leg frames, no Trade).

Core Functions:
    _get_trade_quantity_time_series: Builds daily quantity and cost series.
    create_position_attribution: Loads and combines leg-level attribution.
    compute_position_attribution: Applies quantity and trade adjustments.
    compute_backtest_position_attribution: End-to-end portfolio integration.
    create_static_position_attribution: Unit attribution from leg frames.
    compute_static_position_attribution: End-to-end static path.

Processing Flow (portfolio):
    1. Build trade quantity time series from buy/sell ledgers.
    2. Load and aggregate leg-level option attribution by date.
    3. Scale greek attribution by position quantity.
    4. Apply open/close trade PnL adjustments and transaction costs.
    5. Return normalized daily attribution components.

Processing Flow (static):
    1. Build OptionPnlPayload from caller leg frames (not portfolio RM).
    2. Sum leg attribution via load_option_pnl_data (modern loader).
    3. Synthetic open/close quantity series (no Trade ledgers).
    4. Reuse compute_position_attribution for residual and open/close zeroing.
    5. Commission and slippage_pct costs on open/close qty-change days.

Usage:
    >>> analyzer = PositionAttributionAnalyzer(portfolio)
    >>> result = analyzer.analyze_trade(trade_id)
    >>> daily_attr = result.attribution
    >>>
    >>> static = StaticPositionAttributionAnalyzer(positions={pos.position_id: pos})
    >>> static_result = static.analyze(pos.position_id)
"""


from trade.helpers.helper import change_to_last_busday, to_datetime
from pandas.tseries.offsets import BDay
from typing import Callable, Dict, List, Mapping, Tuple, Union
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
    trade_id: Union[TradeID, str]
    signal_id: Union[SignalID, str]
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
            object.__setattr__(self, "commission", pd.Series(0, index=self.daily_qty.index))
        if self.slippage is None:
            object.__setattr__(self, "slippage", pd.Series(0, index=self.daily_qty.index))
        if self.trade_entry is None:
            object.__setattr__(self, "trade_entry", self.daily_qty.index.min())
        if self.trade_exit is None:
            object.__setattr__(self, "trade_exit", self.daily_qty.index.max())

        super().__post_init__()

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


# ---------------------------------------------------------------------------
# Static path: caller leg frames, no Trade, reuse compute_position_attribution
# ---------------------------------------------------------------------------

_MARKET_GREEK_COLUMNS = ("Delta", "Gamma", "Vega", "Theta", "Rho", "Volga")


def _normalize_leg_direction(direction: str) -> str:
    """Normalize long/short direction tokens to ``"L"`` or ``"S"``.

    Args:
        direction: One of ``LONG``, ``L``, ``SHORT``, ``S`` (any case).

    Returns:
        ``"L"`` for long, ``"S"`` for short.

    Raises:
        ValueError: If direction is not recognized.
    """
    token = str(direction).strip().upper()
    if token in {"L", "LONG"}:
        return "L"
    if token in {"S", "SHORT"}:
        return "S"
    raise ValueError(f"direction must be LONG/L or SHORT/S, got {direction!r}")


def _leg_direction_sign(direction: str) -> int:
    """Return +1 for long and -1 for short.

    Args:
        direction: Leg direction token.

    Returns:
        +1 for long legs, -1 for short legs.
    """
    return -1 if _normalize_leg_direction(direction) == "S" else 1


def _normalize_frame_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of frame with a sorted DatetimeIndex.

    Args:
        frame: Input DataFrame.

    Returns:
        Frame with datetime index sorted ascending.
    """
    out = frame.copy()
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def _lookup_frame_value(frame: pd.DataFrame, date: DATE_HINT, column: str) -> float:
    """Look up a numeric column value on a date from a market frame.

    Args:
        frame: Market timeseries with DatetimeIndex.
        date: Date to look up.
        column: Column name.

    Returns:
        Float value at date.

    Raises:
        KeyError: If date or column is missing.
    """
    frame = _normalize_frame_index(frame)
    ts = pd.to_datetime(date)
    ## Prefer exact label match after normalizing both sides to timestamps.
    if ts not in frame.index:
        ## Allow string-normalized calendar labels present as midnight stamps.
        day = ts.normalize()
        if day in frame.index:
            ts = day
        else:
            raise KeyError(f"date {ts.date()} not found in frame index for column {column!r}")
    if column not in frame.columns:
        raise KeyError(f"column {column!r} not found in frame")
    return float(pd.to_numeric(frame.loc[ts, column], errors="coerce"))


@dataclass(frozen=True)
class LegTimeseries:
    """One option leg market timeseries for static attribution.

    Attributes:
        opttick: Option OCC ticker for payload build / loader.
        direction: ``LONG``/``L`` or ``SHORT``/``S``.
        frame: Market option TS including Midpoint, capitalised greeks, and Vol/vol.
    """

    opttick: str
    direction: str
    frame: pd.DataFrame

    def __post_init__(self) -> None:
        """Validate leg fields and normalize direction.

        Raises:
            ValueError: If frame is empty or direction is invalid.
        """
        if self.frame is None or self.frame.empty:
            raise ValueError(f"LegTimeseries frame is empty for opttick={self.opttick!r}")
        if not self.opttick:
            raise ValueError("LegTimeseries.opttick is required")
        object.__setattr__(self, "direction", _normalize_leg_direction(self.direction))
        object.__setattr__(self, "frame", _normalize_frame_index(self.frame))

    def __repr__(self) -> str:
        """Return a short leg summary."""
        return f"LegTimeseries(opttick={self.opttick!r}, direction={self.direction})"


@dataclass(frozen=True)
class StaticPosition:
    """Static hold position built from per-leg market frames (no Trade).

    Attributes:
        position_id: Free-form position key (also used as trade_id in results).
        legs: One or more :class:`LegTimeseries` legs.
        entry_date: Open date for the synthetic hold.
        exit_date: Close date for the synthetic hold.
        quantity: Constant signed size after open. Defaults to 1.0.
        commission: Absolute dollars per unit on open/close qty changes.
            Defaults to 0.0. Applied via the same per-unit series mechanism as the
            portfolio path.
        slippage_pct: Fraction of structure mark charged as per-unit slippage on
            open/close. Defaults to 0.0. Cost uses structure mark construction.
        signal_id: Grouping label. Defaults to ``"default"``.
    """

    position_id: str
    legs: List[LegTimeseries]
    entry_date: DATE_HINT
    exit_date: DATE_HINT
    quantity: float = 1.0
    commission: float = 0.0
    slippage_pct: float = 0.0
    signal_id: str = "default"

    def __post_init__(self) -> None:
        """Validate window, legs, and cost parameters.

        Raises:
            ValueError: If legs empty, dates invert, or quantity is zero.
        """
        if not self.position_id:
            raise ValueError("StaticPosition.position_id is required")
        if not self.legs:
            raise ValueError("StaticPosition.legs must be non-empty")
        entry = pd.to_datetime(self.entry_date)
        exit_ = pd.to_datetime(self.exit_date)
        if exit_ < entry:
            raise ValueError(
                f"exit_date ({exit_.date()}) must be on or after entry_date ({entry.date()})"
            )
        if float(self.quantity) == 0.0:
            raise ValueError("StaticPosition.quantity must be non-zero")

    def __repr__(self) -> str:
        """Return a short position summary."""
        return (
            f"StaticPosition(position_id={self.position_id!r}, n_legs={len(self.legs)}, "
            f"entry={self.entry_date}, exit={self.exit_date})"
        )


def _resolve_vol_series(frame: pd.DataFrame) -> pd.Series:
    """Extract vol series from a market frame (``vol`` or ``Vol``).

    Args:
        frame: Market option DataFrame.

    Returns:
        Volatility series named ``vol``.

    Raises:
        ValueError: If neither vol column exists.
    """
    if "vol" in frame.columns:
        series = frame["vol"]
    elif "Vol" in frame.columns:
        series = frame["Vol"]
    else:
        raise ValueError("Market frame must include 'vol' or 'Vol'")
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    series = pd.to_numeric(series, errors="coerce")
    series.name = "vol"
    return series


def _payload_from_leg_frame(
    leg: LegTimeseries,
    payload_date: DATE_HINT,
) -> OptionPnlPayload:
    """Build OptionPnlPayload from a leg frame (portfolio RM substitute).

    Same column mapping as create_position_attribution's portfolio payload path.

    Args:
        leg: Leg with market frame.
        payload_date: Payload date; must equal today when calling load_option_pnl_data.

    Returns:
        OptionPnlPayload with vol, greeks, and Midpoint spot.

    Raises:
        ValueError: If required greek / Midpoint columns are missing.
    """
    frame = leg.frame
    missing = [c for c in ("Midpoint",) + _MARKET_GREEK_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"Leg frame for {leg.opttick!r} missing required columns: {missing}"
        )

    greeks = frame[list(_MARKET_GREEK_COLUMNS)].copy()
    greeks.columns = ["delta", "gamma", "vega", "theta", "rho", "volga"]
    for col in greeks.columns:
        greeks[col] = pd.to_numeric(greeks[col], errors="coerce")

    payload = OptionPnlPayload(
        opttick=str(leg.opttick),
        date=to_datetime(payload_date),
    )
    payload.vol = _resolve_vol_series(frame)
    payload.greeks = greeks
    payload.spot = pd.to_numeric(frame["Midpoint"], errors="coerce")
    payload.spot.name = "spot"
    return payload


def structure_mark_at(
    legs: List[LegTimeseries],
    date: DATE_HINT,
) -> float:
    """Compute structure mark as signed sum of leg Midpoints on a date.

    Used for synthetic exec prices and get_position_price on the static path,
    and as the level for slippage_pct cost construction.

    Args:
        legs: Position legs.
        date: Valuation date.

    Returns:
        Structure mark (premium units, same as Midpoint).

    Raises:
        KeyError: If Midpoint is missing for any leg on date.
    """
    total = 0.0
    for leg in legs:
        mid = _lookup_frame_value(leg.frame, date, "Midpoint")
        total += _leg_direction_sign(leg.direction) * mid
    return float(total)


def create_static_position_attribution(position: StaticPosition) -> pd.DataFrame:
    """Create unit multi-leg attribution from static position leg frames.

    Same structure as :func:`create_position_attribution`, but builds payloads
    from :class:`LegTimeseries` frames instead of portfolio RM data.

    Args:
        position: Static position specification.

    Returns:
        Combined unit attribution DataFrame (same shape as portfolio create).

    Raises:
        ValueError: If loader returns empty attribution for a leg.
    """
    entry_date = position.entry_date
    exit_date = position.exit_date
    entry_padding = max(
        pd.to_datetime(entry_date) - pd.Timedelta(days=3),
        to_datetime(OPTION_TIMESERIES_START_DATE),
    )
    exit_padding = to_datetime(pd.to_datetime(exit_date) + pd.Timedelta(days=3))

    attribution_frames = []
    for leg in position.legs:
        payload = _payload_from_leg_frame(leg, payload_date=exit_padding)
        loaded = load_option_pnl_data(
            yesterday=entry_padding,
            today=exit_padding,
            opttick=leg.opttick,
            payload=payload,
        )
        if loaded.attribution is None or loaded.attribution.empty:
            raise ValueError(
                f"load_option_pnl_data returned empty attribution for opttick={leg.opttick!r}"
            )
        leg_attr = loaded.attribution.copy()
        ## Short legs flip unit components (portfolio create uses direction == "S").
        if leg.direction == "S":
            numeric = leg_attr.select_dtypes(include="number").columns
            leg_attr[numeric] = leg_attr[numeric] * -1
        attribution_frames.append(leg_attr)

    combined = sum(attribution_frames)
    return combined


def _quantity_time_series_from_static(
    position: StaticPosition,
    index: pd.DatetimeIndex,
) -> QuantityTimeSeries:
    """Build synthetic QuantityTimeSeries for a static open/close hold.

    Open on first index date, close on last when multi-day (same open/close
    residual path as ledger fills). Per-unit commission is the constant
    commission param; per-unit slippage is |slippage_pct * mark| on event days.

    Args:
        position: Static position (qty, costs, legs for marks).
        index: Hold index (typically attribution dates from entry to exit).

    Returns:
        QuantityTimeSeries consumable by compute_position_attribution.

    Raises:
        ValueError: If index is empty.
    """
    if len(index) == 0:
        raise ValueError("index must be non-empty for static quantity series")

    idx = pd.DatetimeIndex(pd.to_datetime(index)).sort_values()
    idx = idx[~idx.duplicated(keep="first")]
    qty = float(position.quantity)

    quantity_change = pd.Series(0.0, index=idx, dtype=float)
    quantity_change.iloc[0] = qty
    ## Multi-day holds close on the last day so fully_closed residual path runs.
    if len(idx) > 1:
        quantity_change.iloc[-1] = -qty

    daily_qty = quantity_change.cumsum()
    exec_price = pd.Series(0.0, index=idx, dtype=float)
    commission = pd.Series(0.0, index=idx, dtype=float)
    slippage = pd.Series(0.0, index=idx, dtype=float)

    ## Event days only: populate exec + per-unit costs for open/close residual math.
    event_dates = [idx[0]]
    if len(idx) > 1:
        event_dates.append(idx[-1])

    for event in event_dates:
        mark = structure_mark_at(position.legs, event)
        exec_price.loc[event] = abs(mark)
        commission.loc[event] = abs(float(position.commission))
        slippage.loc[event] = abs(float(position.slippage_pct) * abs(mark))

    tick = position.legs[0].opttick if position.legs else str(position.position_id)
    return QuantityTimeSeries(
        tick=str(tick),
        trade_id=str(position.position_id),
        signal_id=str(position.signal_id),
        daily_qty=daily_qty,
        quantity_change=quantity_change,
        exec_price=exec_price,
        commission=commission,
        slippage=slippage,
        trade_entry=idx.min(),
        trade_exit=idx.max(),
    )


def _static_get_position_price(
    legs: List[LegTimeseries],
    _id: Union[TradeID, str],
    date: DATE_HINT,
    force: bool = False,
) -> float:
    """Price callable for compute_position_attribution on the static path.

    Args:
        legs: Position legs for mark construction.
        _id: Unused trade id (kept for signature compatibility).
        date: Valuation date.
        force: Unused; kept for signature compatibility with portfolio partial.

    Returns:
        Structure mark on date.
    """
    del _id, force
    return structure_mark_at(legs, date)


def compute_static_position_attribution(
    position: StaticPosition,
) -> BacktestPositionAttribution:
    """End-to-end static attribution for one StaticPosition.

    Builds unit attribution from leg frames, synthetic open/close quantity
    series, then reuses :func:`compute_position_attribution` so trade_pnl_adj
    and open/close zeroing match the portfolio path.

    Args:
        position: Static hold specification including position_id.

    Returns:
        BacktestPositionAttribution with same columns as the portfolio path.
    """
    unit = create_static_position_attribution(position)
    entry = pd.to_datetime(position.entry_date)
    exit_ = pd.to_datetime(position.exit_date)
    unit = unit.copy()
    unit.index = pd.to_datetime(unit.index)
    unit = unit.loc[entry:exit_]
    if unit.empty:
        raise ValueError(
            f"No attribution rows for position_id={position.position_id!r} "
            f"between {entry.date()} and {exit_.date()}"
        )

    qty_ts = _quantity_time_series_from_static(position, index=unit.index)
    get_price = partial(_static_get_position_price, legs=position.legs)
    computed = compute_position_attribution(
        trade_id=position.position_id,
        attribution=unit,
        qty_ts=qty_ts,
        get_position_price_func=get_price,
    )
    return BacktestPositionAttribution(
        trade_id=str(position.position_id),
        signal_id=str(position.signal_id),
        qty=qty_ts,
        attribution=computed,
    )


class StaticPositionAttributionAnalyzer:
    """Analyzer for static holds from per-leg market frames (no portfolio/Trade).

    Same workflow surface as :class:`PositionAttributionAnalyzer`:
    analyze / analyze_all / convert_attribution_to_df. Results are
    :class:`BacktestPositionAttribution` instances.
    """

    def __init__(self, positions: Mapping[str, StaticPosition]) -> None:
        """Initialize with a mapping of position_id -> StaticPosition.

        Args:
            positions: Non-empty mapping. Keys should match each
                ``StaticPosition.position_id`` (key is used for lookup).

        Raises:
            ValueError: If positions is empty.
            TypeError: If a value is not StaticPosition.
        """
        if not positions:
            raise ValueError("positions must be a non-empty mapping of StaticPosition")
        resolved: Dict[str, StaticPosition] = {}
        for key, value in positions.items():
            if not isinstance(value, StaticPosition):
                raise TypeError(
                    f"positions[{key!r}] must be StaticPosition, got {type(value).__name__}"
                )
            resolved[str(key)] = value
        self.positions = resolved
        self.attribution_cache: Dict[str, BacktestPositionAttribution] = {}

    def analyze(self, position_id: str, force: bool = False) -> BacktestPositionAttribution:
        """Analyze one static position.

        Args:
            position_id: Key into ``positions``.
            force: Recompute even if cached.

        Returns:
            BacktestPositionAttribution for the position.

        Raises:
            KeyError: If position_id is missing.
        """
        position_id = str(position_id)
        if position_id not in self.positions:
            raise KeyError(f"position_id {position_id!r} not found in positions")
        if position_id not in self.attribution_cache or force:
            self.attribution_cache[position_id] = compute_static_position_attribution(
                self.positions[position_id]
            )
        return self.attribution_cache[position_id]

    def analyze_all(self, force: bool = False) -> Dict[str, BacktestPositionAttribution]:
        """Analyze all positions.

        Args:
            force: Recompute even if cached.

        Returns:
            Cache mapping position_id -> BacktestPositionAttribution.
        """
        for position_id in tqdm(self.positions.keys(), desc="Analyzing static positions"):
            self.analyze(position_id, force=force)
        return self.attribution_cache

    def convert_attribution_to_df(
        self,
        groupby: str = "signal_id",
        ignore_missing: bool = False,
    ) -> pd.DataFrame:
        """Convert cached attributions to a grouped summary DataFrame.

        Same contract as :meth:`PositionAttributionAnalyzer.convert_attribution_to_df`.

        Args:
            groupby: ``signal_id``, ``trade_id``, or ``daily``.
            ignore_missing: Skip missing cache entries if True.

        Returns:
            Aggregated attribution × 100.

        Raises:
            ValueError: If cache empty or missing required positions.
            AssertionError: If groupby is invalid.
        """
        assert groupby in ["signal_id", "trade_id", "daily"], (
            "groupby must be one of 'signal_id', 'trade_id', or 'daily'"
        )
        if not self.attribution_cache:
            raise ValueError("No attributions computed yet. Please run analyze_all first.")
        if not ignore_missing:
            missing = [pid for pid in self.positions if pid not in self.attribution_cache]
            if missing:
                raise ValueError(f"Missing attributions for position_ids: {missing}")

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
        daily_df = combined_df.drop(columns=["signal_id", "trade_id"])
        return daily_df.groupby(daily_df.index).sum() * 100
