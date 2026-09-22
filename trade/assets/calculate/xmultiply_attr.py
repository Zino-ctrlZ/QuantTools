"""TimeseriesDataManager-backed option P&L attribution (xMultiply model).

Loads option spot, vol, greeks, underlier, and rates via TimeseriesDataManager,
builds day-over-day changes, and decomposes option P&L into greek components.

This is the single public attribution loader. Legacy OptionDataManager (v1)
paths have been removed. Incomplete greeks are filled with 0 before totaling,
so missing greek rows no longer yield null ``total_pnl``.

Core Functions:
    load_option_pnl_data: Resolve market series and return a decomposed payload.
    calculate_pnl_decomposition: Map DoD changes through greeks into attribution.
    load_symbol_payload / load_rate_payload: Underlier and USD rates helpers.

Processing Flow:
    1. Resolve ``opttick`` from kwargs (including legacy ``dm.opttick`` shim).
    2. Load or validate vol, option spot, greeks, asset spot, and rates.
    3. Build DoD change frame (``add_dod_change``).
    4. Shift greeks forward one session and multiply through DoD changes.

Risk/Assumptions:
    ``fillna(0)`` on the attribution frame means incomplete greeks contribute
    zero rather than nullifying ``total_pnl`` (behavioral change vs legacy v1).
    Underlier spot comes from TimeseriesDataManager, not Yahoo ``retrieve_timeseries``.

Usage:
    >>> from trade.assets.calculate.xmultiply_attr import load_option_pnl_data
    >>> payload = load_option_pnl_data(yesterday=..., today=..., opttick="AAPL...")
"""

from datetime import date, datetime
from typing import Any, List, Optional, Union
from pandas.tseries.offsets import BDay
import pandas as pd
from pydantic import validate_call, ConfigDict  # noqa
from trade.assets.calculate.data_classes import SymbolPayload, OptionPnlPayload, TradePnlInfo, SYMBOL_PAYLOADS
from trade.assets.calculate.enums import AttributionModel
from trade.assets.calculate.adjustments import trade_pnl_adjustment
from trade.helpers.helper import (
    parse_option_tick,
    retrieve_timeseries,  # noqa
    change_to_last_busday,
    get_missing_dates,
    to_datetime,
)
from trade.helpers.Logging import setup_logger
from trade.helpers.decorators import log_time  # noqa
from trade.datamanager.timeseries import TimeseriesDataManager  # noqa
from trade.optionlib.config.defaults import OPTION_TIMESERIES_START_DATE

logger = setup_logger("trade.assets.calculate.xmultiply_attr")


def _is_missing_data(data: Optional[pd.Series | pd.DataFrame]) -> bool:
    """Return True if a pandas object is missing or empty."""
    return data is None or data.empty


def _validate_timeseries_dates(
    field_name: str,
    data: pd.Series | pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
) -> None:
    """Validate index dtype and business-date coverage for provided timeseries."""
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError(f"Provided '{field_name}' must have a DatetimeIndex.")
    if data.empty:
        raise ValueError(f"Provided '{field_name}' is empty.")

    missing_dates = get_missing_dates(data, _start=start_date, _end=end_date)
    if missing_dates:
        missing_str = [d.strftime("%Y-%m-%d") for d in missing_dates]
        logger.warning(f"Provided '{field_name}' is missing expected business dates: {missing_str}")


def _validate_expected_greeks_columns(greeks_data: pd.DataFrame) -> None:
    """Validate that greeks include columns required by calculate_pnl_decomposition."""
    required_greeks_cols = {"delta", "gamma", "vega", "theta", "rho", "volga"}
    missing_cols = required_greeks_cols - set(greeks_data.columns)
    if missing_cols:
        raise ValueError(f"Provided 'greeks' is missing required columns: {sorted(missing_cols)}")


def _validate_symbol_payload(
    field_name: str,
    symbol_payload: SymbolPayload,
    expected_symbol: str,
    start_date: datetime,
    end_date: datetime,
) -> None:
    """Validate symbol payload metadata and date coverage."""
    if symbol_payload.symbol != expected_symbol:
        raise ValueError(f"Provided '{field_name}.symbol' must be '{expected_symbol}', got '{symbol_payload.symbol}'.")
    _validate_timeseries_dates(
        field_name=f"{field_name}.spot",
        data=symbol_payload.spot,
        start_date=start_date,
        end_date=end_date,
    )


def get_symbol_timeseries(symbol) -> TimeseriesDataManager:
    """
    Get a timeseries data manager for a given symbol.
    Args:
        symbol (str): The asset symbol to retrieve data for.
    Returns:
        TimeseriesDataManager: The data manager for the symbol's timeseries data.
    """
    return TimeseriesDataManager(symbol=symbol)


# @log_time()
# @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def load_symbol_payload(symbol: str, today: datetime, yesterday: datetime) -> SymbolPayload:
    """
    Load symbol payload data for a given symbol between yesterday
    and today.
    Args:
        symbol (str): The asset symbol to load data for.
        today (datetime): The end date for the data.
        yesterday (datetime): The start date for the data.
    Returns:
        SymbolPayload: The loaded symbol data.
    """
    spot_series = get_symbol_timeseries(symbol).spot.get_timeseries(start_date=yesterday, end_date=today, undo_adjust=False).timeseries
    spot_series.name = "spot"

    return SymbolPayload(symbol=symbol, datetime=today, spot=spot_series)


# @log_time()
# @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def load_rate_payload(
    today: Union[datetime, date, str],
    yesterday: Union[datetime, date, str],
) -> SymbolPayload:
    """
    Load rate payload data for USD rates between yesterday
    and today.
    Args:
        today (datetime): The end date for the data.
        yesterday (datetime): The start date for the data.
    Returns:
        SymbolPayload: The loaded rate data.
    """
    rates_series: pd.Series = (
        get_symbol_timeseries("RATES_USD").rates.get_timeseries(start_date=yesterday, end_date=today).timeseries
    )
    ## Callers may pass datetime.date (no .date()); normalize via to_datetime.
    rates_series = rates_series[rates_series.index.date >= to_datetime(yesterday).date()]
    rates_series.name = "rates"

    return SymbolPayload(symbol="RATES_USD", datetime=today, spot=rates_series)


# @log_time()
# @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def add_dod_change(
    payload: OptionPnlPayload,
    yesterday: Union[datetime, date, str],
    today: Union[datetime, date, str],
) -> OptionPnlPayload:
    """Add day-over-day change data to the option payload.

    Args:
        payload: The option payload to add data to.
        yesterday: Inclusive start date for the DoD window.
        today: Inclusive end date for the DoD window.

    Returns:
        The updated option payload with day-over-day change data.
    """
    ## Create DoD DataFrame
    raw = pd.DataFrame(index=payload.spot.index)
    opt_change = payload.spot.diff()
    opt_change.name = "opt_change"
    raw = pd.concat([raw, opt_change], axis=1)

    opt_spot = payload.spot
    opt_spot.name = "opt_spot"
    raw = pd.concat([raw, opt_spot], axis=1)

    vol_change = payload.vol.diff()
    vol_change.name = "vol_change"
    raw = pd.concat([raw, vol_change], axis=1)

    rates_change = payload.rates_payload.spot.diff()
    rates_change.name = "rates_change"
    raw = pd.concat([raw, rates_change], axis=1)

    asset_change = payload.asset_payload.spot.diff()
    asset_change.name = "asset_spot_change"
    raw = pd.concat([raw, asset_change], axis=1)

    ## Callers may pass datetime.date (no .date()); normalize via to_datetime.
    yesterday_d = to_datetime(yesterday).date()
    today_d = to_datetime(today).date()
    date_range_bool = (raw.index.date >= yesterday_d) & (raw.index.date <= today_d)
    raw = raw[date_range_bool]

    payload.dod_change = raw

    return payload


# @log_time()
# @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def load_option_pnl_data(
    yesterday: Union[datetime, date, str],
    today: Union[datetime, date, str],
    *,
    opttick: str = None,
    payload: Optional[OptionPnlPayload] = None,
    **kwargs: Any,
) -> OptionPnlPayload:
    """
    Load option data between yesterday and today via TimeseriesDataManager.

    Args:
        yesterday (datetime): The start date for the data.
        today (datetime): The end date for the data.
        opttick (str): The option ticker symbol.
        payload (Optional[OptionPnlPayload]): Optional payload containing any subset
            of expected fields. Provided fields are validated; missing fields are loaded.
        **kwargs: Accepts legacy ``dm=`` objects that expose ``.opttick``; the manager
            itself is ignored and only the ticker is used.

    Returns:
        OptionPnlPayload: The loaded option data with attribution filled in.

    Raises:
        ValueError: If neither ``opttick`` nor ``payload`` can resolve a ticker.
    """
    ## Legacy callers passed OptionDataManager as dm=; only keep the ticker.
    dm = kwargs.get("dm", None)
    if dm is not None and opttick is None:
        dm_opttick = getattr(dm, "opttick", None)
        if dm_opttick is not None:
            opttick = dm_opttick

    if opttick is None and payload is None:
        raise ValueError("One of 'opttick' or 'payload' must be provided.")

    payload_opttick = payload.opttick if payload is not None else None
    effective_opttick = opttick or payload_opttick
    if effective_opttick is None:
        raise ValueError("Could not resolve option ticker. Provide 'opttick' or payload.opttick.")
    if opttick is not None and payload_opttick is not None and opttick != payload_opttick:
        raise ValueError(
            "Provided 'opttick' does not match 'payload.opttick'. "
            f"Got opttick='{opttick}', payload.opttick='{payload_opttick}'."
        )

    ## Normalize before compare / BDay math; callers may pass date or Timestamp.
    today = to_datetime(today)
    yesterday = to_datetime(yesterday)
    if payload is not None and to_datetime(payload.date).date() != today.date():
        raise ValueError(
            f"Provided 'payload.date' must equal 'today'. "
            f"Got payload.date={payload.date}, today={today}."
        )

    option_meta = parse_option_tick(effective_opttick)

    ## Back up yesterday by 1BDAY to ensure inclusive data retrieval
    yesterday = max(
        change_to_last_busday(yesterday - BDay(1)),
        to_datetime(OPTION_TIMESERIES_START_DATE),
    )
    ts = get_symbol_timeseries(option_meta["ticker"])

    provided_vol = payload.vol if payload is not None else None
    if _is_missing_data(provided_vol):
        vol_req = ts.vol.get_timeseries(
            start_date=yesterday,
            end_date=today,
            expiration=option_meta["exp_date"],
            strike=option_meta["strike"],
            right=option_meta["put_call"],
        )
        vol_data = vol_req.timeseries
        vol_data.name = "vol"
    else:
        _validate_timeseries_dates("vol", provided_vol, yesterday, today)
        vol_data = provided_vol

    provided_spot = payload.spot if payload is not None else None
    if _is_missing_data(provided_spot):
        spot_req = ts.option_spot.get_timeseries(
            start_date=yesterday,
            end_date=today,
            expiration=option_meta["exp_date"],
            strike=option_meta["strike"],
            right=option_meta["put_call"],
        )
        spot_data = spot_req.price
        spot_data.name = "spot"
    else:
        _validate_timeseries_dates("spot", provided_spot, yesterday, today)
        spot_data = provided_spot

    provided_greeks = payload.greeks if payload is not None else None
    if _is_missing_data(provided_greeks):
        greeks_req = ts.greeks.get_timeseries(
            start_date=yesterday,
            end_date=today,
            expiration=option_meta["exp_date"],
            strike=option_meta["strike"],
            right=option_meta["put_call"],
        )
        greeks_data = greeks_req.timeseries
        _validate_expected_greeks_columns(greeks_data)
    else:
        _validate_timeseries_dates("greeks", provided_greeks, yesterday, today)
        _validate_expected_greeks_columns(provided_greeks)
        greeks_data = provided_greeks

    provided_asset_payload = payload.asset_payload if payload is not None else None
    if provided_asset_payload is None or _is_missing_data(provided_asset_payload.spot):
        sym_payload = SYMBOL_PAYLOADS.get((option_meta["ticker"], yesterday, today))
        if sym_payload is None:
            sym_payload = load_symbol_payload(symbol=option_meta["ticker"], today=today, yesterday=yesterday)
            SYMBOL_PAYLOADS[(option_meta["ticker"], yesterday, today)] = sym_payload
    else:
        _validate_symbol_payload(
            field_name="asset_payload",
            symbol_payload=provided_asset_payload,
            expected_symbol=option_meta["ticker"],
            start_date=yesterday,
            end_date=today,
        )
        sym_payload = provided_asset_payload

    provided_rates_payload = payload.rates_payload if payload is not None else None
    if provided_rates_payload is None or _is_missing_data(provided_rates_payload.spot):
        rates_payload = SYMBOL_PAYLOADS.get(("RATES_USD", yesterday, today))
        if rates_payload is None:
            rates_payload = load_rate_payload(
                today=today,
                yesterday=yesterday,
            )
            SYMBOL_PAYLOADS[("RATES_USD", yesterday, today)] = rates_payload
    else:
        _validate_symbol_payload(
            field_name="rates_payload",
            symbol_payload=provided_rates_payload,
            expected_symbol="RATES_USD",
            start_date=yesterday,
            end_date=today,
        )
        rates_payload = provided_rates_payload

    payload = OptionPnlPayload(
        opttick=effective_opttick,
        date=today,
        vol=vol_data,
        spot=spot_data,
        greeks=greeks_data,
        asset_payload=sym_payload,
        rates_payload=rates_payload,
    )
    payload.attribution_model = AttributionModel.xMULTIPLY
    payload = add_dod_change(payload, yesterday, today)

    return calculate_pnl_decomposition(payload)


# @log_time()
# @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def calculate_pnl_decomposition(
    payload: OptionPnlPayload, trade_pnl_entries: Optional[List[TradePnlInfo]] = None
) -> OptionPnlPayload:
    """
    Calculate the PnL decomposition for the given option payload.
    Args:
        payload (OptionPnlPayload): The option payload to calculate PnL decomposition for.
        trade_pnl_entries (Optional[List[TradePnlInfo]]): Optional trade PnL adjustments.
            Expecting index to be datetime, values to be spot.
    Returns:
        OptionPnlPayload: The updated option payload with PnL decomposition data.
    """
    greeks = payload.greeks.copy()
    dod_change = payload.dod_change.copy()

    ## Move yesterday's greeks onto today's index so they align with DoD changes.
    greeks["shifted_date"] = greeks.index.to_series().shift(-1)
    greeks = greeks.reset_index().set_index("shifted_date")
    delta_pnl = (dod_change["asset_spot_change"] * greeks["delta"]).dropna()
    delta_pnl.name = "delta_pnl"

    gamma_pnl = ((0.5 * dod_change["asset_spot_change"] ** 2) * greeks["gamma"]).dropna()
    gamma_pnl.name = "gamma_pnl"

    ## Vega / vanna / rho are stored per 1% vol (raw/100). Volga is stored per (1% vol)^2
    ## (raw/100**2). DoD vol_change is decimal, so multiply by 100 once per vol derivative order.
    vega_pnl = (dod_change["vol_change"] * greeks["vega"] * 100).dropna()
    vega_pnl.name = "vega_pnl"

    theta_pnl = (dod_change.index.to_series().diff().dt.days * greeks["theta"]).dropna()
    theta_pnl.name = "theta_pnl"

    if "vanna" not in greeks.columns:
        ## Missing vanna contributes zero rather than nullifying total_pnl.
        vanna_pnl = pd.Series(dtype=float, data=0.0, index=vega_pnl.index)
    else:
        vanna_pnl = (
            dod_change["asset_spot_change"] * dod_change["vol_change"] * greeks["vanna"] * 100
        ).dropna()
    vanna_pnl.name = "vanna_pnl"

    ## 0.5 * (Δσ_pct)^2 * volga_stored  ==  0.5 * (Δσ_decimal)^2 * volga_raw
    volga_pnl = (0.5 * (dod_change["vol_change"] * 100) ** 2 * greeks["volga"]).dropna()
    volga_pnl.name = "volga_pnl"

    rho_pnl = (dod_change["rates_change"] * greeks["rho"] * 100).dropna()
    rho_pnl.name = "rho_pnl"

    opt_change = dod_change["opt_change"].dropna()
    opt_change.name = "opt_dod_change"

    opt_spot = dod_change["opt_spot"].dropna()
    opt_spot.name = "opt_spot"

    all_pnl: pd.DataFrame = pd.concat(
        [
            delta_pnl,
            gamma_pnl,
            vega_pnl,
            theta_pnl,
            vanna_pnl,
            rho_pnl,
            volga_pnl,
            opt_change,
            opt_spot,
        ],
        axis=1,
    )
    all_pnl.index.name = "date"
    ## Incomplete greeks become 0 so total_pnl stays numeric (vs legacy null totals)
    all_pnl.fillna(0, inplace=True)
    all_pnl["total_pnl_excl_trade_pnl"] = all_pnl.drop(columns=["opt_dod_change", "opt_spot"]).sum(axis=1)
    all_pnl["unexplained_pnl"] = all_pnl["opt_dod_change"] - all_pnl["total_pnl_excl_trade_pnl"]
    all_pnl = trade_pnl_adjustment(attribution_table=all_pnl, entry_info=trade_pnl_entries or [])
    payload.attribution = all_pnl
    return payload
