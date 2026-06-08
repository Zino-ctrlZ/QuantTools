from datetime import datetime
from typing import Any, Optional, List
from pandas.tseries.offsets import BDay
import pandas as pd
from pydantic import validate_call, ConfigDict # noqa
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
from trade.helpers.decorators import log_time # noqa
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
def load_rate_payload(today: datetime, yesterday: datetime) -> SymbolPayload:
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
    rates_series = rates_series[rates_series.index.date >= yesterday.date()]
    rates_series.name = "rates"

    return SymbolPayload(symbol="RATES_USD", datetime=today, spot=rates_series)


# @log_time()
# @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def add_dod_change(payload: OptionPnlPayload, yesterday: datetime, today: datetime) -> OptionPnlPayload:
    """
    Add day-over-day change data to the option payload.
    Args:
        payload (OptionPnlPayload): The option payload to add data to.
        yesterday (datetime): The start date for the data.
        today (datetime): The end date for the data.
    Returns:
        OptionPnlPayload: The updated option payload with day-over-day change data.
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

    date_range_bool = (raw.index.date >= yesterday.date()) & (raw.index.date <= today.date())
    raw = raw[date_range_bool]

    payload.dod_change = raw

    return payload


# @log_time()
# @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def load_option_pnl_data(
    yesterday: datetime,
    today: datetime,
    *,
    opttick: str = None,
    payload: Optional[OptionPnlPayload] = None,
    **kwargs: Any,
) -> OptionPnlPayload:
    """
    Load option data for a given option data manager between yesterday and today.
    Args:
        yesterday (datetime): The start date for the data.
        today (datetime): The end date for the data.
        opttick (str): The option ticker symbol.
        payload (Optional[OptionPnlPayload]): Optional payload containing any subset
            of expected fields. Provided fields are validated; missing fields are loaded.
        **kwargs: Any additional keyword arguments.
    Returns:
        OptionPnlPayload: The loaded option data.
    """
    dm = kwargs.get("dm", None)
    if dm is None and opttick is None and payload is None:
        raise ValueError("One of 'opttick' or 'payload' must be provided.")
    if dm is not None:
        raise ValueError("Data manager input is currently not supported. Please provide 'opttick' directly.")

    payload_opttick = payload.opttick if payload is not None else None
    effective_opttick = opttick or payload_opttick
    if effective_opttick is None:
        raise ValueError("Could not resolve option ticker. Provide 'opttick' or payload.opttick.")
    if opttick is not None and payload_opttick is not None and opttick != payload_opttick:
        raise ValueError(
            "Provided 'opttick' does not match 'payload.opttick'. "
            f"Got opttick='{opttick}', payload.opttick='{payload_opttick}'."
        )

    if payload is not None and payload.date != today:
        raise ValueError(f"Provided 'payload.date' must equal 'today'. Got payload.date={payload.date}, today={today}.")

    option_meta = parse_option_tick(effective_opttick)

    ## Back up yesterday by 1BDAY to ensure inclusive data retrieval
    yesterday = max(change_to_last_busday(yesterday - BDay(1)), to_datetime(OPTION_TIMESERIES_START_DATE))
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
        trade_pnl_adjustment (Optional[pd.Series]): Optional series to account for trade PnL adjustments.
        This pnl is considering an entry into the position. It therefore adds an offset so the total pnl matches position pnl
        while excluding trade pnl from the decomposition & totaling matches DoD option pnl change.
            - Expecting index to be datetime, values to be spot.
    Returns:
        OptionPnlPayload: The updated option payload with PnL decomposition data.
    """
    greeks = payload.greeks.copy()
    dod_change = payload.dod_change.copy()

    ## This serves as moving yesterday's greeks to today. In other to align with DoD changes
    greeks["shifted_date"] = greeks.index.to_series().shift(-1)
    greeks = greeks.reset_index().set_index("shifted_date")
    delta_pnl = (dod_change["asset_spot_change"] * greeks["delta"]).dropna()
    delta_pnl.name = "delta_pnl"

    gamma_pnl = ((0.5 * dod_change["asset_spot_change"] ** 2) * greeks["gamma"]).dropna()
    gamma_pnl.name = "gamma_pnl"

    ## Vega is per 1% vol change, hence the multiplication by 100
    ## Currently using vol_change in decimal form (e.g., 0.01 for 1%)
    vega_pnl = (dod_change["vol_change"] * greeks["vega"] * 100).dropna()
    vega_pnl.name = "vega_pnl"

    theta_pnl = (dod_change.index.to_series().diff().dt.days * greeks["theta"]).dropna()
    theta_pnl.name = "theta_pnl"

    if "vanna" not in greeks.columns:
        vanna_pnl = pd.Series(
            dtype=float, data=0.0, index=vega_pnl.index
        )  # Placeholder for vanna PnL, currently set to empty series
    else:
        vanna_pnl = (dod_change["asset_spot_change"] * dod_change["vol_change"] * greeks["vanna"] * 100).dropna()
    vanna_pnl.name = "vanna_pnl"

    volga_pnl = (0.5 * ((dod_change["vol_change"] ** 2) * 100) * greeks["volga"]).dropna()
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
    all_pnl.fillna(0, inplace=True)
    all_pnl["total_pnl_excl_trade_pnl"] = all_pnl.drop(columns=["opt_dod_change", "opt_spot"]).sum(axis=1)
    all_pnl["unexplained_pnl"] = all_pnl["opt_dod_change"] - all_pnl["total_pnl_excl_trade_pnl"]
    all_pnl = trade_pnl_adjustment(attribution_table=all_pnl, entry_info=trade_pnl_entries or [])
    payload.attribution = all_pnl
    return payload
