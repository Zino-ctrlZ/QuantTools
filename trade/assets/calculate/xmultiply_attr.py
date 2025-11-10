from datetime import datetime
from typing import Optional, List
from pandas.tseries.offsets import BDay
import pandas as pd
from pydantic import validate_call, ConfigDict
from trade.assets.calculate.data_classes import (
    SymbolPayload, 
    OptionPnlPayload,
    TradePnlInfo,
    SYMBOL_PAYLOADS
)
from trade.assets.calculate.enums import AttributionModel
from trade.assets.calculate.adjustments import trade_pnl_adjustment
from trade.assets.rates import get_risk_free_rate_helper
from trade.helpers.helper import (
    retrieve_timeseries,
    change_to_last_busday
)
from trade.helpers.Logging import setup_logger
from module_test.raw_code.DataManagers.DataManagers import OptionDataManager

logger = setup_logger('trade.assets.calculate.xmultiply_attr')


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def load_symbol_payload(symbol: str,
                        today: datetime,
                        yesterday: datetime) -> SymbolPayload:
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
    
    spot_series = retrieve_timeseries(
        tick=symbol,
        start=yesterday - BDay(1),
        end=today)['close']
    spot_series.name = 'spot'

    return SymbolPayload(
        symbol=symbol,
        datetime=today,
        spot=spot_series
    )

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def load_rate_payload(today: datetime,
                      yesterday: datetime) -> SymbolPayload:
    """
    Load rate payload data for USD rates between yesterday
    and today.
    Args:
        today (datetime): The end date for the data.
        yesterday (datetime): The start date for the data.
    Returns:
        SymbolPayload: The loaded rate data.
    """
    rates_series: pd.Series = get_risk_free_rate_helper()['annualized']
    rates_series = rates_series[rates_series.index.date >= yesterday.date()]
    rates_series.name = 'rates'

    return SymbolPayload(
        symbol='RATES_USD',
        datetime=today,
        spot=rates_series
    )


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def add_dod_change(payload: OptionPnlPayload,
                   yesterday: datetime,
                   today: datetime) -> OptionPnlPayload:
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
    raw = pd.DataFrame(index = payload.spot.index)
    opt_change = payload.spot.diff()
    opt_change.name = 'opt_change'
    raw = pd.concat([raw, opt_change], axis=1)

    vol_change = payload.vol.diff()
    vol_change.name = 'vol_change'
    raw = pd.concat([raw, vol_change], axis=1)

    rates_change = payload.rates_payload.spot.diff()
    rates_change.name = 'rates_change'
    raw = pd.concat([raw, rates_change], axis=1)

    asset_change = payload.asset_payload.spot.diff()
    asset_change.name = 'asset_spot_change'
    raw = pd.concat([raw, asset_change], axis=1)

    date_range_bool = (raw.index.date >= yesterday.date()) & (raw.index.date <= today.date())
    raw = raw[date_range_bool]

    payload.dod_change = raw

    return payload


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def load_option_pnl_data(yesterday: datetime, 
                     today: datetime, 
                     *,
                     dm: OptionDataManager=None,
                     opttick: str=None) -> OptionPnlPayload:
    
    """
    Load option data for a given option data manager between yesterday and today.
    Args:
        dm (OptionDataManager): The option data manager to load data from.
        yesterday (datetime): The start date for the data.
        today (datetime): The end date for the data.
    Returns:
        OptionPnlPayload: The loaded option data.
    """
    if dm is None and opttick is None:
        raise ValueError("Either 'dm' or 'opttick' must be provided.")
    if dm is None:
        dm = OptionDataManager(opttick=opttick)

    ## Back up yesterday by 1BDAY to ensure inclusive data retrieval
    yesterday = change_to_last_busday(yesterday - BDay(1))
    
    ## Query Vol Data
    vol_req = dm.get_timeseries(
        start=yesterday,
        end=today,
        type_='vol',
        model='bs'
    )
    vol_data = vol_req.post_processed_data['Midpoint_bs_iv']
    vol_data.name='vol'
    
    ## Query Option Spot Data
    spot_req = dm.get_timeseries(
        start=yesterday,
        end=today,
        type_='spot'
    )
    spot_data = spot_req.post_processed_data['Midpoint']
    spot_data.name='spot'
    
    ## Query Greeks Data
    greeks_req = dm.get_timeseries(
        start=yesterday,
        end=today,
        type_='greeks',
        model='bs'
    )
    greeks_cols = greeks_req.post_processed_data.columns.str.contains('Midpoint')
    greeks_data = greeks_req.post_processed_data.loc[:, greeks_cols]
    greeks_data.columns = [col.split('_')[-1] for col in greeks_data.columns]

    ## Load Symbol Payload
    sym_payload = SYMBOL_PAYLOADS.get((dm.symbol, yesterday, today))
    if sym_payload is None:
        sym_payload = load_symbol_payload(
            symbol=dm.symbol,
            today=today,
            yesterday=yesterday
        )
        SYMBOL_PAYLOADS[(dm.symbol, yesterday, today)] = sym_payload
    
    ## Load Rates Payload
    rates_payload = SYMBOL_PAYLOADS.get(('RATES_USD', yesterday, today))
    if rates_payload is None:
        rates_payload = load_rate_payload(
            today=today,
            yesterday=yesterday,
        )
        SYMBOL_PAYLOADS[('RATES_USD', yesterday, today)] = rates_payload

    payload = OptionPnlPayload(
        opttick=dm.opttick,
        date=today,
        vol=vol_data,
        spot=spot_data,
        greeks=greeks_data,
        asset_payload=sym_payload,
        rates_payload=rates_payload
    )
    payload.attribution_model = AttributionModel.xMULTIPLY
    payload = add_dod_change(payload, yesterday, today)

    return calculate_pnl_decomposition(payload)


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def calculate_pnl_decomposition(
    payload: OptionPnlPayload,
    trade_pnl_entries: Optional[List[TradePnlInfo]] = None
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
    greeks['shifted_date'] = greeks.index.shift(1)
    greeks = greeks.reset_index().set_index('shifted_date')
    delta_pnl = (dod_change['asset_spot_change'] * greeks['delta']).dropna()
    delta_pnl.name = 'delta_pnl'

    gamma_pnl = ((0.5 * dod_change['asset_spot_change'] **2) * greeks['gamma']).dropna()
    gamma_pnl.name = 'gamma_pnl'

    ## Vega is per 1% vol change, hence the multiplication by 100
    ## Currently using vol_change in decimal form (e.g., 0.01 for 1%)
    vega_pnl = (dod_change['vol_change'] * greeks['vega'] * 100).dropna()
    vega_pnl.name = 'vega_pnl'

    theta_pnl = (dod_change.index.to_series().diff().dt.days * greeks['theta']).dropna()
    theta_pnl.name = 'theta_pnl'

    vanna_pnl = (dod_change['asset_spot_change'] * dod_change['vol_change'] * greeks['vanna'] * 100).dropna()
    vanna_pnl.name = 'vanna_pnl'

    volga_pnl = (0.5 * ((dod_change['vol_change'] **2) * 100) * greeks['volga']).dropna()
    volga_pnl.name = 'volga_pnl'

    rho_pnl = (dod_change['rates_change'] * greeks['rho'] * 100).dropna()
    rho_pnl.name = 'rho_pnl'

    total_pnl = delta_pnl + gamma_pnl + vega_pnl + theta_pnl + vanna_pnl + rho_pnl + volga_pnl
    total_pnl.name = 'total_pnl_excl_trade_pnl'

    unexplained_pnl = (dod_change['opt_change'] - total_pnl).dropna()
    unexplained_pnl.name = 'unexplained_pnl'

    opt_change = dod_change['opt_change'].dropna()
    opt_change.name = 'opt_dod_change'


    all_pnl: pd.DataFrame = pd.concat(
        [
            delta_pnl,
            gamma_pnl,
            vega_pnl,
            theta_pnl,
            vanna_pnl,
            rho_pnl,
            volga_pnl,
            total_pnl,
            unexplained_pnl,
            opt_change
        ],
        axis=1
    )
    all_pnl.index.name = 'date'
    all_pnl = trade_pnl_adjustment(
        attribution_table=all_pnl,
        entry_info=trade_pnl_entries or []
    )
    payload.attribution = all_pnl
    return payload
