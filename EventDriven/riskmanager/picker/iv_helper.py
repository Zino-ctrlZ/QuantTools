
from EventDriven._vars import load_riskmanager_cache
from trade.optionlib.assets.forward import vectorized_forward_continuous
from trade.helpers.helper import time_distance_helper
from trade.optionlib.vol.implied_vol import vector_bsm_iv_estimation
from trade.optionlib.greeks.analytical.black_scholes import black_scholes_analytic_greeks_vectorized
from trade.datamanager import RatesDataManager
import pandas as pd
from trade.datamanager import DividendDataManager
from trade.datamanager._enums import DivType
import numpy as np

CHAIN_GREEKS_CACHE = load_riskmanager_cache(target="chain_greeks_cache", create_on_missing=True, clear_on_exit=False)
rates_cache = {}




def get_rates_on_date(date):
    string_date = pd.to_datetime(date).strftime("%Y-%m-%d")
    if string_date in rates_cache:
        return rates_cache[string_date]
    rates_dm = RatesDataManager()
    rate = rates_dm.get_rate(date).timeseries.values[0]
    rates_cache[string_date] = rate
    return rate


def _add_greeks_and_iv_to_chain(filtered: pd.DataFrame, date: pd.Timestamp, chain_spot: float) -> pd.DataFrame:
    if filtered.empty:
        return filtered
    date = pd.to_datetime(date).date()
    ## get rates data for the date
    at_time = get_rates_on_date(date)

    ## Filter for contracts with NaN iv to start adding greeks and iv
    nan_iv_chain = filtered[filtered["iv"].isna()]

    ## Check cache for existing iv and greeks before calculating
    cached_data = {}
    for idx, row in nan_iv_chain.iterrows():
        contract_key = (row["root"], row["expiration"], row["strike"], row["right"], date)
        if contract_key in CHAIN_GREEKS_CACHE:
            cached_data[idx] = CHAIN_GREEKS_CACHE[contract_key]

    ## If there are cached values, add them to the filtered DataFrame and return early
    if cached_data:
        for idx, data in cached_data.items():
            filtered.loc[idx, "iv"] = data["iv"]
            filtered.loc[idx, "delta"] = data["delta"]
            filtered.loc[idx, "gamma"] = data["gamma"]
            filtered.loc[idx, "vega"] = data["vega"]
            filtered.loc[idx, "theta"] = data["theta"]
            filtered.loc[idx, "rho"] = data["rho"]
            filtered.loc[idx, "volga"] = data["volga"]

    ## Filter out the contracts that were found in the cache to avoid redundant calculations
    if cached_data:
        nan_iv_chain = nan_iv_chain[~nan_iv_chain.index.isin(cached_data.keys())]
    ## Get dividend data
    div_dm = DividendDataManager(filtered["root"].iloc[0])  # Assuming all contracts in the chain have the same root
    q = div_dm.get_schedule(date, date, DivType.CONTINUOUS).timeseries.values[0]

    ## Calculate forward price for the contracts with NaN iv
    t = np.array(time_distance_helper([date] * len(nan_iv_chain["expiration"].values), nan_iv_chain["expiration"].values))
    q_factor = np.exp(-q * t)
    f = vectorized_forward_continuous(
        S=[chain_spot] * len(nan_iv_chain["expiration"].values),
        r=[at_time] * len(nan_iv_chain["expiration"].values),
        q_factor=q_factor,
        T=t,
    )

    ## Calculate iv for the contracts with NaN iv
    iv = vector_bsm_iv_estimation(
        F=f,
        K=nan_iv_chain["strike"].values,
        T=t,
        r=[at_time] * len(nan_iv_chain["expiration"].values),
        market_price=nan_iv_chain["midpoint"].values,
        right=nan_iv_chain["right"].str.lower().values,
    )

    ## Calculate greeks for the contracts with NaN iv
    greeks = black_scholes_analytic_greeks_vectorized(
        F=f,
        K=nan_iv_chain["strike"].values,
        T=t,
        r=[at_time] * len(nan_iv_chain["expiration"].values),
        sigma=iv,
        option_type=nan_iv_chain["right"].str.lower().values,
    )

    ## Add the calculated iv and greeks back to the filtered chain
    greeks_df = pd.DataFrame(greeks, index=nan_iv_chain.index)
    filtered.loc[nan_iv_chain.index, "iv"] = iv
    filtered.loc[nan_iv_chain.index, "delta"] = greeks_df["delta"]
    filtered.loc[nan_iv_chain.index, "gamma"] = greeks_df["gamma"]
    filtered.loc[nan_iv_chain.index, "vega"] = greeks_df["vega"]
    filtered.loc[nan_iv_chain.index, "theta"] = greeks_df["theta"]
    filtered.loc[nan_iv_chain.index, "rho"] = greeks_df["rho"]
    filtered.loc[nan_iv_chain.index, "volga"] = greeks_df["volga"]

    ## Store the calculated iv and greeks in the CHAIN_GREEKS_CACHE
    for _, row in filtered.iterrows():
        contract_key = (row["root"], row["expiration"], row["strike"], row["right"], date)
        CHAIN_GREEKS_CACHE[contract_key] = {
            "iv": row["iv"],
            "delta": row["delta"],
            "gamma": row["gamma"],
            "vega": row["vega"],
            "theta": row["theta"],
            "rho": row["rho"],
            "volga": row["volga"],
        }

    return filtered