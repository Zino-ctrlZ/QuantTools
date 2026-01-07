from datetime import datetime
import pandas as pd
import numpy as np
from dbase.DataAPI.ThetaData import (
    retrieve_eod_ohlc,
    retrieve_chain_bulk,
)
from trade.helpers.helper import retrieve_timeseries, change_to_last_busday, time_distance_helper
from trade.helpers.Logging import setup_logger
from trade.assets.rates import get_risk_free_rate_helper
from trade.optionlib.assets.forward import (
    vectorized_market_forward_calc,
)
from trade.optionlib.assets.dividend import (
    vector_convert_to_time_frac,
)
from trade.optionlib.config.defaults import DAILY_BASIS
from trade.optionlib.config.ssvi.controller import (
    get_pricing_config,
)
from trade.optionlib.utils.batch_operation import vector_batch_processor
from trade.optionlib.vol.implied_vol import (
    estimate_crr_implied_volatility,
    vector_vol_estimation,
    bsm_vol_est_brute_force,
)

logger = setup_logger("optionlib.ssvi")
logger.info("Loaded vol.ssvi.utils")


## Getters
def get_k_grid(chain: pd.DataFrame, col_name: str = "strike") -> np.ndarray:
    """
    Retrieves the strike prices from the option chain.

    Args:
        chain (pd.DataFrame): The option chain DataFrame.
        col_name (str): The column name to retrieve (default is 'strike').

    Returns:
        np.ndarray: The strike prices for the option chain.
    """
    return chain[col_name].values


def get_t_grid(chain: pd.DataFrame, col_name: str = "t") -> np.ndarray:
    """
    Retrieves the maturities from the option chain.

    Args:
        chain (pd.DataFrame): The option chain DataFrame.
        col_name (str): The column name to retrieve (default is 't').
    Returns:
        np.ndarray: The maturities for the option chain.
    """
    return chain[col_name].values


def get_market_iv_grid(chain: pd.DataFrame, col_name: str = "iv") -> np.ndarray:
    """
    Retrieves the market implied volatilities from the option chain.

    Args:
        chain (pd.DataFrame): The option chain DataFrame.
        col_name (str): The column name to retrieve (default is 'iv').

    Returns:
        np.ndarray: The market implied volatilities for the option chain.
    """
    return chain[col_name].values


def get_fwd_grid(chain: pd.DataFrame, col_name: str = "f") -> float:
    """
    Retrieves the forward price from the option chain.

    Args:
        chain (pd.DataFrame): The option chain DataFrame.
        col_name (str): The column name to retrieve (default is 'f').
    """
    return chain[col_name].iloc[0]  # Assuming F is constant across the chains


## Volatility Calculation Functions
def get_bs_vol_on_chain(
    chain: pd.DataFrame,
    valuation_date: str,
    rate_col_name: str = None,
    forward_col_name: str = "f",
    mid_col_name: str = "midpoint",
) -> pd.Series:
    """
    Estimates the Black-Scholes implied volatility for a given option chain.

    Args:
        chain (pd.DataFrame): The option chain DataFrame.
            Expected Columns: `f`, `strike`, `t`, `midpoint`, `right`.
        valuation_date (str): The date of valuation.

    Returns:
        pd.Series: The estimated Black-Scholes implied volatility for the option chain.
    """
    if rate_col_name is None:
        _r = [get_rates(valuation_date)] * len(chain)
    else:
        _r = chain[rate_col_name]

    params = list(
        zip(
            chain[forward_col_name if forward_col_name in chain.columns else "f"],
            chain["strike"],
            chain["t"],
            _r,
            chain[mid_col_name if mid_col_name in chain.columns else "midpoint"],
            chain["right"].str.lower(),
        )
    )

    return vector_batch_processor(
        vector_vol_estimation,
        bsm_vol_est_brute_force,
        params,
    )


def get_discrete_crr_vol_on_chain(
    chain: pd.DataFrame, valuation_date: str, rates_col_name: str = None, div_type: str = "discrete", n: int = 250
) -> pd.Series:
    """
    Estimates the discrete CRR implied volatility for a given option chain.

    Args:
        chain (pd.DataFrame): The option chain DataFrame.
            Expected Columns: `spot`, `strike`, `t`, `midpoint`, `div_schedule`, `right`.
        valuation_date (str): The date of valuation.

    Returns:
        pd.Series: The estimated discrete CRR implied volatility for the option chain.
    """
    ## Get risk-free rates
    if rates_col_name is None:
        _r = [get_rates(valuation_date)] * len(chain)
    else:
        _r = chain[rates_col_name].tolist()

    ## Pick div based on div_type
    if div_type not in ["discrete", "continuous"]:
        raise ValueError("div_type must be either 'discrete' or 'continuous'.")
    elif div_type == "continuous":
        divs = chain["div_pv"].tolist()
    else:
        divs = chain["div_schedule"].tolist()

    crr_vector_params_discrete = list(
        zip(
            chain["spot"],
            chain["strike"].tolist(),  ## Spot, Strike
            chain["t"],
            _r,  ## Time to Maturity, Risk Free Rate
            chain["midpoint"],  ## Midpoint Price
            divs,  ## Dividends based on div_type
            chain["right"].str.lower().tolist(),  ## Option Type
            [n] * len(chain),  ## Number of Steps
            [div_type] * len(chain),  ## Dividend Type
            [True] * len(chain),
        )
    )  ## American==True, European==False

    return vector_batch_processor(vector_vol_estimation, estimate_crr_implied_volatility, crr_vector_params_discrete)


def get_chain(tick: str, date: str) -> pd.DataFrame:
    """
    Retrieves the option chain for a given ticker and date, along with additional calculated fields.
    Args:
        tick (str): The ticker symbol for the underlying asset.
        date (str): The date for which to retrieve the option chain.
    Returns:
        pd.DataFrame: A DataFrame containing the option chain with additional fields.
    """
    spot = get_spot(tick, date, spot_type="chain_price")
    date = change_to_last_busday(date)
    chain = retrieve_chain_bulk(
        tick,
        0,  ## This is to get all expirations
        date,
        date,
        "16:00",
    )
    chain["spot"] = spot
    chain["valuation_date"] = date
    chain["moneyness"] = chain["Strike"] / chain["spot"]
    chain["log_moneyness"] = np.log(chain["moneyness"])
    chain["T"] = chain["Expiration"].apply(
        lambda x: time_distance_helper(
            x,
            date,
        )
    )
    chain["T"] = chain["T"].astype(float)
    chain["DTE"] = chain["T"] * DAILY_BASIS

    return chain


def format_chain(chain: pd.DataFrame) -> pd.DataFrame:
    """
    Formats the option chain DataFrame by renaming columns and converting types.

    Args:
        chain (pd.DataFrame): The option chain DataFrame to format.

    Returns:
        pd.DataFrame: The formatted option chain DataFrame.
    """
    chain.columns = chain.columns.str.lower()
    chain["right"] = chain["right"].str.lower()
    return chain


def confine_chain_with_pricing_config(chain: pd.DataFrame) -> pd.DataFrame:
    """
    Confines the chain to the pricing configuration limits.

    Args:
        chain (pd.DataFrame): The option chain DataFrame.
            expected columns: ['dte', 'moneyness']

    Returns:
        pd.DataFrame: The confined option chain.
    """
    conf = get_pricing_config()
    upper_dte = conf["VOL_SURFACE_MAX_DTE_THRESHOLD"]
    lower_dte = conf["VOL_SURFACE_MIN_DTE_THRESHOLD"]
    upper_moneyness = conf["VOL_SURFACE_MAX_MONEYNESS_THRESHOLD"]
    lower_moneyness = conf["VOL_SURFACE_MIN_MONEYNESS_THRESHOLD"]

    return chain[
        (chain["dte"] >= lower_dte)
        & (chain["dte"] <= upper_dte)
        & (chain["moneyness"] >= lower_moneyness)
        & (chain["moneyness"] <= upper_moneyness)
    ]


def get_forward_price_on_chain(
    chain: pd.DataFrame, valuation_date: str, r: float = None, div_type: str = "discrete"
) -> pd.DataFrame:
    """
    Calculates the forward price for a given option chain.

    Args:
        chain (pd.DataFrame): The option chain DataFrame.
        valuation_date (str): The date of valuation.
        end_date (str): The expiration date of the option.
        r (float): The risk-free rate.
        div_type (str): Type of dividend ('discrete' or 'continuous').

    Returns:
        float: The calculated forward price.
    """

    ## This is per-ticker function. There must be only one ticker, one spot price
    assert len(chain["root"].unique()) == 1, "Chain must contain options from only one ticker."
    assert len(chain["spot"].unique()) == 1, "Chain must contain a single spot price."
    assert len(chain["valuation_date"].unique()) == 1, "Chain must contain a single valuation date."
    assert div_type in ["discrete", "continuous"], "div_type must be either 'discrete' or 'continuous'."

    ## For speed, we will use unique items, and merge back later
    chain = chain.copy()
    end_dates = chain["expiration"].unique()
    valuation_dates = [valuation_date] * len(end_dates)
    S = [chain["spot"].tolist()[0]] * len(end_dates)
    tickers = [chain["root"].iloc[0]] * len(end_dates)
    if r is None:
        r = [get_rates(valuation_date)] * len(end_dates)
    else:
        r = [r] * len(end_dates)

    ## This function returns similar things based on div_type
    ## 1. If div_type is 'discrete', it returns the forward price, (dividend schedule & present value of dividends (It's sum of dividends))
    ## 2. If div_type is 'continuous', it returns the forward price, (dividend rate & present value of dividend rate)
    f, (actual, pv) = vectorized_market_forward_calc(
        ticks=tickers,
        S=S,
        valuation_dates=valuation_dates,
        end_dates=end_dates,
        r=r,
        div_type=div_type,
        return_div=True,
    )

    ## Create a series for merging
    f = pd.Series(f, index=end_dates, name="f")
    pv = pd.Series(pv, index=end_dates, name="div_pv")

    if div_type == "discrete":
        actual = vector_convert_to_time_frac(
            actual,
            valuation_dates=[valuation_date] * len(actual),
            end_dates=end_dates,
        )

    ## Merge back to chain
    actual = pd.Series(actual, index=end_dates, name="div_schedule")
    chain = chain.merge(actual, left_on="expiration", right_index=True, how="left")
    chain = chain.merge(f, left_on="expiration", right_index=True, how="left")
    chain = chain.merge(pv, left_on="expiration", right_index=True, how="left")

    ## Calculate moneyness and log moneyness based on forward price
    chain["f_moneyness"] = chain["strike"] / chain["f"]
    chain["f_log_moneyness"] = np.log(chain["f_moneyness"])

    return chain


def get_option_eod_price(date, contract_series):
    """
    Retrieves the end-of-day price for a given option contract on a specific date.

    Args:
        date (datetime): The date for which to retrieve the price.
        contract_series (pd.Series): The series containing option contract details.

    Returns:
        float: The end-of-day price of the option contract.
    """
    eod_data = retrieve_eod_ohlc(
        symbol=contract_series["root"],
        end_date=date,
        start_date=date,
        exp=str(contract_series["expiration"]),
        right=contract_series["right"],
        strike=contract_series["strike"],
    )
    return eod_data.Midpoint[0]


def get_spot(tick, date, spot_type="close"):
    """
    Retrieves the spot price for a given ticker on a specific date.
    """
    return retrieve_timeseries(tick, date, date, spot_type=spot_type)["close"][0]


def get_rates(date):
    """
    Retrieves the risk-free rate for a given date.

    Args:
        date (datetime): The date for which to retrieve the risk-free rate.

    Returns:
        float: The risk-free rate for the specified date.
    """
    date = pd.to_datetime(date).strftime("%Y-%m-%d")
    return get_risk_free_rate_helper()["annualized"][date]
