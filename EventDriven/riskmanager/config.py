import pandas as pd
import numpy as np
from datetime import datetime


AVOID_OPTTICKS = {
    'AAPL': ['AAPL20241220C225', 'AAPL20241220C220', ],
    'TSLA': ['TSLA20230120C1250', 'TSLA20230120C1275']
    }


FFWD_OPT_BY_UNDERLIER = {
    'TSLA': ['2022-01-05']
}


def get_avoid_opticks(ticker) -> list:
    """Get the list of options to avoid."""
    """Get the list of options to avoid for a given ticker and date."""
    if ticker in AVOID_OPTTICKS:
        return AVOID_OPTTICKS[ticker]   
    return []

def add_avoid_optick(optick, ticker):
    """Add an option tick to avoid for a given ticker and date."""
    if ticker not in AVOID_OPTTICKS:
        AVOID_OPTTICKS[ticker] = []
    if optick not in AVOID_OPTTICKS[ticker]:
        AVOID_OPTTICKS[ticker].append(optick)


def get_fwd_opt_by_underlier():
    """Get the list of forward options by underlier."""
    return FFWD_OPT_BY_UNDERLIER

def add_fwd_opt_by_underlier(underlier, fwd_date):
    """Add a forward option for a given underlier."""
    if underlier not in FFWD_OPT_BY_UNDERLIER:
        FFWD_OPT_BY_UNDERLIER[underlier] = []
    if fwd_date not in FFWD_OPT_BY_UNDERLIER[underlier]:
        FFWD_OPT_BY_UNDERLIER[underlier].append(fwd_date)

def ffwd_data(data:pd.DataFrame, underlier:str) -> pd.DataFrame:
    """Fill Forward data for a given underlier and date."""
    if underlier not in FFWD_OPT_BY_UNDERLIER:
        return data
    fwd_dates = FFWD_OPT_BY_UNDERLIER[underlier]
    for dt in fwd_dates:
        if dt not in data.index:
            continue
        data.loc[data.index == dt,:] = np.nan
        data.fillna(method='ffill', inplace=True)
    return data
    