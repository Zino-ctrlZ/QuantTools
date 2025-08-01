from datetime import datetime, timedelta
import pandas as pd
# import numpy as np


def time_distance_helper(end: str, strt: str = None) -> float:
    """
    Calculate the time distance between two dates in years.
    Args:
        end (str): Expiration date/End Date in 'YYYY-MM-DD' format.
        strt (str, optional): Start date in 'YYYY-MM-DD' format. Defaults to today's date.
    Returns:
        float: Time distance in years.
    """
    if strt is None:
        strt = datetime.today()
    
    end = pd.to_datetime(end)
    end = end.replace(hour = 16, minute = 0, second = 0, microsecond = 0,)
    parsed_dte, start_date = pd.to_datetime(end), pd.to_datetime(strt)
    if start_date.hour == 0 and start_date.minute == 0 and start_date.second == 0:
        start_date = start_date.replace(hour=16, minute=0, second=0, microsecond=0)
    days = (parsed_dte - start_date).total_seconds()

    T = days/(365.25*24*3600)
    return T