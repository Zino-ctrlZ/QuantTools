from datetime import datetime, date
import pandas as pd
from trade.helpers.helper import is_USholiday

def format_dates(*args):
    return [pd.to_datetime(arg) for arg in args]

def subtract_dates(date1: datetime, date2: datetime) -> int:
    """
    Subtracts two dates and returns the difference in days.
    """

    return (pd.to_datetime(date1) - pd.to_datetime(date2)).days

def get_months_between(start_date: datetime, end_date: datetime) -> int:
    """
    Returns the number of months between two dates.
    """
    return (end_date.year - start_date.year) * 12 + end_date.month - start_date.month

def validate_dates(*args):
    """
    Validates if the input date is a valid datetime object.
    """
    for dt in args:
        if not isinstance(dt, (datetime, pd.Timestamp, date)):
            raise ValueError(f"Invalid date: {dt}. Expected a datetime object.")
        
        if is_USholiday(dt):
            raise ValueError(f"Date {dt} is a US holiday. Please choose a different date.")
        
        if dt.weekday() in [5, 6]:
            raise ValueError(f"Date {dt} falls on a weekend. Please choose a weekday.")

