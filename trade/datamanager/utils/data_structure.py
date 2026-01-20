from datetime import datetime
from typing import Union
import pandas as pd
from trade.helpers.helper import to_datetime


def _data_structure_sanitize(
    df: Union[pd.Series, pd.DataFrame],
    start: Union[datetime, str],
    end: Union[datetime, str],
) -> Union[pd.Series, pd.DataFrame]:
    """Sanitizes the data structure by removing duplicates and sorting the index."""
    print(f"Sanitizing data from {start} to {end}...")
    if not isinstance(df, (pd.Series, pd.DataFrame)):
        raise TypeError(f"Expected pd.Series or pd.DataFrame for sanitization, got {type(df)}")

    # Ensure DatetimeIndex. If not, attempt conversion
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = to_datetime(df.index, format="%Y-%m-%d")
        except Exception as e:
            raise TypeError("Expected DatetimeIndex for sanitization of timeseries data.") from e

    # Remove duplicates, keeping the last occurrence
    df = df[~df.index.duplicated(keep="last")]

    # Sort the index
    df = df.sort_index()

    # if dataframe, lower case columns
    if isinstance(df, pd.DataFrame):
        df.columns = df.columns.str.lower()

    # Filter by start and end dates
    df = df[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]

    # Re-sort after filtering
    df = df.sort_index()

    # Index name=datetime
    df.index.name = "datetime"

    return df
