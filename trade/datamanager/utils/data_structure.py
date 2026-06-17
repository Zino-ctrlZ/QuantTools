from datetime import datetime
from typing import Union
import pandas as pd
import numpy as np # noqa
from trade.datamanager.exceptions import EmptyDataException
from trade.helpers.helper import to_datetime
from trade.helpers.Logging import setup_logger
from trade.datamanager.utils.logging import get_logging_level, UTILS_LOGGER_NAME
from trade import HOLIDAY_SET # noqa

logger = setup_logger(UTILS_LOGGER_NAME, stream_log_level=get_logging_level())
PANDAS_DATA_HINT = Union[pd.Series, pd.DataFrame]

class EmptyFloat(float):
    """"""
    

def _data_structure_sanitize(
    df: Union[pd.Series, pd.DataFrame],
    start: Union[datetime, str],
    end: Union[datetime, str],
    source_name: str = "",
) -> Union[pd.Series, pd.DataFrame]:
    """Sanitizes timeseries by deduping, filtering, and padding to business days.

    Reindexes to the business-day grid between the first and last observed dates,
    then forward-fills missing vendor prints (e.g. IRX gaps).
    """
    logger.info(f"Sanitizing data from {start} to {end}...")
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
    df = df[(df.index.date >= pd.to_datetime(start).date()) & (df.index.date <= pd.to_datetime(end).date())]

    if df.empty:
        raise EmptyDataException(f"No data available after sanitization between {start} and {end}. Source: {source_name}")

    # Re-sort after filtering
    df = df.sort_index()

    # Index name=datetime
    df.index.name = "datetime"

    # Pad to business-day grid, then forward-fill missing vendor prints.
    all_bus_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq="B")
    all_bus_days = [d for d in all_bus_days if d.strftime("%Y-%m-%d") not in HOLIDAY_SET]
    df = df.reindex(all_bus_days, fill_value=np.nan)

    # Filter out holidays
    df = df[~df.index.strftime("%Y-%m-%d").isin(HOLIDAY_SET)]

    return df
