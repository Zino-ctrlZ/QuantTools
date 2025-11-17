from __future__ import annotations
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

OPTION_TIMESERIES_START_DATE: str|datetime = '2017-01-01'
Y1_LAGGED_START_DATE: str|datetime = (pd.to_datetime(OPTION_TIMESERIES_START_DATE) - relativedelta(years=1)).strftime('%Y-%m-%d')
Y2_LAGGED_START_DATE: str|datetime = (pd.to_datetime(OPTION_TIMESERIES_START_DATE) - relativedelta(years=2)).strftime('%Y-%m-%d')
Y3_LAGGED_START_DATE: str|datetime = (pd.to_datetime(OPTION_TIMESERIES_START_DATE) - relativedelta(years=3)).strftime('%Y-%m-%d')