## Initialisation of the trade package
import warnings
warnings.filterwarnings("ignore")
import os, sys
import json
import pandas_market_calendars as mcal
import pandas as pd
from datetime import datetime
sys.path.append(os.environ.get('DBASE_DIR', ''))
sys.path.append(os.environ.get('WORK_DIR', ''))

POOL_ENABLED = None
def set_pool_enabled(value: bool):
    """
    Set the pool enabled flag.
    """
    global POOL_ENABLED
    POOL_ENABLED = value

def get_pool_enabled():
    """
    Get the pool enabled flag.
    """
    return POOL_ENABLED

set_pool_enabled(bool(os.environ.get('POOL_ENABLED')))

## Universal Holidays set
## Get Business days FOR NYSE (Some days are still trading days)
nyse = mcal.get_calendar('NYSE')
schedule = nyse.schedule(start_date='2000-01-01', end_date=datetime.today())
all_trading_days = mcal.date_range(schedule, frequency='1D').date
all_days = pd.date_range(start='2000-01-01', end=datetime.today(), freq='B')
holidays = set(all_days.difference(all_trading_days).strftime('%Y-%m-%d').to_list())
HOLIDAY_SET = set(holidays)

## Additional holidays
HOLIDAY_SET.update({
    '2025-01-09', ## Jimmy Carter's Death
}) 


## Import Pricing Config
with open(f"{os.environ['WORK_DIR']}/pricingConfig.json") as f:
    PRICING_CONFIG = json.load(f)