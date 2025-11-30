"""
Quick test - just check if cached managers are being used
"""

import pandas as pd
from pandas.tseries.offsets import BDay
from trade.helpers.helper import parse_option_tick
from DataManagers.DataManagers_cached import CachedOptionDataManager

opttick = 'AAPL20241220C215'
parsed = parse_option_tick(opttick)
exp = pd.to_datetime(parsed['exp_date'])
start = exp - BDay(5)  # Just 5 days for speed
end = exp

print(f"Testing {opttick} from {start.date()} to {end.date()}")

manager = CachedOptionDataManager(opttick=opttick)

# Check what managers are installed
print(f"\nspot_manager type: {type(manager.spot_manager).__name__}")
print(f"vol_manager type: {type(manager.vol_manager).__name__}")
print(f"greek_manager type: {type(manager.greek_manager).__name__}")

print("\nIf all show 'Cached*Manager', then factor-level caching is active!")
