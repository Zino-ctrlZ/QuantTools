from pathlib import Path
import os
from datetime import datetime
from EventDriven.riskmanager.market_data import MarketTimeseries

DM_GEN_PATH = Path(os.getenv("GEN_CACHE_PATH")) / "dm_gen_cache"
TS = MarketTimeseries(_end=datetime.now())
