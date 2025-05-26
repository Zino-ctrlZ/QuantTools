from pathlib import Path
import os
from trade.helpers.helper import setup_logger, CustomCache
logger = setup_logger('DataManagers.cache.py')
DB_CACHE_LOCATION = Path(os.environ['WORK_DIR']) / '.cache' 
DB_CACHE = CustomCache(DB_CACHE_LOCATION, fname = 'dm_cache', clear_on_exit=False, expire_days=180)

def get_cache():
    """
    Returns the cache object.
    """
    return DB_CACHE