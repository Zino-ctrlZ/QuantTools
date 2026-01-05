# Cache Mechanism

## Overview

The caching system uses **persistent disk storage** with SQLite databases to store option data factors (spot, vol, greeks). Data persists across sessions with a 200-day expiry policy.

## Storage Architecture

### Cache Location (On Disk)
```
$WORK_DIR/.cache/data_manager_cache/
├── dm_option_spot.db        # Raw spot data (bid, ask, last, volume, OI)
├── dm_bs_vol.db             # Black-Scholes implied volatility
├── dm_bs_greeks.db          # Black-Scholes greeks
├── dm_binomial_vol.db       # Binomial implied volatility
└── dm_binomial_greeks.db    # Binomial greeks
```

**Path Resolution:**
```python
import os
WORK_DIR = os.getenv('WORK_DIR', os.path.expanduser('~'))
cache_dir = os.path.join(WORK_DIR, '.cache', 'data_manager_cache')
```

Default: `~/.cache/data_manager_cache/` (Unix) or `C:\Users\<user>\.cache\data_manager_cache\` (Windows)

### Database Structure

Each SQLite database stores data using a **two-level key hierarchy**:

**Primary Key**: `(interval, ticker)` tuple
- `interval`: Time granularity ('1d', '1h', '5m', etc.)
- `ticker`: Option ticker (e.g., 'AAPL20250620C250')

**Value**: Pickled pandas DataFrame with DatetimeIndex

Example database content:
```python
{
    ('1d', 'AAPL20250620C250'): DataFrame with dates as index,
    ('1d', 'TSLA20250321C260'): DataFrame with dates as index,
    ('1h', 'AAPL20250620C250'): DataFrame with dates as index (hourly data),
    # ... more entries
}
```

## Cache Operations

### 1. Loading Cache Instances

```python
from DataManagers.option_cache.helpers import load_option_data_cache

# Load all 5 cache instances
spot_cache, bs_vol, bs_greeks, binomial_vol, binomial_greeks = load_option_data_cache()
```

**What happens:**
```python
def load_option_data_cache():
    cache_dir = os.path.join(os.getenv('WORK_DIR'), '.cache', 'data_manager_cache')
    
    # Create CustomCache instances
    spot_cache = CustomCache('dm_option_spot', cache_dir=cache_dir, expire_days=200)
    bs_vol = CustomCache('dm_bs_vol', cache_dir=cache_dir, expire_days=200)
    bs_greeks = CustomCache('dm_bs_greeks', cache_dir=cache_dir, expire_days=200)
    binomial_vol = CustomCache('dm_binomial_vol', cache_dir=cache_dir, expire_days=200)
    binomial_greeks = CustomCache('dm_binomial_greeks', cache_dir=cache_dir, expire_days=200)
    
    return (spot_cache, bs_vol, bs_greeks, binomial_vol, binomial_greeks)
```

### 2. Retrieving Cached Data

```python
from DataManagers.option_cache.utils import get_cached_data

cached_data = get_cached_data(cache, interval, opttick)
# Returns: DataFrame with DatetimeIndex or None if not cached
```

**Implementation:**
```python
def get_cached_data(cache: CustomCache, interval: str, opttick: str) -> pd.DataFrame | None:
    key = (interval, opttick)
    result = cache.get(key)
    
    if result is not None and isinstance(result, pd.DataFrame):
        if not result.empty and isinstance(result.index, pd.DatetimeIndex):
            return result
    
    return None
```

### 3. Saving to Cache

```python
from DataManagers.option_cache.utils import save_to_cache

save_to_cache(cache, interval, opttick, new_data, merge_with_existing=True)
```

**Parameters:**
- `cache`: CustomCache instance
- `interval`: Time granularity ('1d')
- `opttick`: Option ticker
- `new_data`: DataFrame to save (with DatetimeIndex)
- `merge_with_existing`: If True, merge with existing cached data

**Implementation:**
```python
def save_to_cache(
    cache: CustomCache,
    interval: str,
    opttick: str,
    new_data: pd.DataFrame,
    merge_with_existing: bool = True,
) -> None:
    if new_data.empty or not isinstance(new_data.index, pd.DatetimeIndex):
        return
    
    key = (interval, opttick)
    
    # Normalize column names to lowercase
    new_data = new_data.copy()
    new_data.columns = [x.lower() for x in new_data.columns]
    
    if merge_with_existing:
        # Get existing cached data
        existing = get_cached_data(cache, interval, opttick)
        
        if existing is not None:
            # Merge: concat + remove duplicates + sort
            merged = pd.concat([existing, new_data])
            merged = merged[~merged.index.duplicated(keep='last')]
            merged = merged.sort_index()
            cache.set(key, merged)
        else:
            # No existing data, save new
            cache.set(key, new_data)
    else:
        # Overwrite existing data
        cache.set(key, new_data)
```

### 4. Checking Cache Completeness

```python
from DataManagers.option_cache.utils import check_cache_completeness

is_complete, missing_dates = check_cache_completeness(cached_data, start, end)
```

**Returns:**
- `is_complete`: True if cache covers entire date range
- `missing_dates`: List of missing business dates

**Implementation:**
```python
def check_cache_completeness(
    cached_data: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp
) -> tuple[bool, list]:
    if cached_data is None or cached_data.empty:
        return False, []
    
    # Get requested business dates
    requested_dates = pd.bdate_range(start, end)
    
    # Get cached dates
    cached_dates = pd.DatetimeIndex(cached_data.index)
    
    # Find missing dates
    missing = requested_dates.difference(cached_dates)
    
    return len(missing) == 0, missing.tolist()
```

## Cache Selection Logic

Different cache databases are selected based on request parameters:

```python
def get_cache_for_type_and_model(data_type: str, model: str, caches: tuple):
    spot_cache, bs_vol, bs_greeks, binomial_vol, binomial_greeks = caches
    
    if data_type == 'spot':
        return spot_cache  # Spot always uses same cache
    elif data_type == 'vol':
        return bs_vol if model == 'bs' else binomial_vol
    elif data_type == 'greeks':
        return bs_greeks if model == 'bs' else binomial_greeks
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
```

## Data Format in Cache

### Spot Cache Format
```python
# Key: ('1d', 'AAPL20250620C250')
# Value: DataFrame
DataFrame({
    'bid': [214.5, 215.0, ...],
    'ask': [215.0, 215.5, ...],
    'last': [214.75, 215.25, ...],
    'volume': [1000, 1500, ...],
    'openinterest': [5000, 5100, ...],
}, index=DatetimeIndex(['2024-01-01', '2024-01-02', ...]))
```

### Vol Cache Format
```python
# Key: ('1d', 'AAPL20250620C250')
# Value: DataFrame
DataFrame({
    'iv': [0.25, 0.26, ...],
    'ask_iv': [0.26, 0.27, ...],
    'bid_iv': [0.24, 0.25, ...],
    'mid_iv': [0.25, 0.26, ...],
    'last_iv': [0.25, 0.26, ...],
}, index=DatetimeIndex(['2024-01-01', '2024-01-02', ...]))
```

### Greeks Cache Format
```python
# Key: ('1d', 'AAPL20250620C250')
# Value: DataFrame
DataFrame({
    'delta': [0.55, 0.56, ...],
    'gamma': [0.02, 0.021, ...],
    'vega': [0.15, 0.16, ...],
    'theta': [-0.05, -0.051, ...],
    'rho': [0.12, 0.13, ...],
    'vanna': [0.01, 0.011, ...],
    'volga': [0.008, 0.009, ...],
    'midpoint_delta': [0.54, 0.55, ...],
    # ... more greek variants
}, index=DatetimeIndex(['2024-01-01', '2024-01-02', ...]))
```

## Cache Expiry

**Expiry Policy**: 200 days from last access

```python
cache = CustomCache('dm_option_spot', expire_days=200)
```

**How it works:**
- Each cache entry tracks last access time
- On cache access, timestamp is updated
- Entries not accessed for 200 days are automatically purged
- No manual cleanup needed

## Cache Management

### Clear Specific Cache
```python
from DataManagers.option_cache.helpers import load_option_data_cache

spot_cache, bs_vol, bs_greeks, binomial_vol, binomial_greeks = load_option_data_cache()

# Clear all BS greeks
bs_greeks.clear()

# Clear specific ticker
key = ('1d', 'AAPL20250620C250')
bs_greeks.delete(key)
```

### Clear All Caches
```python
from DataManagers.option_cache import clear_all_caches

clear_all_caches()  # Clears all 5 cache databases
```

### Get Cache Statistics
```python
# Check cache size
cache_data = bs_greeks.cache  # Access underlying dict
print(f"Cached tickers: {len(cache_data)}")

# List all cached tickers
for (interval, ticker) in cache_data.keys():
    print(f"{ticker} ({interval})")
```

## Performance Characteristics

### Read Performance
- **Cache Hit**: ~0.001s per lookup (in-memory dictionary)
- **Cache Miss**: 0s (returns None immediately)

### Write Performance
- **New Entry**: ~0.01s per save (pickle + SQLite write)
- **Merge**: ~0.02s per save (load + concat + dedupe + save)

### Storage Size
- **Spot data**: ~50KB per ticker per 60 days
- **Vol data**: ~20KB per ticker per 60 days
- **Greeks data**: ~100KB per ticker per 60 days
- **Total**: ~170KB per ticker per 60 days

For 40 tickers:
- Total cache size: ~6.8 MB
- Time to populate: ~479s (first run)
- Time to retrieve: ~29s (subsequent runs)

## Column Normalization

To prevent issues with inconsistent column names:

**Save (normalize to lowercase):**
```python
data.columns = [x.lower() for x in data.columns]
cache.set(key, data)
```

**Load (capitalize to match parent format):**
```python
cached_data = cache.get(key)
cached_data.columns = [x.capitalize() for x in cached_data.columns]
```

This ensures:
- No duplication (e.g., 'Openinterest' vs 'openinterest')
- Consistent cache storage format
- Compatible with parent manager expectations

## Next: Underlier Caching

See `5_UNDERLIER_CACHING.md` for details on how underlier data (stock prices, dividends) is handled.
