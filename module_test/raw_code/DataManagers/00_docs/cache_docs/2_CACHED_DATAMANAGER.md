# Cached DataManager

## Overview

`CachedOptionDataManager` is a drop-in replacement for `OptionDataManager` that adds intelligent factor-level caching. It inherits from `OptionDataManager` and intercepts data queries at three critical bottlenecks.

## Architecture

### Inheritance Structure
```python
CachedOptionDataManager(OptionDataManager)
├── Uses: CachedSpotManager
├── Uses: CachedVolManager  
└── Uses: CachedGreeksManager
```

### Factor-Level Caching
Instead of caching at the final result level, we cache at each **factor** (computation step):

```
User Request: Get greeks data
    ↓
CachedGreeksManager
    ├─ Check greeks cache
    ├─ If miss: Calculate greeks (needs vol data)
    │   ↓
    │   CachedVolManager
    │   ├─ Check vol cache
    │   ├─ If miss: Calculate vol (needs spot data)
    │   │   ↓
    │   │   CachedSpotManager
    │   │   ├─ Check spot cache
    │   │   └─ If miss: Query ThetaData
    │   │
    │   └─ Save vol to cache
    │
    └─ Save greeks to cache
```

## Key Design Principles

### 1. Delegation Pattern
Each cached manager wraps the original manager and delegates to it when cache misses occur:

```python
class CachedVolManager:
    def __init__(self, symbol: str, caches: tuple, opttick: str):
        self._parent_manager = VolDataManager(symbol)  # Original manager
        
    def calculate_iv(self, **kwargs):
        # Check cache first
        cached_vol = get_cached_data(...)
        if cache_hit:
            return cached_vol
        
        # Cache miss: delegate to parent
        self._parent_manager.calculate_iv(**kwargs)
```

### 2. Transparent Integration
The cached managers intercept methods at the bottleneck points:

- **CachedSpotManager**: Intercepts `SpotDataManager.query_thetadata()`
- **CachedVolManager**: Intercepts `VolDataManager.calculate_iv()`
- **CachedGreeksManager**: Intercepts `GreeksDataManager.calculate_greeks()`

### 3. Partial Cache Support
The system intelligently handles partial cache hits:

```python
# Scenario: Cache has days 1-30, user requests days 1-60
cached_data = get_cached_data(cache, '1d', 'AAPL20250620C250')
is_complete, missing_dates = check_cache_completeness(cached_data, start, end)

if not is_complete:
    # Query only missing dates (31-60)
    new_data = parent.query_thetadata(min(missing_dates), max(missing_dates))
    # Merge with cached data
    merged = pd.concat([cached_data, new_data])
```

## Three Cached Managers

### CachedSpotManager
**Intercepts**: ThetaData spot queries  
**Caches**: Raw option spot data (bid, ask, last, volume, open interest)  
**Cache Key**: `(interval, opttick)` → e.g., `('1d', 'AAPL20250620C250')`

**Logic:**
```python
def query_thetadata(self, start, end, **kwargs):
    # 1. Check spot cache
    cached_spot = get_cached_data(spot_cache, interval, opttick)
    
    # 2. Check if cache covers date range
    if cache_complete:
        return cached_spot[start:end]  # HIT
    
    # 3. Query only missing dates
    new_data = self._parent_manager.query_thetadata(missing_start, missing_end)
    
    # 4. Save to cache
    save_to_cache(spot_cache, interval, opttick, new_data)
    
    return new_data
```

### CachedVolManager
**Intercepts**: IV calculation  
**Caches**: Calculated implied volatility columns  
**Cache Key**: `(interval, opttick)` + model type (BS/Binomial)

**Logic:**
```python
def calculate_iv(self, **kwargs):
    # 1. Get raw data from data_request
    raw_data = kwargs['data_request'].raw_spot_data
    
    # 2. Check vol cache
    cached_vol = get_cached_data(vol_cache, interval, opttick)
    
    # 3. Check if cache covers dates
    if cache_complete:
        # Add vol columns to raw_data from cache
        for col in vol_columns:
            raw_data[col] = cached_vol[col]
        return  # Parent modifies in-place
    
    # 4. Calculate vol for missing dates
    self._parent_manager.calculate_iv(**kwargs)
    
    # 5. Extract vol columns and save to cache
    vol_data = raw_data[vol_columns]
    save_to_cache(vol_cache, interval, opttick, vol_data)
```

### CachedGreeksManager
**Intercepts**: Greeks calculation  
**Caches**: All greeks (delta, gamma, vega, theta, rho, vanna, volga)  
**Cache Key**: `(interval, opttick)` + model type (BS/Binomial)

**Logic:**
```python
def calculate_greeks(self, **kwargs):
    # 1. Get raw data from data_request
    raw_data = kwargs['data_request'].raw_spot_data
    
    # 2. Check greeks cache
    cached_greeks = get_cached_data(greeks_cache, interval, opttick)
    
    # 3. Check if cache covers dates
    if cache_complete:
        # Add greek columns to raw_data from cache
        greek_keywords = ['delta', 'gamma', 'vega', 'theta', 'rho', 'vanna', 'volga']
        for col in cached_greeks.columns:
            if any(g in col.lower() for g in greek_keywords):
                raw_data[col] = cached_greeks[col]
        return  # Parent modifies in-place
    
    # 4. Calculate greeks for missing dates
    self._parent_manager.calculate_greeks(**kwargs)
    
    # 5. Extract greek columns and save to cache
    greek_cols = [col for col in raw_data.columns 
                  if any(g in col.lower() for g in greek_keywords)]
    greeks_data = raw_data[greek_cols]
    save_to_cache(greeks_cache, interval, opttick, greeks_data)
```

## Column Normalization

To prevent duplication issues (e.g., 'openinterest' vs 'Openinterest'), we normalize column names:

- **Save to cache**: Lowercase (`delta`, `gamma`, `spot`)
- **Load from cache**: Capitalize to match parent format (`Delta`, `Gamma`, `Spot`)

```python
# Saving
data.columns = [x.lower() for x in data.columns]
save_to_cache(cache, interval, opttick, data)

# Loading
cached_data.columns = [x.capitalize() for x in cached_data.columns]
raw_data[col] = cached_data[col]
```

## Cache Configuration

### Enable/Disable Caching

**Per-instance:**
```python
# Disable for specific instance
dm = CachedOptionDataManager('AAPL20250620C250', enable_cache=False)
```

**Globally:**
```python
# Disable for all instances
CachedOptionDataManager.disable_caching()

# Re-enable
CachedOptionDataManager.enable_caching()
```

### MySQL Query Skip
For maximum performance, you can skip MySQL queries (only use cache):

```python
from DataManagers.DataManagers import set_skip_mysql_query

set_skip_mysql_query(True)  # Skip MySQL, use only cache
# ... run queries ...
set_skip_mysql_query(False)  # Re-enable MySQL
```

This provides **277x speedup** when cache is already populated.

## Performance Gains

### Single Ticker (60 days)
- **Uncached**: 7.41s
- **Cached**: 0.99s
- **Speedup**: 7.5x

### 10 Tickers (60 days each)
- **Uncached**: 93.38s
- **Cached**: 7.17s
- **Speedup**: 13.0x

### 40 Tickers (60 days each)
- **Uncached**: 478.90s (~8 minutes)
- **Cached**: 28.89s (~29 seconds)
- **Speedup**: 16.6x (94% time reduction)

## Code Location
```
module_test/raw_code/DataManagers/DataManagers_cached.py
```

## Usage Example

```python
from DataManagers.DataManagers_cached import CachedOptionDataManager

# Initialize (same API as original)
dm = CachedOptionDataManager('AAPL20250620C250')

# First query (cache MISS) - fetches and caches
greeks = dm.get_timeseries('2024-01-01', '2024-12-31', type_='greeks', model='bs')
# Time: ~7.41s

# Second query (cache HIT) - retrieves from cache
greeks = dm.get_timeseries('2024-01-01', '2024-12-31', type_='greeks', model='bs')
# Time: ~0.99s (7.5x faster!)

# Partial query uses cached data for overlap
greeks = dm.get_timeseries('2024-07-01', '2024-12-31', type_='greeks', model='bs')
# Uses cached Jan-Jun, fetches Jul-Dec only
```

## Next: Interaction Between Original and Cached

See `3_INTERACTION.md` for details on how the cached and original managers work together.
