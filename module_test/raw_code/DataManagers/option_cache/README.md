# Option Data Caching System

A high-performance caching layer for option data management that delivers **16x+ speedup** for repeated queries.

## Overview

The caching system intercepts data queries at three critical bottlenecks:
1. **CachedSpotManager** - Caches ThetaData spot queries
2. **CachedVolManager** - Caches implied volatility calculations  
3. **CachedGreeksManager** - Caches greeks calculations

## Architecture

### Factor-Level Caching
```
CachedOptionDataManager
├── CachedSpotManager (intercepts ThetaData queries)
├── CachedVolManager (intercepts IV calculations)
└── CachedGreeksManager (intercepts greeks calculations)
```

### Cascade Caching
- **Greeks query** → checks greeks cache → checks vol cache → checks spot cache
- Reuses cached data at each level
- Only fetches missing data

### Column Normalization
- **Save**: Lowercase column names (`delta`, `gamma`, `spot`)
- **Load**: Capitalized column names (`Delta`, `Gamma`, `Spot`)
- Prevents column duplication issues

## Usage

### Basic Usage
```python
from DataManagers.DataManagers_cached import CachedOptionDataManager

# Initialize with option ticker
dm = CachedOptionDataManager('AAPL20250620C250')

# First query (cache MISS) - fetches and caches data
data = dm.get_timeseries('2024-01-01', '2024-12-31', 
                         type_='greeks', model='bs')

# Second query (cache HIT) - retrieves from cache (16x faster)
data = dm.get_timeseries('2024-01-01', '2024-12-31', 
                         type_='greeks', model='bs')
```

### Advanced Options
```python
# Disable caching for specific instance
dm = CachedOptionDataManager('AAPL20250620C250', enable_cache=False)

# Disable caching globally
CachedOptionDataManager.disable_caching()

# Re-enable caching globally
CachedOptionDataManager.enable_caching()

# Use MarketTimeseries for underlier data
dm = CachedOptionDataManager('AAPL20250620C250', 
                             use_market_timeseries=True)
```

## Performance

### Benchmark Results (40 tickers, 60 days each)
- **Uncached**: 478.90s (~8 minutes)
- **Cached**: 28.89s (~29 seconds)
- **Speedup**: 16.6x faster (94.0% time reduction)
- **Cache hit rate**: 100%

### Per-Ticker Performance
- Average uncached: 11.97s
- Average cached: 0.72s
- Speedup range: 87-97%

## Cache Storage

### Location
```
$WORK_DIR/.cache/data_manager_cache/
├── dm_option_spot.db       # Spot data cache
├── dm_bs_vol.db            # Black-Scholes vol cache
├── dm_bs_greeks.db         # Black-Scholes greeks cache
├── dm_binomial_vol.db      # Binomial vol cache
└── dm_binomial_greeks.db   # Binomial greeks cache
```

### Cache Management
```python
from DataManagers.option_cache.helpers import load_option_data_cache

# Load cache instances
spot_cache, bs_vol, bs_greeks, binom_vol, binom_greeks = load_option_data_cache()

# Clear specific cache
bs_greeks.clear()

# Clear all caches
for cache in [spot_cache, bs_vol, bs_greeks, binom_vol, binom_greeks]:
    cache.clear()
```

## Features

### ✅ Implemented
- Factor-level caching (spot, vol, greeks)
- Cascade caching (greeks → vol → spot)
- Model-aware caching (BS vs binomial)
- Column normalization (lowercase storage)
- Partial cache handling (fills gaps)
- MySQL query skipping for performance
- MarketTimeseries integration for underliers

### 🔧 Configuration
- Only caches `interval='1d'` data
- Expires after 200 days (configurable)
- Thread-safe SQLite backend
- Automatic cache key generation

## Testing

### Run Performance Tests
```bash
cd DataManagers/option_cache/tests

# Multi-ticker performance test (40 tickers)
python test_multi_ticker_performance.py

# Single ticker comparison
python test_performance_comparison.py

# Partial cache scenarios
python test_partial_cache.py
```

### Test Coverage
- ✅ Single ticker performance
- ✅ Multi-ticker scalability (40 tickers)
- ✅ Partial cache handling
- ✅ Cascade caching efficiency
- ✅ Column normalization
- ✅ Gap detection and filling

## Design Decisions

### Why Factor-Level Caching?
Factor-level caching intercepts at the actual bottlenecks rather than the orchestrator level:
- **Spot queries** take ~5-10s (ThetaData API)
- **Vol calculations** take ~2-5s (numerical solvers)
- **Greeks calculations** take ~5-15s (multiple derivatives)

By caching at each factor, we:
1. Avoid redundant API calls
2. Skip expensive calculations
3. Enable cascade reuse (greeks uses cached vol + spot)

### Why Column Normalization?
Original issue: `'Openinterest'` vs `'openinterest'` caused:
- Column duplication
- Cache key mismatches
- Recursion errors

Solution: Lowercase on save, capitalize on load
- Ensures consistency
- Matches parent class output format
- Prevents accumulation of duplicate columns

### Why No Today Data?
Intraday data is incomplete and can change:
- Market is still open
- Data may be partial
- Could cache stale values

Solution: Optional `drop_today` parameter filters today's data before caching

## Files

### Core Module
```
option_cache/
├── __init__.py           # Package exports
├── helpers.py            # Cache loading utilities  
├── utils.py              # Cache operations (get, save, check)
├── README.md             # This file
└── tests/                # Test suite
    ├── test_multi_ticker_performance.py
    ├── test_performance_comparison.py
    ├── test_partial_cache.py
    └── TEST_SUMMARY.md
```

### Main Files
- `DataManagers_cached.py` - Main cached manager classes
- `option_cache/utils.py` - Cache operations
- `option_cache/helpers.py` - Cache initialization

## Troubleshooting

### Cache Not Working
1. Check if caching is enabled: `CachedOptionDataManager._CACHE_ENABLED`
2. Verify interval is '1d': Only daily data is cached
3. Check cache location: `$WORK_DIR/.cache/data_manager_cache/`

### Stale Data
```python
# Clear cache for specific ticker
from DataManagers.option_cache.helpers import load_option_data_cache
spot_cache, bs_vol, bs_greeks, binom_vol, binom_greeks = load_option_data_cache()

# Clear greeks cache
bs_greeks.clear()
```

### Performance Not Improving
1. Ensure `set_skip_mysql_query(True)` is called
2. Check if MarketTimeseries is available for underlier caching
3. Verify cache hits with logging: `logger.setLevel('INFO')`

## Future Enhancements

- [ ] Intraday caching (5min, 1min intervals)
- [ ] Intelligent cache warming
- [ ] Cache compression for large datasets
- [ ] Distributed caching for multi-process environments
- [ ] Cache statistics and monitoring dashboard

## References

- **Performance Test Results**: `tests/TEST_SUMMARY.md`
- **Architecture Documentation**: `../00_docs/COMPLETION_SUMMARY.md`
- **Cache Flow Diagram**: `../00_docs/CACHE_FLOW_DIAGRAM.md`

---

**Author**: DataManagers Cache Team  
**Last Updated**: November 29, 2025  
**Version**: 1.0.0
