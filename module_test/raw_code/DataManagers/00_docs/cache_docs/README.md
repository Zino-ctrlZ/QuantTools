# Option Cache Documentation

Complete documentation for the option data caching system.

## Overview

The option caching system provides **16x+ speedup** for option data queries through intelligent factor-level caching. It's a drop-in replacement for `OptionDataManager` that caches at three bottlenecks: spot queries, vol calculations, and greeks calculations.

## Quick Start

```python
from DataManagers.DataManagers_cached import CachedOptionDataManager

# Same API as OptionDataManager
dm = CachedOptionDataManager('AAPL20250620C250')

# First query (cache miss) - fetches and caches
greeks = dm.get_timeseries('2024-01-01', '2024-12-31', type_='greeks', model='bs')
# Time: ~7.41s

# Second query (cache hit) - retrieves from cache
greeks = dm.get_timeseries('2024-01-01', '2024-12-31', type_='greeks', model='bs')
# Time: ~0.99s (7.5x faster!)
```

## Documentation Structure

Read the docs in order to understand the complete system:

### 1. [Original DataManager](1_ORIGINAL_DATAMANAGER.md)
**Start here** to understand the base system.

**Topics:**
- Class hierarchy (OptionDataManager, SpotDataManager, VolDataManager, GreeksDataManager)
- Core responsibilities (data retrieval, processing, storage)
- Key methods (`get_timeseries()`, `query_thetadata()`, `calculate_iv()`, `calculate_greeks()`)
- Data dependencies (underlier data requirements)
- Lazy loading mechanism (EODData, IntraData)
- Performance characteristics
- Limitations addressed by caching

**Key Takeaway:** The original manager is powerful but has performance bottlenecks that the cached version addresses.

---

### 2. [Cached DataManager](2_CACHED_DATAMANAGER.md)
**Core caching concepts** and architecture.

**Topics:**
- Architecture overview (factor-level caching)
- Delegation pattern (wrapping original managers)
- Three cached managers (CachedSpotManager, CachedVolManager, CachedGreeksManager)
- Partial cache support (intelligent date range handling)
- Column normalization (preventing duplication)
- Cache configuration (enable/disable)
- Performance benchmarks (7-16x speedup)

**Key Takeaway:** Factor-level caching intercepts at bottlenecks, delegating to original on cache miss.

---

### 3. [Interaction](3_INTERACTION.md)
**How cached and original managers work together.**

**Topics:**
- Class relationship (inheritance)
- Interception points (spot query, vol calculation, greeks calculation)
- Data flow example (complete request trace)
- Method override with delegation
- In-place modification preservation
- Transparent caching (same API)

**Key Takeaway:** The cached manager IS AN original manager with added caching layers - fully backward compatible.

---

### 4. [Cache Mechanism](4_CACHE_MECHANISM.md)
**How cache storage works** (on disk with SQLite).

**Topics:**
- Storage architecture (5 SQLite databases in `$WORK_DIR/.cache/`)
- Database structure (two-level key hierarchy)
- Cache operations (load, get, save, check completeness)
- Cache selection logic (by data type and model)
- Data format (spot, vol, greeks)
- Cache expiry (200-day policy)
- Cache management (clear, statistics)
- Column normalization (lowercase save, capitalize load)

**Key Takeaway:** Persistent disk storage with SQLite provides fast, durable caching across sessions.

---

### 5. [Underlier Caching & Lazy Loading](5_UNDERLIER_CACHING.md)
**How underlier data** (stock prices, dividends) **is cached.**

**Topics:**
- What is underlier data (s0, y, r)
- Lazy loading mechanism (EODData, IntraData)
- When lazy loading is triggered
- Stock class data source (3-tier lookup)
- MarketTimeseries optional cache
- Integration with CachedOptionDataManager
- Global access pattern
- Complete data flow (with underlier caching)
- Three-level cache hierarchy
- Performance impact (2-3x additional speedup)

**Key Takeaway:** Underlier data uses 3-tier caching (MarketTimeseries → EODData → Database/API) with lazy loading for efficiency.

---

## Architecture Diagram

```
CachedOptionDataManager (inherits from OptionDataManager)
│
├─ CachedSpotManager
│  ├─ Wraps: SpotDataManager
│  ├─ Intercepts: query_thetadata()
│  ├─ Cache: dm_option_spot.db
│  └─ Saves: Bid, Ask, Last, Volume, OpenInterest
│
├─ CachedVolManager
│  ├─ Wraps: VolDataManager
│  ├─ Intercepts: calculate_iv()
│  ├─ Cache: dm_bs_vol.db / dm_binomial_vol.db
│  └─ Saves: IV, Ask_IV, Bid_IV, Mid_IV, Last_IV
│
└─ CachedGreeksManager
   ├─ Wraps: GreeksDataManager
   ├─ Intercepts: calculate_greeks()
   ├─ Cache: dm_bs_greeks.db / dm_binomial_greeks.db
   └─ Saves: Delta, Gamma, Vega, Theta, Rho, Vanna, Volga
```

## Cache Flow

```
User Request: Get greeks for 60 days
    ↓
CachedGreeksManager
    ├─ Check greeks cache
    ├─ Found: 30 days cached
    ├─ Missing: 30 days
    │
    ├─ For missing 30 days:
    │  ↓
    │  CachedVolManager
    │  ├─ Check vol cache
    │  ├─ Found: 20 days cached
    │  ├─ Missing: 10 days
    │  │
    │  ├─ For missing 10 days:
    │  │  ↓
    │  │  CachedSpotManager
    │  │  ├─ Check spot cache
    │  │  ├─ Missing: 10 days
    │  │  ├─ Query ThetaData (10 days only)
    │  │  └─ Save to spot cache
    │  │
    │  ├─ Calculate vol (10 days)
    │  └─ Save to vol cache
    │
    ├─ Calculate greeks (30 days)
    └─ Save to greeks cache
    ↓
Return complete greeks (60 days)
```

## Performance Summary

| Scenario | Time | Speedup |
|----------|------|---------|
| Single ticker (60 days) | 0.99s vs 7.41s | 7.5x |
| 10 tickers (60 days each) | 7.17s vs 93.38s | 13.0x |
| 40 tickers (60 days each) | 28.89s vs 478.90s | **16.6x** |

**Total time reduction: 94%**

## Cache Locations (On Disk)

```
$WORK_DIR/.cache/data_manager_cache/
├── dm_option_spot.db        # Raw spot data
├── dm_bs_vol.db             # Black-Scholes IV
├── dm_bs_greeks.db          # Black-Scholes greeks
├── dm_binomial_vol.db       # Binomial IV
└── dm_binomial_greeks.db    # Binomial greeks
```

Default: `~/.cache/data_manager_cache/`

## Additional Resources

- **Module README**: `../README.md` - Quick reference and usage examples
- **Test Summary**: `../tests/TEST_SUMMARY.md` - Performance benchmarks
- **Code**: `../../DataManagers_cached.py` - Implementation
- **Cache Utils**: `../utils.py` - Cache operations
- **Cache Helpers**: `../helpers.py` - Cache loading

## Support

For questions or issues:
1. Check the relevant documentation section above
2. Review the module README (`../README.md`)
3. Examine test files in `../tests/` for usage examples
