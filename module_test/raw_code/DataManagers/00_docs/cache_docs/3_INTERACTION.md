# Interaction Between Original and Cached DataManagers

## Overview

The cached and original managers work together through **delegation and interception**. The cached manager inherits from the original and selectively overrides methods to add caching logic, while delegating actual computation to the original manager on cache misses.

## Class Relationship

```python
class CachedOptionDataManager(OptionDataManager):
    """Cached version inherits from original."""
    
    def __init__(self, *args, enable_cache=True, **kwargs):
        # Call parent constructor
        super().__init__(*args, **kwargs)
        
        # Load cache instances
        self._caches = load_option_data_cache()
        
        # Replace child managers with cached versions
        self.spot_manager = CachedSpotManager(self.symbol, self._caches, self.opttick)
        self.vol_manager = CachedVolManager(self.symbol, self._caches, self.opttick)
        self.greeks_manager = CachedGreeksManager(self.symbol, self._caches, self.opttick)
```

**Key Points:**
1. `CachedOptionDataManager` **IS AN** `OptionDataManager` (inheritance)
2. Child managers are replaced with cached versions
3. All other methods and functionality remain unchanged

## Interception Points

The cached managers intercept at three specific methods:

### 1. Spot Query Interception

**Original Flow:**
```
OptionDataManager.get_timeseries()
    ↓
SpotDataManager.query_thetadata()
    ↓
ThetaData API
```

**Cached Flow:**
```
CachedOptionDataManager.get_timeseries()
    ↓
CachedSpotManager.query_thetadata()
    ├─ Check spot cache
    ├─ If HIT: return cached data
    └─ If MISS:
        ↓
        SpotDataManager.query_thetadata()  ← Delegates to original
        ↓
        ThetaData API
        ↓
        Save to cache
```

### 2. Vol Calculation Interception

**Original Flow:**
```
VolDataManager.calculate_iv()
    ↓
Calculate IV from spot prices + underlier data
    ↓
Modify data_request.raw_spot_data in-place
```

**Cached Flow:**
```
CachedVolManager.calculate_iv()
    ├─ Check vol cache
    ├─ If HIT: add cached vol columns to raw_spot_data
    └─ If MISS:
        ↓
        VolDataManager.calculate_iv()  ← Delegates to original
        ↓
        Calculate IV
        ↓
        Save vol columns to cache
```

### 3. Greeks Calculation Interception

**Original Flow:**
```
GreeksDataManager.calculate_greeks()
    ↓
Calculate greeks from vol + spot + underlier data
    ↓
Modify data_request.raw_spot_data in-place
```

**Cached Flow:**
```
CachedGreeksManager.calculate_greeks()
    ├─ Check greeks cache
    ├─ If HIT: add cached greek columns to raw_spot_data
    └─ If MISS:
        ↓
        GreeksDataManager.calculate_greeks()  ← Delegates to original
        ↓
        Calculate greeks
        ↓
        Save greek columns to cache
```

## Data Flow Example: Complete Request

Let's trace a greeks request through both systems:

### Scenario: Request greeks for 60 days (30 cached, 30 missing)

```
USER: dm.get_timeseries('2024-01-01', '2024-02-29', type_='greeks')
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ CachedOptionDataManager.get_timeseries()                        │
│ - Inherits from OptionDataManager.get_timeseries()             │
│ - No override, uses parent's implementation                     │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ OptionDataManager.get_timeseries() [PARENT]                    │
│ - Creates QueryRequestParameter                                 │
│ - Checks database for existing data                             │
│ - Calls greeks_manager.calculate_greeks()                       │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ CachedGreeksManager.calculate_greeks()                          │
│ - Check greeks cache                                            │
│ - Found: Jan 1-30 cached ✓                                     │
│ - Missing: Jan 31 - Feb 29                                      │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
                  ┌──────────┴──────────┐
                  │  CACHE PARTIAL      │
                  └──────────┬──────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ CachedGreeksManager: Need to calculate greeks for Jan 31-Feb 29│
│ → Calls GreeksDataManager.calculate_greeks() [PARENT]          │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ GreeksDataManager.calculate_greeks() [PARENT]                  │
│ - Need vol data first                                           │
│ - Calls vol_manager.calculate_iv()                              │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ CachedVolManager.calculate_iv()                                 │
│ - Check vol cache                                               │
│ - Found: Jan 1-30 cached ✓                                     │
│ - Missing: Jan 31 - Feb 29                                      │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
                  ┌──────────┴──────────┐
                  │  CACHE PARTIAL      │
                  └──────────┬──────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ CachedVolManager: Need to calculate vol for Jan 31-Feb 29      │
│ → Calls VolDataManager.calculate_iv() [PARENT]                 │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ VolDataManager.calculate_iv() [PARENT]                         │
│ - Need spot data first                                          │
│ - Calls spot_manager.query_thetadata()                          │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ CachedSpotManager.query_thetadata()                             │
│ - Check spot cache                                              │
│ - Found: Jan 1-30 cached ✓                                     │
│ - Missing: Jan 31 - Feb 29                                      │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
                  ┌──────────┴──────────┐
                  │  CACHE PARTIAL      │
                  └──────────┬──────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ CachedSpotManager: Need to query spot for Jan 31-Feb 29        │
│ → Calls SpotDataManager.query_thetadata() [PARENT]             │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ SpotDataManager.query_thetadata() [PARENT]                     │
│ - Query ThetaData API for Jan 31-Feb 29                        │
│ - Return spot data                                              │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ CachedSpotManager: Save Jan 31-Feb 29 spot to cache            │
│ - Return merged data (Jan 1-Feb 29)                            │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ VolDataManager.calculate_iv() [PARENT]                         │
│ - Calculate IV for Jan 31-Feb 29                               │
│ - Add vol columns to raw_spot_data                             │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ CachedVolManager: Save Jan 31-Feb 29 vol to cache              │
│ - Merge Jan 1-30 cached + Jan 31-Feb 29 new                    │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ GreeksDataManager.calculate_greeks() [PARENT]                  │
│ - Calculate greeks for Jan 31-Feb 29                           │
│ - Add greek columns to raw_spot_data                           │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ CachedGreeksManager: Save Jan 31-Feb 29 greeks to cache        │
│ - Merge Jan 1-30 cached + Jan 31-Feb 29 new                    │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ CachedOptionDataManager: Return complete result                │
│ - raw_spot_data contains: spot + vol + greeks                  │
│ - All data from Jan 1-Feb 29 (60 days)                         │
└─────────────────────────────────────────────────────────────────┘
```

## Key Interaction Patterns

### 1. Method Override with Delegation
```python
class CachedVolManager:
    def __init__(self, symbol, caches, opttick):
        # Create parent manager
        self._parent_manager = VolDataManager(symbol)
    
    def calculate_iv(self, **kwargs):
        # Check cache
        cached = get_cached_data(...)
        if cache_hit:
            return use_cache()
        
        # Delegate to parent
        self._parent_manager.calculate_iv(**kwargs)  # Original logic runs
        
        # Save result to cache
        save_to_cache(...)
```

### 2. In-Place Modification Preservation
The parent managers modify `data_request.raw_spot_data` in-place. The cached managers preserve this behavior:

```python
# Parent modifies in-place
def calculate_iv(self, **kwargs):
    data_request = kwargs['data_request']
    data_request.raw_spot_data['iv'] = calculated_iv
    # Returns None

# Cached manager preserves in-place modification
def calculate_iv(self, **kwargs):
    if cache_hit:
        # Also modify in-place
        data_request = kwargs['data_request']
        data_request.raw_spot_data['iv'] = cached_iv
        return
    
    # Let parent modify in-place
    self._parent_manager.calculate_iv(**kwargs)
```

### 3. Transparent Caching
From the user's perspective, the API is identical:

```python
# Original
dm = OptionDataManager('AAPL20250620C250')
data = dm.get_timeseries('2024-01-01', '2024-12-31', type_='greeks')

# Cached (same API)
dm = CachedOptionDataManager('AAPL20250620C250')
data = dm.get_timeseries('2024-01-01', '2024-12-31', type_='greeks')
```

The only difference is performance - the cached version is 7-16x faster on subsequent queries.

## Advantages of This Design

1. **Minimal Code Changes**: Original manager unchanged
2. **Backward Compatible**: Can switch between original and cached seamlessly
3. **Composable**: Cache layers can be enabled/disabled independently
4. **Maintainable**: Cache logic separate from business logic
5. **Testable**: Can test original and cached managers independently

## Next: Cache Mechanism

See `4_CACHE_MECHANISM.md` for details on how the cache storage works.
