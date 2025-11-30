# Original DataManager

## Overview

The original `OptionDataManager` is the core data management system for option pricing and analysis. It handles fetching, processing, and storing option data from various sources (primarily ThetaData API).

## Class Hierarchy

```
OptionDataManager (Parent)
├── SpotDataManager (Child)
├── VolDataManager (Child)
├── GreeksDataManager (Child)
└── AttributionDataManager (Child)
```

## Core Responsibilities

### 1. Data Retrieval
- **ThetaData API Integration**: Fetches option quotes, OHLC, open interest
- **Database Integration**: Queries MySQL for previously saved data
- **Smart Querying**: Only fetches missing dates, not entire ranges

### 2. Data Processing
- **Spot Data**: Raw option prices, bid/ask, volume, open interest
- **Implied Volatility**: Calculates IV using Black-Scholes or Binomial models
- **Greeks**: Computes Delta, Gamma, Vega, Theta, Rho, Vanna, Volga
- **Attribution**: Analyzes PnL attribution across factors

### 3. Data Storage
- **MySQL Database**: Persists processed data for future use
- **Scheduling**: Queues large data requests for background processing
- **Validation**: Checks for data completeness and integrity

## Key Methods

### `get_timeseries(start, end, interval='1d', type_='spot', model='bs', extra_cols=None)`
**Main entry point** for data retrieval.

**Parameters:**
- `start`, `end`: Date range
- `interval`: Time granularity ('1d' for daily, '1h' for hourly, etc.)
- `type_`: Data type ('spot', 'vol', 'greeks', 'attribution')
- `model`: Pricing model ('bs' for Black-Scholes, 'binomial')
- `extra_cols`: Additional columns to include

**Returns:**
- `Request` object containing:
  - `raw_spot_data`: DataFrame with all requested data
  - `input_params`: Dictionary of input parameters used
  - `metadata`: Processing information

**Flow:**
```
1. Create QueryRequestParameter with request details
2. Check database for existing data
3. If missing dates:
   a. Query ThetaData for missing dates
   b. Process data (IV, greeks, etc.)
   c. Save to database
   d. Merge with cached data
4. Return complete dataset
```

### Child Managers

#### `SpotDataManager.query_thetadata(start, end, **kwargs)`
- Fetches raw option spot data from ThetaData
- Returns DataFrame with columns: Date, Bid, Ask, Last, Volume, OpenInterest, etc.

#### `VolDataManager.calculate_iv(type_, **kwargs)`
- Calculates implied volatility from spot prices
- Uses Black-Scholes or Binomial model
- Requires underlier data (spot, dividends, risk-free rate)

#### `GreeksDataManager.calculate_greeks(type_, **kwargs)`
- Computes option greeks from vol data
- Calculates both standard and midpoint greeks
- Returns Delta, Gamma, Vega, Theta, Rho, Vanna, Volga

## Data Dependencies

### Underlier Data Required
When calculating vol or greeks, the manager needs:

1. **Underlier Spot Price (s0)**: Current stock price
2. **Dividends (y)**: Dividend yield
3. **Risk-Free Rate (r)**: Treasury rate

These are fetched via:
```python
# Lazy-loaded through EODData object
input_params = self.eod  # or self.intra for intraday
s0 = input_params['s0_chain']['close']  # Triggers lazy load if not cached
```

### Lazy Loading Mechanism
The `EODData` and `IntraData` objects use `__getitem__` to lazy-load underlier data:

```python
class EODData:
    def __getitem__(self, key):
        if key not in self._eod:
            self._lazy_load(key)  # Fetches from Stock class
        return self._eod[key]
```

When accessed, the Stock class checks:
1. MarketTimeseries cache (if available)
2. Database
3. ThetaData API (last resort)

## Performance Characteristics

### Without Caching
- **Typical query**: 2-3 seconds for 60 days
- **API calls**: 
  - 1 call for option spot
  - 1 call for underlier spot
  - 1 call for dividends
  - 1 call for risk-free rate
- **Bottlenecks**:
  - ThetaData API latency (~500ms per call)
  - IV calculation (CPU intensive)
  - Greeks calculation (CPU intensive)

### Database Caching
- MySQL caches **processed** data (vol, greeks)
- Does NOT cache raw ThetaData responses
- Partial range queries still require full API calls

## Limitations

1. **No Incremental Queries**: Fetching missing dates still requires separate API calls
2. **No Factor-Level Caching**: Each data type (spot, vol, greeks) fetched independently
3. **No Underlier Caching**: Underlier data refetched for each option query
4. **Database Overhead**: MySQL queries can be slow for large datasets

## Code Location
```
module_test/raw_code/DataManagers/DataManagers.py
```

## Usage Example

```python
from DataManagers.DataManagers import OptionDataManager

# Initialize with option ticker
dm = OptionDataManager('AAPL20250620C250')

# Fetch spot data
spot = dm.get_timeseries('2024-01-01', '2024-12-31', type_='spot')

# Fetch vol data (requires underlier data)
vol = dm.get_timeseries('2024-01-01', '2024-12-31', type_='vol', model='bs')

# Fetch greeks (requires vol data)
greeks = dm.get_timeseries('2024-01-01', '2024-12-31', type_='greeks', model='bs')

# Access results
df = spot.raw_spot_data
print(df[['Bid', 'Ask', 'Last', 'Volume']])
```

## Next: Cached DataManager

The cached implementation addresses these limitations by adding intelligent caching layers at the bottlenecks. See `2_CACHED_DATAMANAGER.md` for details.
