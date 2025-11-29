# Portfolio.analyze_positions() Performance Analysis
## Deep Dive into _create_ctx() and Nested Functions

**Baseline Performance:**
- Average: 198.66 ms
- Std Dev: 54.69 ms  
- CV: 27.53% (HIGH - indicates caching/I/O variability)
- Range: 167.92 - 565.01 ms

---

## Code Flow Analysis

### 1. `_create_ctx(date)` - Main Method (EventDriven/new_portfolio.py:540-605)

**Current Implementation:**
```python
def _create_ctx(self, date: pd.Timestamp) -> PositionAnalysisContext:
    positions = self.current_positions
    positions_states = []
    
    # LOOP OVER POSITIONS (Lines 549-571)
    for tick, pos_pack in positions.items():
        for signal_id, position in pos_pack.items():
            trade_id = position["position"]["trade_id"]
            qty = position["quantity"]
            entry_price = position["entry_price"] / qty
            
            # CALL 1: Get position market data (172 ms avg observed)
            current_position_data = self.risk_manager.market_data.get_at_time_position_data(
                position_id=trade_id, date=date
            )
            
            # CALL 2: Get underlier market data (26 ms avg observed)
            current_underlier_data = self.risk_manager.market_data.market_timeseries.get_at_index(
                sym=tick, index=date
            )
            
            current_price = position["market_value"] / qty
            pnl = (current_price - entry_price) * qty
            
            # CREATE PYDANTIC DATACLASS
            pos_state = PositionState(...)
            positions_states.append(pos_state)
    
    # AGGREGATE OPERATIONS (Lines 573-583)
    cash = sum(self.allocated_cash_map.values())
    pnl = sum([x.pnl for x in positions_states])
    total_value = cash + pnl
    
    # CREATE DATACLASSES (Lines 585-603)
    portfolio_state = PortfolioState(...)
    meta = PortfolioMetaInfo(...)
    ctx = PositionAnalysisContext(...)
    
    return ctx
```

---

## Bottleneck Analysis

### CRITICAL PATH #1: `get_at_time_position_data()` (~87% of time)
**File:** EventDriven/riskmanager/market_timeseries.py:117-147

```python
def get_at_time_position_data(self, position_id: str, date: Union[datetime, str]) -> AtTimePositionData:
    # ISSUE 1: Calls get_position_data which does dict lookup
    position_data = self.get_position_data(position_id)  # Line 121
    
    if position_data.empty:
        logger.critical(f"Position data for {position_id} not found in cache.")
        return None
    
    # ISSUE 2: String conversion on EVERY call
    date = pd.to_datetime(date).strftime("%Y-%m-%d")  # Line 125
    
    # ISSUE 3: Index lookup (not optimized)
    if date not in position_data.index:  # Line 126
        logger.critical(f"Date {date} not found in position data for {position_id}.")
        return None
    
    # ISSUE 4: DataFrame .loc[] access
    row = position_data.loc[date]  # Line 129
    
    # ISSUE 5: Dict building in loop
    skips = {}
    for col in self._skip_calc_config.skip_columns:  # Lines 130-134
        skips[col] = {
            "skip_day": row.get(f"{col}_skip_day", False),
            "skip_day_count": row.get(f"{col}_skip_day_count", 0),
        }
    
    # ISSUE 6: Creating dataclass with many .get() calls
    return AtTimePositionData(
        position_id=position_id,
        date=date,
        close=row["Midpoint"],
        bid=row["Closebid"],
        skips=skips,
        ask=row["Closeask"],
        midpoint=row["Midpoint"],
        delta=row.get("Delta", np.nan),  # Lines 138-145
        gamma=row.get("Gamma", np.nan),
        vega=row.get("Vega", np.nan),
        theta=row.get("Theta", np.nan),
    )
```

**Nested Call:** `get_position_data()` (Line 85)
```python
def get_position_data(self, position_id: str) -> pd.DataFrame:
    # Simple dict lookup - FAST
    return self.position_data_cache.get(position_id, pd.DataFrame())
```

---

### CRITICAL PATH #2: `get_at_index()` (~13% of time)
**File:** EventDriven/riskmanager/market_data.py:224-256

```python
def get_at_index(self, sym: str, index: pd.Timestamp, interval: str = '1d') -> AtIndexResult:
    # ISSUE 7: Checking if already loaded
    already_available = self.already_loaded(sym, interval)  # Line 233
    
    if not already_available:
        print("Reloading timeseries data for symbol %s.", sym)
        self.load_timeseries(...)  # Heavy operation
        self._last_refresh = ny_now()
    
    # ISSUE 8: Type checking and conversion EVERY call
    if isinstance(index, (str, datetime)):  # Lines 245-246
        index = pd.Timestamp(index)
    
    # ISSUE 9: Dict lookup check
    if sym not in self._spot.get(interval, {}):  # Line 247
        raise ValueError(f"Symbol {sym} not found in timeseries data.")
    
    # ISSUE 10: Type checking AGAIN
    if not (isinstance(index, pd.Timestamp) or isinstance(index, datetime)):  # Line 249
        raise ValueError("Index must be a pandas Timestamp or datetime object.")
    
    # ISSUE 11: String conversion
    index = index.strftime('%Y-%m-%d')  # Line 250
    
    # ISSUE 12: Multiple conditional dict lookups with ternary operators
    spot = self._spot[interval][sym].loc[index] if sym in self._spot.get(interval, {}) else None
    chain_spot = self._chain_spot[interval][sym].loc[index] if sym in self._chain_spot.get(interval, {}) else None
    dividends = self._dividends[interval][sym].loc[index] if sym in self._dividends.get(interval, {}) else None
    rates = self.rates.loc[index] if self.rates is not None else None
    
    # ISSUE 13: Creating dataclass
    return AtIndexResult(spot=spot, chain_spot=chain_spot, dividends=dividends, sym=sym, date=index, rates=rates)
```

---

## Optimization Opportunities (Priority Order)

### **TASK #1: Cache date string conversions** ⭐⭐⭐
**Impact:** HIGH (repeated on every call)
**Complexity:** LOW

**Problem:**
```python
# Called in get_at_time_position_data (Line 125)
date = pd.to_datetime(date).strftime("%Y-%m-%d")

# Called in get_at_index (Line 250)
index = index.strftime('%Y-%m-%d')
```

**Solution:**
```python
# In _create_ctx, convert once before loop
date_str = date.strftime('%Y-%m-%d')

# Pass pre-formatted string to both functions
# OR add internal cache: _date_str_cache = {}
```

**Expected Speedup:** 5-10% (removes repeated pd.to_datetime + strftime)

---

### **TASK #2: Pre-fetch position data dict** ⭐⭐⭐
**Impact:** HIGH (reduces function call overhead)
**Complexity:** LOW

**Problem:**
```python
# Currently in loop:
for tick, pos_pack in positions.items():
    for signal_id, position in pos_pack.items():
        current_position_data = self.risk_manager.market_data.get_at_time_position_data(...)
```

**Solution:**
```python
# Before loop: batch fetch all position data
position_data_cache = {}
for tick, pos_pack in positions.items():
    for signal_id, position in pos_pack.items():
        trade_id = position["position"]["trade_id"]
        position_data_cache[trade_id] = self.risk_manager.market_data.position_data_cache.get(
            trade_id, pd.DataFrame()
        )

# Then in loop: direct dict access instead of function call
for tick, pos_pack in positions.items():
    for signal_id, position in pos_pack.items():
        trade_id = position["position"]["trade_id"]
        position_df = position_data_cache[trade_id]
        if not position_df.empty and date_str in position_df.index:
            row = position_df.loc[date_str]
            # Build AtTimePositionData directly
```

**Expected Speedup:** 15-25% (eliminates function call overhead + repeated lookups)

---

### **TASK #3: Eliminate redundant type checks in get_at_index** ⭐⭐⭐
**Impact:** MEDIUM (called once per position per date)
**Complexity:** LOW

**Problem:**
```python
# Lines 245-250: Three separate checks/conversions
if isinstance(index, (str, datetime)):
    index = pd.Timestamp(index)
# ...
if not (isinstance(index, pd.Timestamp) or isinstance(index, datetime)):
    raise ValueError(...)
index = index.strftime('%Y-%m-%d')
```

**Solution:**
```python
# Single conversion path
if not isinstance(index, pd.Timestamp):
    index = pd.Timestamp(index)
index_str = index.strftime('%Y-%m-%d')
```

**Expected Speedup:** 3-5% (reduces redundant isinstance checks)

---

### **TASK #4: Optimize skip column dict building** ⭐⭐
**Impact:** MEDIUM (nested dict construction per position)
**Complexity:** LOW

**Problem:**
```python
# Lines 130-134 in get_at_time_position_data
skips = {}
for col in self._skip_calc_config.skip_columns:
    skips[col] = {
        "skip_day": row.get(f"{col}_skip_day", False),
        "skip_day_count": row.get(f"{col}_skip_day_count", 0),
    }
```

**Solution A - Dict Comprehension:**
```python
skips = {
    col: {
        "skip_day": row.get(f"{col}_skip_day", False),
        "skip_day_count": row.get(f"{col}_skip_day_count", 0),
    }
    for col in self._skip_calc_config.skip_columns
}
```

**Solution B - Pre-compute column names:**
```python
# In __init__:
self._skip_col_names = [
    (col, f"{col}_skip_day", f"{col}_skip_day_count") 
    for col in self._skip_calc_config.skip_columns
]

# In method:
skips = {
    col: {
        "skip_day": row.get(skip_day_col, False),
        "skip_day_count": row.get(skip_count_col, 0),
    }
    for col, skip_day_col, skip_count_col in self._skip_col_names
}
```

**Expected Speedup:** 2-3% (reduces string formatting overhead)

---

### **TASK #5: Use .get() with default for row access** ⭐⭐
**Impact:** LOW-MEDIUM (safer + potentially faster)
**Complexity:** LOW

**Problem:**
```python
# Current approach mixes direct access and .get()
close=row["Midpoint"],  # KeyError risk
delta=row.get("Delta", np.nan),  # Safe
```

**Solution:**
```python
# Consistent approach using .get() throughout
close=row.get("Midpoint", np.nan),
bid=row.get("Closebid", np.nan),
ask=row.get("Closeask", np.nan),
midpoint=row.get("Midpoint", np.nan),
delta=row.get("Delta", np.nan),
# ... etc
```

**Expected Speedup:** 1-2% (eliminates KeyError checks, more predictable)

---

### **TASK #6: Inline simple calculations** ⭐
**Impact:** LOW (micro-optimization)
**Complexity:** LOW

**Problem:**
```python
# Lines 552-555 in _create_ctx
qty = position["quantity"]
entry_price = position["entry_price"] / qty
# ...
current_price = position["market_value"] / qty
pnl = (current_price - entry_price) * qty
```

**Solution:**
```python
# Combine operations to reduce temporary variables
qty = position["quantity"]
pnl = position["market_value"] - position["entry_price"]
entry_price = position["entry_price"] / qty
current_price = position["market_value"] / qty
```

**Expected Speedup:** <1% (minor)

---

### **TASK #7: Use list comprehension for pnl sum** ⭐
**Impact:** LOW (already pretty fast)
**Complexity:** LOW

**Problem:**
```python
# Line 575
pnl = sum([x.pnl for x in positions_states])
```

**Solution:**
```python
# Remove unnecessary list creation
pnl = sum(x.pnl for x in positions_states)
```

**Expected Speedup:** <1% (minor, but cleaner)

---

### **TASK #8: Cache PortfolioMetaInfo** ⭐
**Impact:** LOW (only created once per call, but unnecessary)
**Complexity:** LOW

**Problem:**
```python
# Lines 586-593: Created on EVERY call with same data
meta = PortfolioMetaInfo(
    portfolio_name="bkt_test_11",
    initial_cash=self.initial_capital,
    start_date=self.risk_manager.start_date,
    end_date=self.risk_manager.end_date,
    t_plus_n=self.t_plus_n,
    is_backtest=True,
)
```

**Solution:**
```python
# In OptionSignalPortfolio.__init__:
self._cached_portfolio_meta = PortfolioMetaInfo(
    portfolio_name="bkt_test_11",
    initial_cash=self.initial_capital,
    start_date=self.risk_manager.start_date,
    end_date=self.risk_manager.end_date,
    t_plus_n=self.t_plus_n,
    is_backtest=True,
)

# In _create_ctx:
ctx = PositionAnalysisContext(
    date=date,
    portfolio=portfolio_state,
    portfolio_meta=self._cached_portfolio_meta,
)
```

**Expected Speedup:** <1% (reduces dataclass creation overhead)

---

### **TASK #9: Optimize get_at_index dict lookups** ⭐⭐
**Impact:** MEDIUM (called once per position)
**Complexity:** MEDIUM

**Problem:**
```python
# Lines 251-254: Redundant sym checks
spot = self._spot[interval][sym].loc[index] if sym in self._spot.get(interval, {}) else None
chain_spot = self._chain_spot[interval][sym].loc[index] if sym in self._chain_spot.get(interval, {}) else None
dividends = self._dividends[interval][sym].loc[index] if sym in self._dividends.get(interval, {}) else None
```

**Solution:**
```python
# Check once, extract all
interval_data = self._spot.get(interval, {})
if sym in interval_data:
    spot = interval_data[sym].loc[index_str]
    chain_spot = self._chain_spot[interval][sym].loc[index_str]
    dividends = self._dividends[interval][sym].loc[index_str]
else:
    spot = chain_spot = dividends = None

rates = self.rates.loc[index_str] if self.rates is not None else None
```

**Expected Speedup:** 3-5% (reduces dict.get() calls)

---

### **TASK #10: Consider batch DataFrame .loc access** ⭐⭐⭐
**Impact:** POTENTIALLY HIGH (if multiple positions)
**Complexity:** HIGH

**Problem:**
```python
# Current: Individual .loc[] per position
for position in positions:
    row = position_df.loc[date_str]
```

**Solution (Advanced):**
```python
# If multiple positions share same underlying:
# Extract all needed rows at once using .loc with list
dates_needed = [date_str]  # Could be multiple dates in future optimization
rows = position_df.loc[dates_needed]
```

**Expected Speedup:** 10-20% (if processing multiple positions/dates)
**Note:** Requires refactoring to batch process positions

---

## Summary: Optimization Priority

### Immediate Quick Wins (Implement First):
1. **Task #1**: Cache date string conversions (5-10%)
2. **Task #2**: Pre-fetch position data (15-25%)
3. **Task #3**: Eliminate redundant type checks (3-5%)
4. **Task #4**: Optimize skip dict building (2-3%)

**Combined Expected Speedup: 25-43%**

### Secondary Optimizations:
5. Task #5: Consistent .get() usage (1-2%)
6. Task #7: Generator for sum (< 1%)
7. Task #8: Cache PortfolioMetaInfo (<1%)
8. Task #9: Optimize get_at_index lookups (3-5%)

**Additional Speedup: 5-8%**

### Advanced (Larger Refactor):
9. Task #10: Batch DataFrame access (10-20% potential)

---

## Variance Analysis (CV: 27.53%)

**High variance indicates:**
1. **First-run cache warming** - First call loads data, subsequent calls hit cache
2. **DataFrame .loc[] performance variability** - Pandas indexing can vary
3. **Dict lookups** - Python dict resizing during execution
4. **Garbage collection** - Pydantic dataclass creation

**Strategies to Reduce Variance:**
- Pre-warm caches before benchmarking
- Use fixed-size data structures
- Minimize object creation
- Profile with `cProfile` to identify variance sources

---

## Next Steps

1. **Profile with cProfile**: Get exact timing breakdown
2. **Implement Tasks #1-4**: Target 30-40% speedup
3. **Re-benchmark**: Measure actual improvement
4. **Implement Tasks #5-8**: Additional 5-8% improvement
5. **Consider Task #10**: If processing multiple positions becomes common

---

## Testing Strategy

For each optimization:
1. Create test case with sample position
2. Run benchmark (200 iterations)
3. Compare to baseline (198.66 ms)
4. Verify correctness (ctx outputs identical)
5. Document actual speedup vs expected

**Success Criteria:**
- Average time < 140 ms (30% improvement minimum)
- CV < 20% (reduced variance)
- All tests pass (no regressions)
