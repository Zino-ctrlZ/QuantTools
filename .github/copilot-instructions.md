# QuantTools DataManager System - Copilot Instructions

## Project Overview
This is a quantitative trading system focused on options pricing and risk management. 

## Code Style & Standards

### Type Hints
- Always use complete type hints for all function parameters and return values
- Use `Union[datetime, str]` for date parameters that accept multiple formats
- Use `Optional[T]` for nullable parameters
- Import types from `typing`: `Optional, Union, List, Dict, Tuple, ClassVar`

### Date/Time Conversion
- **Always use `to_datetime` from `trade.helpers.helper` for datetime conversions**
- Never use `datetime.strptime()` or `pd.to_datetime()` directly
- Import: `from trade.helpers.helper import to_datetime`
- Handles both single values and iterables
- Tries "%Y-%m-%d" format first, then lets pandas guess if that fails
- Supports optional `format` parameter for custom formats

**Example:**
```python
from trade.helpers.helper import to_datetime

# Single string conversion
date_obj = to_datetime("2026-01-15")

# With custom format
date_obj = to_datetime("15-01-2026", format="%d-%m-%Y")

# Iterable conversion
dates = to_datetime(["2026-01-15", "2026-01-16", "2026-01-17"])

# Already datetime - returns as-is
date_obj = to_datetime(datetime.now())
```

### Docstrings
- Use Google-style docstrings for all classes and methods
- Include Args, Returns, Raises, and Examples sections
- Examples should be executable and demonstrate real-world usage

**Example:**
```python
def get_forward_timeseries(
    self,
    start_date: Union[datetime, str],
    end_date: Union[datetime, str],
    maturity_date: Union[datetime, str],
    div_type: Optional[DivType] = None,
    *,
    dividend_result: Optional[DividendsResult] = None,
    use_chain_spot: bool = True,
) -> ForwardResult:
    """Returns daily forward prices from valuation dates to maturity.

    Computes forward prices for each business day in [start_date, end_date],
    where each forward is valued to the fixed maturity_date. Uses discrete
    dividends (Schedule objects) or continuous yields depending on div_type.

    Args:
        start_date: First valuation date (YYYY-MM-DD string or datetime).
        end_date: Last valuation date (YYYY-MM-DD string or datetime).
        maturity_date: Fixed horizon date for all forwards (e.g., option expiry).
        div_type: DivType.DISCRETE or DivType.CONTINUOUS. Defaults to DISCRETE.
        dividend_result: Pre-computed dividend data. If None, fetches internally.
        use_chain_spot: If True, uses split-adjusted chain_spot prices.

    Returns:
        ForwardResult containing daily_discrete_forward or daily_continuous_forward
        Series with DatetimeIndex, plus the dividend_result used and cache key.

    Raises:
        ValueError: If maturity_date < start_date.
        ValueError: If dividend_result.undo_adjust != use_chain_spot.

    Examples:
        >>> # Basic usage with automatic dividend fetching
        >>> fwd_mgr = ForwardDataManager("AAPL")
        >>> result = fwd_mgr.get_forward_timeseries(
        ...     start_date="2025-01-01",
        ...     end_date="2025-01-31",
        ...     maturity_date="2025-06-20",
        ...     div_type=DivType.DISCRETE,
        ...     use_chain_spot=True
        ... )
        >>> print(result.daily_discrete_forward.head())
        datetime
        2025-01-02    155.32
        2025-01-03    156.01
        ...

        >>> # Provide pre-computed dividends for efficiency
        >>> div_mgr = DividendDataManager("AAPL")
        >>> div_result = div_mgr.get_schedule_timeseries(
        ...     start_date="2025-01-01",
        ...     end_date="2025-01-31",
        ...     maturity_date="2025-06-20",
        ...     undo_adjust=True
        ... )
        >>> fwd_result = fwd_mgr.get_forward_timeseries(
        ...     start_date="2025-01-01",
        ...     end_date="2025-01-31",
        ...     maturity_date="2025-06-20",
        ...     dividend_result=div_result,
        ...     use_chain_spot=True
        ... )
    """
```

### Naming Conventions
- **Classes:**
  - Managers end with `Manager`: `DividendDataManager`, `RatesDataManager`
  - Results end with `Result`: `DividendsResult`, `ForwardResult`, `RatesResult`
  - Configs end with `Config`: `DividendsConfig`
- **Methods:**
  - Use `get_*` for retrieval methods: `get_schedule()`, `get_rate()`
  - Use `_load_*` for private loading helpers: `_load_spot()`, `_load_rates()`
  - Use `_compute_*` for calculation methods: `_compute_forward_discrete()`
- **Variables:**
  - Use `_str` suffix for string dates: `start_str`, `end_str`, `mat_str`
  - Use `_dt` suffix for date objects: `start_dt`, `end_dt`, `mat_dt`

### Dataclasses
- Prefer `@dataclass` over regular classes for data containers
- Use pydantic `@dataclass` for validation when needed (Only on strict data models)
    - Import from `pydantic.dataclasses` for pydantic dataclasses and alias as `pydantic_dataclass`
- Result classes should inherit from base `Result` class
- Use `frozen=True, slots=True` for immutable configs (e.g., `CacheSpec`)

**Example:**
```python
@dataclass(frozen=True, slots=True)
class CacheSpec:
    """Configuration for cache initialization."""
    base_dir: Optional[Path] = DM_GEN_PATH.as_posix()
    default_expire_days: Optional[int] = 500
    default_expire_seconds: Optional[int] = None
    cache_fname: Optional[str] = None
    clear_on_exit: bool = False

@dataclass
class DividendsResult(Result):
    """Result container for dividend data."""
    daily_discrete_dividends: Optional[pd.Series] = None
    daily_continuous_dividends: Optional[pd.Series] = None
    dividend_type: Optional[DivType] = None
    key: Optional[str] = None
    undo_adjust: Optional[bool] = None
    
    def is_empty(self) -> bool:
        if self.dividend_type == DivType.DISCRETE:
            return self.daily_discrete_dividends is None or self.daily_discrete_dividends.empty
        return True
```
