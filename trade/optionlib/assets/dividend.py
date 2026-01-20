from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Tuple, Union, Any
import math
from dateutil.rrule import rrule, MONTHLY
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from trade.helpers.helper import (
    compare_dates,
    time_distance_helper,
)
from trade.helpers.Context import Context
from trade.assets.Stock import Stock
from ..utils.market_data import get_div_schedule
from ..utils.format import assert_equal_length
from ..utils.timing import format_dates, subtract_dates, validate_dates
from ..config.defaults import DAILY_BASIS, DIVIDEND_LOOKBACK_YEARS, DIVIDEND_FORECAST_METHOD
from trade.helpers.Logging import setup_logger

logger = setup_logger("trade.optionlib.assets.dividend", stream_log_level="DEBUG")

SECONDS_IN_YEAR = 365.0 * 24.0 * 3600.0
SECONDS_IN_DAY = 24.0 * 3600.0
FREQ_MAP = {
    "monthly": 1,
    "quarterly": 3,
    "semiannual": 6,
    "annual": 12,
}


# Helper Functions
def classify_frequency(days):
    if 20 < days < 40:
        return "monthly"
    elif 50 < days < 80:
        return "bi-monthly"
    elif 80 < days < 110:
        return "quarterly"
    elif 170 < days < 200:
        return "semi-annual"
    elif 330 < days < 370:
        return "annual"
    else:
        return "irregular"


def infer_frequency(div_history: pd.DataFrame):
    """
    Infer the frequency of dividends based on the historical data.
    div_history: pd.DataFrame - Historical dividend data with datetime index.
    Returns a string representing the inferred frequency.
    """
    date_diffs = div_history.index.to_series().diff()
    day_diffs = date_diffs.dt.days
    _ = day_diffs.mode()[0]
    frequency_labels = day_diffs.apply(classify_frequency)
    return frequency_labels.mode()[0]


def infer_dividend_growth_rate(
    div_df: pd.DataFrame, valuation_date: datetime, lookback_years: int = 5, method: str = "cagr"
) -> float:
    """
    Infer the growth rate of dividends based on historical data.
    div_df: pd.DataFrame - Historical dividend data with 'amount' column.
    valuation_date: datetime - The date for valuation purposes.
    lookback_years: int - Number of years to look back for historical dividends (default is 5).
    method: str - Method to infer growth rate ('constant', 'avg', 'cagr', 'regression').
    Returns the inferred growth rate as a float.
    """
    # Ensure proper datetime format and sort
    valuation_date = pd.to_datetime(valuation_date).date()
    div_df = div_df.sort_index()
    div_df = div_df.loc[div_df.index.date < valuation_date]

    if all(div_df.amount == 0.0):
        return 0.0

    # Filter by lookback period
    cutoff_date = valuation_date - pd.DateOffset(years=lookback_years)
    df_filtered = div_df.loc[div_df.index.date >= cutoff_date.date()].copy()

    if len(df_filtered) < 2:
        return 0.0

    if method == "constant":
        return 0.0  # No growth

    elif method == "avg":
        # Compute year-over-year changes
        # Special cutoff to ensure we have enough data
        # Avg needs at least 2 years of data to compare years
        cutoff_date = valuation_date - pd.DateOffset(years=lookback_years + 1)
        df_filtered = div_df.loc[div_df.index.date >= cutoff_date.date()].copy()
        df_filtered["year"] = df_filtered.index.year
        yearly_avg = df_filtered.groupby("year")["amount"].mean()
        diffs = yearly_avg.pct_change().dropna()
        return diffs.mean()

    elif method == "cagr":
        first_date = df_filtered.index[0]
        last_date = df_filtered.index[-1]
        n_years = (last_date - first_date).days / DAILY_BASIS
        start = df_filtered.iloc[0]["amount"]
        end = df_filtered.iloc[-1]["amount"]
        if start <= 0:
            raise ValueError("Starting dividend must be positive for CAGR.")
        return (end / start) ** (1 / n_years) - 1

    elif method == "regression":
        df_filtered["ordinal_date"] = df_filtered.index.map(
            datetime.toordinal
        )  ## Convert dates to ordinal (numbers) for regression
        df_filtered["log_amount"] = np.log(df_filtered["amount"])  ## Log-transform the amount for regression
        X = df_filtered[["ordinal_date"]]
        y = df_filtered["log_amount"]
        model = LinearRegression().fit(X, y)
        # Convert daily log return to annualized rate
        annualized_growth = model.coef_[0] * DAILY_BASIS
        return annualized_growth

    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'constant', 'avg', 'cagr', 'regression'.")


def get_last_dividends(div_history: pd.DataFrame, valuation_date: datetime, size=4) -> Tuple[datetime, float]:
    """
    Get the nearest dividend date and amount after the valuation date.

    div_history: pd.DataFrame - Historical dividend data with 'amount' column.
    valuation_date: datetime - The date for valuation purposes.

    Returns a tuple of (nearest_dividend_date, nearest_dividend_amount).
    """
    valuation_date = format_dates(valuation_date)[0].date()
    future_divs = div_history[div_history.index.date <= valuation_date]

    if future_divs.empty:
        return 0.0

    # Get the last 'size' dividends before the valuation date
    last_divs = sum(future_divs.tail(size)["amount"])

    return last_divs


def project_dividends(
    valuation_date: datetime,
    end_date: datetime,
    div_history: pd.DataFrame,
    inferred_growth_rate: float,
) -> Tuple[List[float], List[datetime], datetime]:
    """
    Project future dividends based on historical data and inferred growth rate.

    valuation_date: datetime - The date for valuation purposes.
    expiration_date: datetime - The date when the option expires.
    div_history: pd.DataFrame - Historical dividend data with 'amount' column.
    inferred_growth_rate: float - Estimated annual growth rate of dividends.

    Returns a list of projected dividends with their payment dates.
    """
    end_date, valuation_date = format_dates(end_date, valuation_date)
    typical_spacing = div_history.index.to_series().diff().dt.days.mode()[0]
    expected_dividend_size = int((subtract_dates(end_date, valuation_date) // typical_spacing) + 1)
    period_inferred = classify_frequency(typical_spacing)
    past_divs = div_history.loc[div_history.index.date < valuation_date.date()]
    last_div = past_divs.iloc[-1]["amount"] if not past_divs.empty else 0.0
    last_date = past_divs.index[-1].date() if not past_divs.empty else valuation_date
    periodic_growth = inferred_growth_rate / (12 / FREQ_MAP[period_inferred])
    dividend_list = [last_div * (1 + periodic_growth) ** i for i in range(expected_dividend_size)]
    date_list = [
        last_date + relativedelta(months=i * FREQ_MAP[period_inferred]) for i in range(1, expected_dividend_size + 1, 1)
    ]

    ## Cutoff any dates beyond end_date
    filtered_dividends = [
        (dt, amt) for dt, amt in zip(date_list, dividend_list) if compare_dates.is_before(dt, end_date)
    ]
    if not filtered_dividends:
        return [], [], last_date
    date_list, dividend_list = zip(*filtered_dividends)

    return dividend_list, date_list, last_date


def _dual_project_dividends(
    valuation_date: datetime,
    end_date: datetime,
    div_history: pd.DataFrame,
    inferred_growth_rate: float,
) -> Tuple[List[float], List[datetime], datetime]:
    """
    Project future dividends based on historical data and inferred growth rate.
    This function is similar to project_dividends. The difference lies in using historical dividend for dates < today, and projected dividends for dates >= today.

    valuation_date: datetime - The date for valuation purposes.
    expiration_date: datetime - The date when the option expires.
    div_history: pd.DataFrame - Historical dividend data with 'amount' column.
    inferred_growth_rate: float - Estimated annual growth rate of dividends.

    Returns a list of projected dividends with their payment dates.
    """
    end_date, valuation_date = format_dates(end_date, valuation_date)
    typical_spacing = div_history.index.to_series().diff().dt.days.mode()[0]
    period_inferred = classify_frequency(typical_spacing)

    ## Push back valuation date by period & typical spacing * 2 to capture historical dividends
    new_valuation_date = valuation_date - relativedelta(days=typical_spacing * 8)

    ## Get dividends btwn valuation date and today
    historical_divs = div_history.loc[
        (div_history.index.date >= new_valuation_date.date())
        &
        ## Filter to include only dividends between valuation date and today. With today inclusive
        (div_history.index.date <= datetime.today().date())
    ]

    date_list = list(historical_divs.index.date)
    amount_list = list(historical_divs["amount"])
    if not date_list:
        return [], [], valuation_date

    ## Expected dividend size:
    ## Since we pushed valuation date back, we will include it in expected dividend size calculation
    expected_dividend_size = int((subtract_dates(end_date, new_valuation_date) // typical_spacing) + 1)
    expected_dividend_size_for_original_valuation = int(
        (subtract_dates(end_date, valuation_date) // typical_spacing) + 1
    )
    logger.info(
        f"Expected Dividend Size before adjustment: {expected_dividend_size}, for original valuation: {expected_dividend_size_for_original_valuation}. Size from historical divs: {len(date_list)}"
    )

    ## Project future dividends after today
    last_div = amount_list[-1] if amount_list else 0.0

    ## Last date is going to be latest date in date_list in other to project future dividends from there
    last_date = date_list[-1] if date_list else valuation_date

    ## We reduce expected dividend size by the number of historical dividends we have
    expected_dividend_size -= len(date_list)

    ## If expected dividend size is less than 0, set to 0
    if expected_dividend_size < 0:
        expected_dividend_size = 0

    logger.info(f"Expected Dividend Size to be projected: {expected_dividend_size}")
    periodic_growth = inferred_growth_rate / (12 / FREQ_MAP[period_inferred])

    ## Generate projected dividends starting from last_date
    dividend_list = [last_div * (1 + periodic_growth) ** i for i in range(expected_dividend_size)]
    logger.info(f"Projected Dividend List: {dividend_list}")

    ## Combine historical and projected dividends
    dividend_list = amount_list + dividend_list
    logger.info(f"Combined Dividend List: {dividend_list}")

    ## Combine historical and projected dates
    date_list = date_list + [
        last_date + relativedelta(months=i * FREQ_MAP[period_inferred]) for i in range(1, expected_dividend_size + 1, 1)
    ]
    logger.info(f"Combined Date List: {date_list}")

    ## Cutoff any dates beyond end_date
    filtered_dividends = [
        (dt, amt)
        for dt, amt in zip(date_list, dividend_list)
        if compare_dates.inbetween(date=dt, start=valuation_date.date(), end=end_date)
    ]
    date_list, dividend_list = zip(*filtered_dividends) if filtered_dividends else ([], [])

    return dividend_list, date_list, last_date


# Abstract base class for dividends
class Dividend(ABC):
    """
    Abstract base class for dividends.
    This class defines the interface for dividend calculations and schedules.
    It should be subclassed to implement specific dividend types.
    """

    @abstractmethod
    def get_present_value(self, *args, **kwargs) -> float:
        """
        Calculate the present value of the dividend.
        """

    @abstractmethod
    def get_type(self) -> str:
        """
        Get the type of the dividend.
        """

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.get_type()}>"


class ScheduleEntry(tuple):
    __slots__ = ()

    def __new__(cls, date: Any, amount: Any = None):
        # Case 1: called normally -> ScheduleEntry(date, amount)
        if amount is not None:
            return super().__new__(cls, (date, float(amount)))

        # Case 2: called by pickle -> ScheduleEntry((date, amount))
        if isinstance(date, tuple) and len(date) == 2:
            d, a = date
            return super().__new__(cls, (d, float(a)))

        raise TypeError("ScheduleEntry requires (date, amount) or ((date, amount),)")

    @property
    def date(self) -> datetime:
        return self[0]

    @property
    def amount(self) -> float:
        return self[1]

    def __mul__(self, value) -> "ScheduleEntry":
        if not isinstance(value, (int, float)):
            raise TypeError(f"Can only multiply ScheduleEntry by int or float, not {type(value)}")
        return ScheduleEntry(self.date, self.amount * value)

    def __repr__(self) -> str:
        use_date = self.date.strftime('%Y-%m-%d') if isinstance(self.date, datetime) else str(self.date)
        return f"<ScheduleEntry: {use_date} - {self.amount}>"


class Schedule:
    """
    Class to represent a dividend schedule for a given date.
    """

    def __init__(self, schedule: List[Tuple[datetime, float]]):
        """
        Initialize a Schedule object.
        schedule: List[Tuple[datetime, float]] - A list of tuples containing dividend dates and amounts.
        """
        self._schedule: List[Tuple[datetime, float]] = schedule

    @property
    def schedule(self) -> List[ScheduleEntry]:
        return [ScheduleEntry(dt, amt) for dt, amt in self._schedule]

    @schedule.setter
    def schedule(self, value: List[Tuple[datetime, float]]):
        self._schedule = value

    def get_schedule(self) -> List[Tuple[datetime, float]]:
        """
        Get the dividend schedule as a list of tuples containing date and amount.
        Returns:
            List[Tuple[datetime, float]]: A list of tuples where each tuple contains a dividend date and its corresponding amount.
        """
        return self.schedule

    def __repr__(self):
        return f"<Schedule: {len(self.schedule)} dividends>"

    def __len__(self):
        """
        Get the number of dividends in the schedule.
        Returns:
            int: The number of dividends in the schedule.
        """
        return len(self.schedule)

    def __str__(self):
        """
        Get a string representation of the schedule.
        Returns:
            str: A string representation of the schedule.
        """
        return self.__repr__()

    def __mul__(self, value: float) -> "Schedule":
        """
        Multiply all amounts in the schedule by a scalar value.
        value: float - The scalar value to multiply by.
        Returns:
            Schedule: A new Schedule object with updated amounts.
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"Can only multiply Schedule by int or float, not {type(value)}")
        new_schedule = [(entry.date, entry.amount * value) for entry in self.schedule]
        return Schedule(new_schedule)

    def __iter__(self):
        """
        Make the Schedule object iterable.
        """
        return iter(self.schedule)


# Concrete class for a dividend schedule
class DividendSchedule(Dividend):
    """
    A class to represent a schedule of dividends.
    """

    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        freq: str = "quarterly",
        amount: Union[float, List[float]] = 1.0,
        valuation_date: datetime = None,
        basis: int = 365,
        **kwargs: Union[str, int, float, datetime, None],
    ):
        """
        Initialize a DividendSchedule object.
        start_date: datetime - The start date of the dividend schedule. Starts the schedule.
        end_date: datetime - The end date of the dividend schedule.
        freq: str - The frequency of dividends ('monthly', 'quarterly', 'semiannual', 'annual').
        amount: float or list - The dividend amount (can be a scalar or a list).
        valuation_date: datetime - The date for valuation purposes.
        basis: int - The day count basis (default is 365).
        """
        if freq not in FREQ_MAP:
            raise ValueError(f"Unsupported frequency '{freq}'. Use one of {list(FREQ_MAP.keys())}.")
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        self.valuation_date = valuation_date or start_date
        self.basis = basis
        self.input_amount = amount
        self._setup_schedule()

    def _setup_schedule(self):
        """ """
        # Generate dividend dates
        months = FREQ_MAP[self.freq]

        ## Generate dates using rrule
        self.dates = list(rrule(freq=MONTHLY, interval=months, dtstart=self.start_date, until=self.end_date))
        self.dates = [dt for dt in self.dates if compare_dates.is_after(dt, self.start_date)]
        if not self.dates:
            raise ValueError("No dividend dates generated. Check your start and end dates.")

        # Handle amount (scalar or list)
        if isinstance(self.input_amount, list):
            if len(self.input_amount) < len(self.dates):
                raise ValueError("Amount list must cover all dividend dates.")
            self.amounts = self.input_amount[: len(self.dates)]
        else:
            self.amounts = [self.input_amount] * len(self.dates)

        self.schedule = Schedule(list(zip(self.dates, self.amounts)))

    def get_schedule(self) -> List[Tuple[datetime, float]]:
        """
        Get the dividend schedule as a list of tuples containing date and amount.
        Returns:
            List[Tuple[datetime, float]]: A list of tuples where each tuple contains a dividend date and its corresponding amount.
        """
        return self.schedule

    def get_year_fractions(self) -> List[Tuple[float, float]]:
        """
        Calculate the year fractions from the valuation date to each dividend date.
        Returns:
            List[Tuple[float, float]]: A list of tuples containing the time distance and the corresponding amount.
        """
        return Schedule(
            [
                (time_distance_helper(dt, self.valuation_date), amt)
                for dt, amt in self.schedule
                if dt > self.valuation_date
            ]
        )

    def get_present_value(self, discount_rate: float, sum_up: bool = True, **kwargs) -> float:
        """
        Calculate the present value of the dividend schedule using a discount rate.
        discount_rate: float - The discount rate to apply.
        """
        pv = []
        for dt, amt in self.schedule:
            if compare_dates.is_after(dt, self.valuation_date):
                time_fraction = time_distance_helper(dt, self.valuation_date)
                pv_amt = amt * math.exp(-discount_rate * time_fraction)
                pv.append(pv_amt)
        return sum(pv) if sum_up else pv

    def get_type(self) -> str:
        """
        Get the type of the dividend schedule.
        """
        return "discrete"

    def __repr__(self):
        return f"<DividendSchedule: {len(self.schedule)} dividends from {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}>"


class ContinuousDividendYield(Dividend):
    """
    A class to represent a continuous dividend yield.
    This class models a continuous dividend yield, which is typically used in financial mathematics.
    It calculates the present value of dividends using an exponential discount factor.
    """

    def __init__(
        self, yield_rate: float, start_date: datetime, end_date: datetime, valuation_date: datetime = None, **kwargs
    ):
        """
        Initialize a ContinuousDividendYield object.
        yield_rate: float - The continuous dividend yield (between 0 and 1).
        start_date: datetime - The date when the yield starts.
        valuation_date: datetime - The date for valuation purposes (default is start_date).
        """
        super().__init__()
        if not (0 <= yield_rate < 1):
            raise ValueError("Dividend yield must be between 0 and 1.")
        self.yield_rate = yield_rate
        self.start_date = start_date
        self.valuation_date = valuation_date or start_date
        self.end_date = end_date
        self.T = time_distance_helper(
            self.end_date,
            self.valuation_date,
        )

    def get_yield(self) -> float:
        """
        Get the continuous dividend yield.
        Returns:
            float: The continuous dividend yield.
        """
        return self.yield_rate

    def get_present_value(self, end_date: datetime = None, **kwargs) -> float:
        """
        Return the exponential discount factor from q over T:
        e^{-qT}
        """
        T = self.T if end_date is None else time_distance_helper(end_date, self.valuation_date)
        return math.exp(-self.yield_rate * T)

    def get_type(self) -> str:
        """
        Get the type of the dividend yield.
        """
        return "continuous"

    def __repr__(self):
        return f"<ContinuousDividendYield: q={self.yield_rate:.4f}>"


# Market Models
class MarketDividendSchedule(DividendSchedule):
    """
    A dividend schedule that projects future dividends based on historical data and inferred growth rates.
    This class extends the DividendSchedule class to include methods for inferring growth rates and projecting dividends.
    """

    def __init__(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        valuation_date: datetime = None,
        lookback_years: int = DIVIDEND_LOOKBACK_YEARS,
        **kwargs,
    ):
        """
        Initialize a MarketDividendSchedule object.
        ticker: str - The stock ticker symbol.
        start_date: datetime - The start date for the dividend schedule.
        end_date: datetime - The end date for the dividend schedule.
        valuation_date: datetime - The date for valuation purposes (default is start_date).
        lookback: int - Number of years to look back for historical dividends (default is 8).

        ps: user can set spot price at asset level by utilizing div_object.spot_price = xxx
        """

        ## Validate Dates
        validate_dates(valuation_date, start_date, end_date)

        ## Format Dates
        valuation_date, start_date, end_date = format_dates(valuation_date, start_date, end_date)

        with Context(
            end_date=valuation_date.strftime("%Y-%m-%d")
        ):  ## To ensure spot being accessed is for the specific valuation date
            self.asset = Stock(ticker)

        div = get_div_schedule(ticker, filter_specials=True)
        div = div[div.index.date <= valuation_date.date()]  # Filter to include only dividends before the valuation date
        self.ticker = ticker
        self.lookback_years = lookback_years
        self.div_history = div
        self._projected_freq = self._infer_frequency(self.div_history)
        self.growth_method = kwargs.get("growth_method", DIVIDEND_FORECAST_METHOD)
        self.amount = 0.0
        self.valuation_date = valuation_date or start_date
        self.growth_rate = self._infer_growth_rate(
            self.div_history, lookback=self.lookback_years, method=self.growth_method
        )
        self._projected_dividends, self.payment_dates = self._project_schedule(
            div_history=self.div_history,
            start_date=start_date,
            end_date=end_date,
            valuation_date=valuation_date,
            **kwargs,
        )
        self.model_start_date = start_date

        # # Create the schedule
        super().__init__(
            start_date=self.last_div_date or start_date,
            end_date=end_date,
            freq=self._projected_freq,
            amount=self._projected_dividends,
            valuation_date=valuation_date or start_date,
            **kwargs,
        )

    @property
    def spot_price(self):
        return self.asset.spot_price

    @spot_price.setter
    def spot_price(self, v):
        self.asset.spot_price = v

    def _infer_frequency(self, div_history: pd.DataFrame) -> str:
        """
        Infer the frequency of dividends based on the historical data.
        """
        if div_history.empty:
            return "quarterly"
        return infer_frequency(div_history)

    def _infer_growth_rate(self, div_history: pd.DataFrame, lookback=8, method="cagr") -> float:
        """
        Infer the dividend growth rate based on historical data.
        """
        if div_history.empty:
            return 0.0

        return infer_dividend_growth_rate(
            div_history, valuation_date=self.valuation_date, lookback_years=lookback, method=method
        )

    def _project_schedule(
        self,
        div_history: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        valuation_date: str | datetime = None,
        **kwargs,
    ) -> Tuple[List[float], List[datetime]]:
        """
        Project future dividends based on historical data and inferred growth rate.
        """
        if div_history.empty:
            return 0.0
        dividend_list, payment_dates, last_date = project_dividends(
            valuation_date=valuation_date or start_date,
            end_date=end_date,
            div_history=div_history,
            inferred_growth_rate=self.growth_rate,
        )
        self.last_div_date = last_date
        return dividend_list, payment_dates


class MarketContinuousDividends(ContinuousDividendYield):
    """
    A continuous dividend yield model that uses historical dividend data as forward dividend yield
    """

    def __init__(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        valuation_date: datetime = None,
        spot_price: float = None,
        **kwargs,
    ):
        """
        Initialize a MarketContinuousDividends object.
        ticker: str - The stock ticker symbol.
        start_date: datetime - The start date for the dividend schedule.
        end_date: datetime - The end date for the dividend schedule.
        valuation_date: datetime - The date for valuation purposes (default is start_date).
        lookback_years: int - Number of years to look back for historical dividends (default is 8).
        """
        validate_dates(valuation_date, start_date, end_date)
        valuation_date, start_date, end_date = format_dates(valuation_date, start_date, end_date)
        div = get_div_schedule(ticker, filter_specials=True)
        div = div[
            div.index.date <= valuation_date.date()
        ]  ## Filter to include only dividends before the valuation date
        self.div = div
        self._projected_freq = self._infer_frequency(div)
        self.valuation_date = valuation_date or start_date
        self.ticker = ticker
        with Context(
            end_date=valuation_date.strftime("%Y-%m-%d")
        ):  ## To ensure spot being accessed is for the specific valuation date
            self.asset = Stock(ticker)
        self.__set_yield_rate()

        # Initialize the ContinuousDividendYield with the inferred yield rate
        super().__init__(
            yield_rate=self.current_q,
            start_date=start_date,
            end_date=end_date,
            valuation_date=valuation_date or start_date,
            **kwargs,
        )

    @property
    def spot_price(self):
        return self.asset.spot_price

    @spot_price.setter
    def spot_price(self, v):
        self.asset.spot_price = v
        self.__set_yield_rate()

    def __set_yield_rate(self):
        self.current_q = (
            get_last_dividends(self.div, valuation_date=self.valuation_date, size=FREQ_MAP[self._projected_freq])
            / self.spot_price
        )
        self.yield_rate = self.current_q

    def _infer_frequency(self, div_history: pd.DataFrame) -> str:
        """
        Infer the frequency of dividends based on the historical data.
        """
        if div_history.empty:
            return "quarterly"
        return infer_frequency(div_history)


# Vectorized Dividend Functions
def get_div_histories(tickers: List[str] | np.ndarray) -> dict:
    """
    Get historical dividend schedules for multiple tickers.
    tickers: List[str] or np.ndarray - List of ticker symbols.
    Returns a dictionary where keys are ticker symbols and values are DataFrames of dividend schedules.
    """

    assert_equal_length(tickers)
    unique_ticks = set(tickers)
    tick_history = {t: get_div_schedule(t) for t in unique_ticks}
    return tick_history


def get_vectorized_dividend_scehdule(
    tickers: list | np.ndarray,
    start_dates: List[datetime],
    end_dates: List[datetime],
    valuation_dates: List[datetime] = None,
    **kwargs,
) -> List[Schedule]:
    """
    Generate a vectorized dividend schedule for multiple tickers.
    tickers: list or np.ndarray - List of ticker symbols.
    start_dates: List[datetime] - List of start dates for each ticker.
    end_dates: List[datetime] - List of end dates for each ticker.
    valuation_dates: List[datetime] - List of valuation dates for each ticker (default is None, uses start_dates).
    kwargs: Additional keyword arguments for dividend growth rate inference.
    Returns a list of lists containing projected dividend amounts and their corresponding dates.
    """

    schedules = []
    lookback_yrs = kwargs.get("lookback_yrs", DIVIDEND_LOOKBACK_YEARS)
    method = kwargs.get("method", DIVIDEND_FORECAST_METHOD)

    ## Check for dual method
    is_dual_method = method.startswith("constant+")

    ## Adjust method if dual
    if is_dual_method:
        method = method.split("+")[1]

    tick_history = get_div_histories(tickers)
    valuation_dates = valuation_dates or start_dates
    for ticker, _, end_date, val_date in zip(tickers, start_dates, end_dates, valuation_dates):
        gr = infer_dividend_growth_rate(tick_history[ticker], val_date, lookback_yrs, method)

        ## Dual method uses historical dividends for dates < today, projected dividends for dates >= today
        if is_dual_method or method == "constant":
            logger.info(f"Using dual projection method for ticker {ticker}")
            payments, dates, _ = _dual_project_dividends(
                valuation_date=val_date, end_date=end_date, div_history=tick_history[ticker], inferred_growth_rate=gr
            )
        else:
            logger.info(f"Using standard projection method for ticker {ticker}")
            payments, dates, _ = project_dividends(
                valuation_date=val_date, end_date=end_date, div_history=tick_history[ticker], inferred_growth_rate=gr
            )

        entries = list(zip(dates, payments))
        schedules.append(Schedule((entries)))
    return schedules


def vector_convert_to_time_frac(
    schedules: List[Schedule], valuation_dates: List[datetime], end_dates: List[datetime]
) -> List[Schedule]:
    """
    Convert a list of schedules to a list Tuple[T (Time to expiry in years), Dividend Amount].

    schedules: List[Schedule] - List of schedules where each schedule is a list of (amount, date) tuples wrapped in a Schedule object.
    valuation_dates: List[datetime] - List of valuation dates corresponding to each schedule.
    end_dates: List[datetime] - List of end dates corresponding to each schedule.

    Returns a list of lists containing time fractions and amounts.
    """
    # assert_equal_length(schedules, valuation_dates, end_dates)

    out: List[Schedule] = []

    for sch, val, end in zip(schedules, valuation_dates, end_dates):
        # Convert once

        # If schedule dates are sorted, you can optionally early-break (see note below)
        converted = []
        for dt, amt in sch:  # dt is datetime, amt is float
            # Exclusive bounds: val < dt < end
            days_in_seconds = (dt - val.date()).days * 86400
            if val.date() < dt < end.date():
                t = days_in_seconds / SECONDS_IN_YEAR
                converted.append((t, amt))

        out.append(Schedule(converted))

    return out


def vectorized_discrete_pv(
    schedules: List[List[ScheduleEntry]], r: List[list], _valuation_dates: List[datetime], _end_dates: List[datetime]
) -> List[float]:
    """
    Calculate the present value of a list of dividend schedules using vectorized operations.
    schedules: List[list] - List of schedules where each schedule is a list of (amount, date) tuples.
    r: List[float] - List of discount rates corresponding to each schedule.
    _valuation_dates: List[datetime] - List of valuation dates corresponding to each schedule.
    _end_dates: List[datetime] - List of end dates corresponding to each schedule.
    Returns a list of present values for each schedule.
    """
    assert_equal_length(
        schedules, r, _end_dates, _valuation_dates, names=["schedules", "r", "_end_dates", "_valuation_dates"]
    )
    df_cache = {}

    pv = []
    SECONDS_IN_YEAR = 365.0 * 24.0 * 3600.0

    for i, sch in enumerate(schedules):
        ri = r[i]  # rate for this schedule
        val = _valuation_dates[i]
        end = _end_dates[i]

        # Use integer seconds
        val_ts = int(val.timestamp())

        total = 0.0

        # sch entries are (date, div) per your point (2)
        for dt, x in sch:
            if val.date() < dt < end.date():
                days_in_seconds = (dt - val.date()).days * 86400

                key = (ri, val_ts, days_in_seconds)
                df = df_cache.get(key)

                if df is None:
                    t = days_in_seconds / SECONDS_IN_YEAR
                    df = math.exp(-ri * t)
                    df_cache[key] = df

                total += x * df

        pv.append(total)

    return pv


def get_vectorized_dividend_rate(tickers: str | List[str], spots: List[float], valuation_dates: List[float]):
    """
    Get the vectorized dividend rate for a list of tickers based on their historical dividend data.

    tickers: str or List[str] - Ticker symbols of the stocks.
    spots: List[float] - Current spot prices for each ticker.
    valuation_dates: List[datetime] - Dates for which to calculate the dividend rates.

    Returns a numpy array of dividend rates.
    """
    assert_equal_length(tickers, spots, valuation_dates)
    tick_history = get_div_histories(tickers)
    div_rates = [get_last_dividends(tick_history[t], valuation_dates[i]) / spots[i] for i, t in enumerate(tickers)]
    return np.array(div_rates)


def get_vectorized_continuous_dividends(
    div_rates: List[float], _valuation_dates: List[datetime], _end_dates: List[datetime]
):
    """
    Get the vectorized continuous dividend discount factors.
    div_rates: List[float] - List of continuous dividend rates.
    _valuation_dates: List[datetime] - List of valuation dates.
    _end_dates: List[datetime] - List of end dates.
    Returns a numpy array of discount factors.
    """

    assert_equal_length(
        div_rates,
        _valuation_dates,
        _end_dates,
        names=["div_rates", "_valuation_dates", "_end_dates"],
    )
    discounted = [
        math.exp(
            -div_rate
            * time_distance_helper(
                _end_dates[i],
                _valuation_dates[i],
            )
        )
        for i, div_rate in enumerate(div_rates)
    ]
    return np.array(discounted)
