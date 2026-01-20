
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, Union
from trade.helpers.helper import to_datetime, is_busday, is_USholiday
from trade.helpers.helper import ny_now
from trade.optionlib.assets.dividend import SECONDS_IN_DAY, SECONDS_IN_YEAR

DATE_HINT = Union[datetime, str]

def time_distance_helper(start: datetime, end: datetime) -> float:
    """Calculates time distance in years between two dates."""
    delta = (to_datetime(end) - to_datetime(start)).days * SECONDS_IN_DAY
    return delta / SECONDS_IN_YEAR


@dataclass(slots=True)
class DateRangePacket:
    """
    Simple container for start/end date ranges with both datetime and string formats.
    """

    start_date: DATE_HINT
    end_date: DATE_HINT
    start_str: Optional[str] = None
    end_str: Optional[str] = None
    maturity_date: Optional[DATE_HINT] = None
    maturity_str: Optional[str] = None

    def __post_init__(self):
        self.start_date = to_datetime(self.start_date)
        self.end_date = to_datetime(self.end_date)
        if self.maturity_date is not None:
            self.maturity_date = to_datetime(self.maturity_date)

        self.start_str = self.start_str or self.start_date.strftime("%Y-%m-%d")
        self.end_str = self.end_str or self.end_date.strftime("%Y-%m-%d")
        if self.maturity_date is not None:
            self.maturity_str = self.maturity_str or self.maturity_date.strftime("%Y-%m-%d")
        else:
            self.maturity_str = None


def _should_save_today(max_date: date) -> bool:
    """
    Determines if data should be saved today based on the max_date and current time in New York.
    """
    today = date.today()
    current_hour = ny_now().hour
    return max_date >= today and current_hour >= 16

def is_available_on_date(date: date) -> bool:
    """
    Returns True if the given date is a business day and not a US holiday, False otherwise.
    """
    date = to_datetime(date).strftime("%Y-%m-%d")
    return is_busday(date) and not is_USholiday(date)