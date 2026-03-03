from bisect import bisect_left, bisect_right
from datetime import date
from typing import List
from trade.optionlib.assets.dividend import ScheduleEntry

def slice_schedule(full_schedule: List[ScheduleEntry], val_date: date, mat_date: date) -> List[ScheduleEntry]:
    """
    Return entries in full_schedule with entry.date in [val_date, mat_date].
    Assumes full_schedule is sorted by entry.date ascending and each entry has .date (datetime.date).
    """
    dates = [e.date for e in full_schedule]
    i0 = bisect_left(dates, val_date)
    i1 = bisect_right(dates, mat_date)
    return full_schedule[i0:i1]