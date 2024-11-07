import datetime
from datetime import datetime
import pandas as pd


def parse_date(date_str):
    # Try parsing the date with different formats
    possible_formats = ['%Y-%m-%d', '%m/%d/%Y',
                        '%d-%b-%Y', '%d%b%y', '%Y.%m.%d', '%Y-%m-%d %H:%M:%S']
    for date_format in possible_formats:
        try:
            return pd.to_datetime(date_str)
        except ValueError:
            pass

    # If none of the formats match, raise an exception
    raise ValueError(f"Unable to parse date: {date_str}")


def parse_time(time_str):
    # Try parsing the time with different formats
    possible_formats = ['%H:%M:%S', '%I:%M%p', '%I%p']
    for time_format in possible_formats:
        try:
            return datetime.strptime(time_str, time_format).time()
        except ValueError:
            pass

    # If none of the formats match, raise an exception
    raise ValueError(f"Unable to parse time: {time_str}")
