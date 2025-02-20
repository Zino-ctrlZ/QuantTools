from contextlib import contextmanager
import datetime
from datetime import datetime
import pandas as pd
from contextlib import contextmanager
from trade.helpers.helper import change_to_last_busday
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import BDay
# from trade.helpers.Configuration import Configuration, initialize_configuration
from trade.helpers.Configuration import ConfigProxy, initialize_configuration
Configuration = ConfigProxy()

## Change to Class ContextManager


@contextmanager
def Context(timewidth: str = None, timeframe: str = None, start_date: str = None, end_date: str = None, build_time: str = '16:00', print_context: bool = False):
    """
    This is a Context Manager that manages how the time window for the Stock, Option and other market related data classes create their timeseries and other important datapoints.
    
    Any end_date other than today will return end of day data for the end_date. Whereas, today will return data upto the current time.
    
    params:
    ---------

    timewidth: Expressed as full word for number of steps in timeframe. Eg timewidth 1, timeframe 'day' is daily data for 1 day. timewidth 2, timeframe 'day' is daily data for 2 days. Default is 1.
    timeframe: Expressed as full word for the timeframe. Eg timewidth 1, timeframe 'day' is daily data for 1 day. timewidth 2, timeframe 'day' is daily data for 2 days. Default is 'day'.
    start_date: Start date for the data. Default is 1 year ago from end_date. If time is passed alomg with date, it will be used as the start time. Default is 9:30am.
    end_date: End date for the data. Default is today. If time is passed along with date, it will be used as the end time. Default is 4pm (EOD). End date serves as build date for the data/model construction
        Other end_date behaviours if end_date is none:
            - If current time is before 9:30, end_date is yesterday as at 4pm.
            - If current time is after 4pm, end_date is today as at 4pm.
            - If current time is between 9:30 and 4pm, end_date is today as at current time.

        Other end_date behaviors if end_date is passed:
            - If no time is passed, end_date is set to EOD of the end_date.
            - If no time is passed, but end_date is set to today, time is set to current time
    
    build_time: Time of day to build the data. Default is 4pm.
    print_context: Print the context settings. Default is False.

    """
    initialize_configuration()
    try:
        if timeframe is not None:
            Configuration.timeframe = str(timeframe)
        else:
            Configuration.timeframe = 'day'

        if timewidth is not None:
            Configuration.timewidth = timewidth
        else:
            Configuration.timewidth = '1'

        if start_date is not None:
            start_date = pd.to_datetime(start_date )
            Configuration.start_date = datetime.strftime(
                start_date, format='%Y-%m-%d %H:%M:%S')
        else:
            if end_date is not None:
                start_date = pd.to_datetime(end_date + ' 9:30')
                start_date = start_date - relativedelta(years=1)
                Configuration.start_date = datetime.strftime(
                    start_date, format='%Y-%m-%d %H:%M:%S')
            else:
                today = datetime.today().replace(hour=pd.Timestamp('9:30').time().hour, minute=pd.Timestamp('9:30').time().minute, 
                                                 second=pd.Timestamp('9:30').time().second, microsecond=0)- relativedelta(years=1)
                Configuration.start_date = datetime.strftime(
                    today, format='%Y-%m-%d %H:%M:%S')

        if end_date is not None:
            ## TEMP (MAYBE): Enforcing np Non-business day for now
            end_date = change_to_last_busday(pd.to_datetime(end_date)
)
            ## If no time is passed and date is today set to current time if btwn 9:30 & 4pm
            if datetime.today().date() == end_date.date() and end_date.time() == pd.Timestamp('00:00').time():
                end_date = datetime.now()
            
            ## If no time is passed and not today, set default to build_time 
            if end_date.time() == pd.Timestamp('00:00').time() and end_date.date() != datetime.today().date():
                ##TEMP: For now, all passed dates will be set to EOD
                build_time = '16:00'
                end_date = end_date.replace(hour=pd.Timestamp(build_time).time().hour, minute=pd.Timestamp(build_time).time().minute, second=pd.Timestamp(build_time).time().second, microsecond=0)
            Configuration.end_date = datetime.strftime(
                end_date, format='%Y-%m-%d %H:%M:%S')
        else:

            ## Setting the end date to EOD today if current time is later than 4pm
            build_time = '16:00'
            if datetime.today().time() > pd.Timestamp(build_time).time():
                today = datetime.today().replace(hour=pd.Timestamp(build_time).time().hour, minute=pd.Timestamp(build_time).time().minute, second=pd.Timestamp(build_time).time().second, microsecond=0)
            
            ## Setting end date to prev day EOD if current time is before 9:30am
            elif datetime.today().time() < pd.Timestamp('9:30').time():
                today = (datetime.today() - BDay(1)).replace(hour=pd.Timestamp('16:00').time().hour, minute=pd.Timestamp('16:00').time().minute, second=pd.Timestamp('9:30').time().second, microsecond=0)
            
            ## Setting end date to now if current time is between 9:30am and 4pm
            else:
                today = datetime.today()

            Configuration.end_date = datetime.strftime(
                today, format='%Y-%m-%d %H:%M:%S')

        build_time = pd.Timestamp(build_time)
        if print_context:
            print(f"""
            Settings in this Context:
            Start Date: {Configuration.start_date}
            End Date: {Configuration.end_date}
            Multiplier: {Configuration.timewidth}
            Timespan: {Configuration.timeframe}
            """)
        #context_values = {'timewidth': timewidth, 'timeframe': timeframe, 'start_date': start_date, 'end_date': end_date}
        yield

    finally:
        
        Configuration.timewidth = None
        Configuration.timeframe = None
        Configuration.start_date = None
        Configuration.end_date = None


def clear_context():
    Configuration.timewidth = None
    Configuration.timeframe = None
    Configuration.start_date = None
    Configuration.end_date = None
    return 'Context Cleared'