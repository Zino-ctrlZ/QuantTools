import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from pandas import DatetimeIndex
from pandas.tseries.holiday import USFederalHolidayCalendar
from queue import Queue
from typing import Dict, Optional

from EventDriven.event import Event

class EventScheduler:
    def __init__(self, start_date, end_date, schedules = ['NYSE'] ):
        """
        Initializes the event scheduler with a range of dates.sss
        start_date: can be date or date string format
        end_date: can be date or date string format
        schedules: list of market calendars to use for scheduling events default: ['NYSE'], note: this performs a union of all the valid days in the schedules provided (future iteration)
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.valid_days = np.array([])
        #leave for future iterations
        # for schedule in schedules:
        #     assert schedule in mcal.get_calendar_names(), f"Invalid calendar name: {schedule}"
        #     self.valid_days = np.append(self.valid_days, mcal.get_calendar(schedule).valid_days(start_date=self.start_date, end_date=self.end_date))
        
        # self.market_dates = pd.to_datetime(np.unique(self.valid_days))
        self.market_dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        self.events_map: Dict[str, Queue] = {
            d.strftime('%Y-%m-%d'): Queue(maxsize=0) for d in self.market_dates
        }
        self.date_iterator = iter(self.events_map.keys())  # Iterator for fast traversal
        self.current_date = next(self.date_iterator)  # Start with the first date
        self.events_dict = []
    
    
    
    def get_current_queue(self) -> Queue:
        """Returns the queue for the current date."""
        return self.events_map[self.current_date]

    def empty(self) -> bool:
        """Checks if the current day's queue has any events."""
        return not self.get_current_queue().empty()

    def has_scheduled_events(self) -> bool:
        """Checks if any queue (current or future) has events scheduled."""
        return any(not q.empty() for q in self.events_map.values())

    def advance_date(self) -> Optional[DatetimeIndex]:
        """Moves to the next date in sequence, if available."""
        try:
            self.current_date = next(self.date_iterator)
            print(f"Advancing to next date: {self.current_date}")
            return self.current_date
        except StopIteration:
            return None  # No more dates left
    
    def put(self, event: Event):
        """Adds an event to the queue of the current date."""
        self.get_current_queue().put(event)
        self.store_event(event)

    def schedule_event(self, event_date, event: Event):
        """
        Schedules an event for a specific future date.
        date: can be any date string format
        Event: Event object to be scheduled
        """
        event_date = pd.to_datetime(event_date)
        event_date_str = self.clean_date(event_date)
        if event_date < pd.to_datetime(self.current_date):
            raise ValueError(f"Cannot schedule event to past date {event_date}.")
        if event_date_str not in self.events_map:
            raise ValueError(f"Date {event_date} is out of backtest range.")
        
        if event_date_str not in self.events_map:
            print(f"Event date {event_date_str} not found in backtest range.")
            return 
        
        self.events_map[event_date_str].put(event)
        self.store_event(event)
        

    def get_next_trading_day(self)-> DatetimeIndex:
        """
        Returns the next business day for the current date.
        """
        return self.market_dates[self.market_dates > pd.to_datetime(self.current_date)].min()
    
    def clean_date(self, date):
        """
        Cleans date string to ensure it is in the correct format
        """
        return pd.to_datetime(date).strftime('%Y-%m-%d')
    
    def store_event(self, event: Event):
        """
        Stores an event in the events dictionary.
        """
        if event.type != 'MARKET':
            self.events_dict.append({"event": event, "date": self.current_date})
        