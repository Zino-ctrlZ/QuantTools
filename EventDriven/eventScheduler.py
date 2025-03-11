import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from pandas import DatetimeIndex
from pandas.tseries.holiday import USFederalHolidayCalendar
from queue import Queue
from typing import Dict, Optional

from EventDriven.event import Event
from trade.helpers.Logging import setup_logger

class EventScheduler:
    def __init__(self, start_date, end_date ):
        """
        Initializes the event scheduler with a range of dates.sss
        start_date: can be date or date string format
        end_date: can be date or date string format
        schedules: list of market calendars to use for scheduling events default: ['NYSE'], note: this performs a union of all the valid days in the schedules provided (future iteration)
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.market_dates = pd.bdate_range(start=self.start_date, end=self.end_date)
        self.events_map: Dict[str, Queue] = {
            self.clean_date(d): Queue(maxsize=0) for d in self.market_dates
        }
        self.current_date = self.clean_date(self.start_date)
        self.events_dict = []
        self.logger = setup_logger('OptionSignalEventScheduler')
    
    
    @property
    def events(self):
        return pd.DataFrame(self.events_dict).set_index('datetime').sort_index()
    
    def get_queue(self, date) -> Optional[Queue]:
        """Returns the queue for a specific date."""
        return self.events_map[self.clean_date(date)]
        
    def get_current_queue(self) -> Queue | None:
        """Returns the queue for the current date."""
        return self.events_map.get(self.current_date, None)

    def empty(self, date = None) -> bool:
        """Checks if the current day's queue has any events."""
        if date is not None:
            return self.get_queue(date).empty()
        return not self.get_current_queue().empty()

    def has_scheduled_events(self) -> bool:
        """Checks if any queue (current or future) has events scheduled."""
        return any(not q.empty() for q in self.events_map.values())

    def get_next_nonempty_queue(self) -> Optional[Queue]:
        """Returns the next non-empty queue, if available."""
        for date, queue in self.events_map.items():
            if not queue.empty():
                return (date,queue)
        
    def advance_date(self, date=None) -> Optional[DatetimeIndex | None]:
        """Moves to the next available date in the sequence, skipping over missing ones."""
        try:
            if date:
                next_date = self.clean_date(date + pd.offsets.BDay(1))
            else:
                current_date_idx = list(self.events_map.keys()).index(self.current_date)
                next_date = list(self.events_map.keys())[current_date_idx + 1] if current_date_idx + 1 < len(self.events_map) else None

            if next_date:
                self.current_date = next_date
                self.logger.info(f"Advancing to next date: {self.current_date}")
                return self.current_date
            else:
                self.current_date = None
                self.logger.info("No more dates left.")
                return None  # No more dates left

        except (ValueError, IndexError):
            self.logger.error("Error advancing date. Possibly out of range.")
            return None
    
    def put(self, event: Event):
        """Adds an event to the queue of the current date."""
        self.get_current_queue().put(event)
        self.store_event(event)
        self.logger.info(f"Event added to {self.current_date} queue: {event}")

    def schedule_event(self, event_date, event: Event):
        """
        Schedules an event for a specific future date.
        date: can be any date string format
        Event: Event object to be scheduled
        """
        event_date = pd.to_datetime(event_date)
        event_date_str = self.clean_date(event_date)
        current_date = pd.to_datetime(self.current_date)
        if event_date < pd.to_datetime(current_date):
            self.logger.error(f"Cannot schedule event to past date {event}.")
        
        if event_date_str not in self.events_map:
            print(f"Event date {event_date_str} not found in backtest range.")
            self.logger.error(f"Event date {event_date_str} not found in backtest range")
            # self.events_map[event_date_str] = Queue(maxsize=0)
            return
        
        self.logger.info(f"Scheduling event for {event_date_str} queue: {event}")
        self.events_map[event_date_str].put(event)
        self.store_event(event)
        
    
    def clean_date(self, date):
        """
        Cleans date string to ensure it is in the correct format
        """
        return pd.to_datetime(date).strftime('%Y%m%d')
    
    def store_event(self, event: Event):
        """
        Stores an event in the events dictionary.
        """
        self.events_dict.append(event.__dict__)
        