from datetime import date, timedelta
from queue import Queue
from typing import Dict, Optional

from EventDriven.event import Event

class EventScheduler:
    def __init__(self, start_date: date, end_date: date):
        self.events_map: Dict[date, Queue] = {
            d: Queue(maxsize=0) for d in self._date_range(start_date, end_date)
        }
        self.current_date = start_date
        self.end_date = end_date
        self.date_iterator = iter(self.events_map.keys())  # Iterator for fast traversal
    
    def _date_range(self, start: date, end: date):
        """Generate a range of dates from start to end (inclusive)."""
        return [start + timedelta(days=i) for i in range((end - start).days + 1)]
    
    def get_current_queue(self) -> Queue:
        """Returns the queue for the current date."""
        return self.events_map[self.current_date]

    def empty(self) -> bool:
        """Checks if the current day's queue has any events."""
        return not self.get_current_queue().empty()

    def has_scheduled_events(self) -> bool:
        """Checks if any queue (current or future) has events scheduled."""
        return any(not q.empty() for q in self.events_map.values())

    def advance_date(self) -> Optional[date]:
        """Moves to the next date in sequence, if available."""
        try:
            self.current_date = next(self.date_iterator)
            return self.current_date
        except StopIteration:
            return None  # No more dates left
    
    def put(self, event: Event):
        """Adds an event to the queue of the current date."""
        self.get_current_queue().put(event)

    def schedule_event(self, event_date: date, event: Event):
        """Schedules an event for a specific future date."""
        if event_date < self.current_date:
            raise ValueError(f"Cannot schedule event to past date {event_date}.")
        if event_date not in self.events_map:
            raise ValueError(f"Date {event_date} is out of backtest range.")
        self.events_map[event_date].put(event)
