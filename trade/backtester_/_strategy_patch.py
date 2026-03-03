## TODO: DELETE FILE IF UNUSED##
import pandas as pd
from datetime import datetime, date

class SignalCollector:
    COLLECT_SIGNALS = True

    def __init__(self):
        self.__signals = []

    @property
    def signals(self):
        return self.__signals

    def register_signal(self, signal):
        self.__signals.append(signal)

    @property
    def signal_df(self):
        if not self.__signals:
            return pd.DataFrame(columns=["type", "date", "price", "info"]).set_index("date")
        return pd.DataFrame(self.__signals)
    
    def get_signals(self):
        return self.__signals
    
    def clear_signals(self):
        self.__signals = []

    def get_signals_as_df(self):
        return self.signal_df


## DECORATOR APPROACH - Add signal collection to any strategy
def collect_signals_decorator(strategy_class):
    """
    Decorator to add signal collection capabilities to any strategy class.

    Usage:
        @collect_signals
        class MyStrategy(Strategy):
            def should_open(self):
                return self.data.Close[-1] > self.data.Close[-2]

            def should_close(self):
                return self.data.Close[-1] < self.data.Close[-2]

        or
        strategy_with_signals = collect_signals(MyStrategy)
    """

    class SignalCollectingStrategy(strategy_class):
        __signal_collection_enabled__ = True
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.signal_collector = SignalCollector()

        def next(self):
            super().next()
            # Only collect signals if the global flag is enabled
            if SignalCollector.COLLECT_SIGNALS:
                self._register_signals()

        def _register_signals(self):
            # Get date
            dt = getattr(self, "date", None)
            if dt is None:
                dt = self.data.index[-1]
            if dt is None or not isinstance(dt, (pd.Timestamp, datetime, date, str)):
                raise ValueError(f"Date is not properly set in the strategy or data. It is {dt}")

            # Check for open signal (if method exists)
            if hasattr(self, "is_open_signal"):
                if not callable(getattr(self, "should_open")):
                    raise NotImplementedError("The 'should_open' attribute must be callable.")
                open_signal = self.is_open_signal()
                if open_signal:
                    signal = {"type": "open", 
                              "date": dt, 
                              "price": self.data.Close[-1], 
                              "info": "Open signal generated",
                              "signal_value": 1}
                    self.signal_collector.register_signal(signal)
            else:
                raise NotImplementedError(
                    "The strategy must implement 'is_open_signal' method. "
                    "This is required for signal collection. "
                    "It should be a method that returns a boolean indicating "
                    "an open signal regardless of position status."
                )

            # Check for close signal (if method exists)
            if hasattr(self, "is_close_signal"):
                if not callable(getattr(self, "is_close_signal")):
                    raise NotImplementedError("The 'is_close_signal' attribute must be callable.")
                close_signal = self.is_close_signal()
                if close_signal:
                    signal = {
                        "type": "close",
                        "date": dt,
                        "price": self.data.Close[-1],
                        "info": "Close signal generated",
                        "signal_value": -1
                    }
                    self.signal_collector.register_signal(signal)
            else:
                raise NotImplementedError(
                    "The strategy must implement 'is_close_signal' method. "
                    "This is required for signal collection. "
                    "It should be a method that returns a boolean indicating "
                    "a close signal regardless of position status."
                )

            if not open_signal and not close_signal:
                ## Register no new signal
                signal = {"type": "none", 
                          "date": dt, 
                          "price": self.data.Close[-1], 
                          "info": "No signal generated",
                          "signal_value": 0}
                self.signal_collector.register_signal(signal)


    return SignalCollectingStrategy


## Helper function to check if a strategy has signal collection
def has_signal_collection(strategy_class):
    """
    Check if a strategy class has signal collection enabled.

    Works with both class and instance.
    """
    # If it's an instance, get its class
    if not isinstance(strategy_class, type):
        strategy_class = type(strategy_class)

    # Check for marker attribute
    if hasattr(strategy_class, "__signal_collection_enabled__"):
        return True

    # Fallback: check for signal collection methods
    return hasattr(strategy_class, "_register_signals") and hasattr(strategy_class, "signal_collector")