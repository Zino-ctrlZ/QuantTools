# class Configuration():
#     """
#     This Class holds important Global values utilized in Context Manager and Other important Classes such as: Stock, Options etc.

#     """
#     timewidth = None
#     timeframe = None
#     start_date = None
#     end_date = None

import threading

# Thread-local storage
Configuration = threading.local()

# Initialization function
def initialize_configuration():
    if not hasattr(Configuration, 'timewidth'):
        Configuration.timewidth = None
    if not hasattr(Configuration, 'timeframe'):
        Configuration.timeframe = None
    if not hasattr(Configuration, 'start_date'):
        Configuration.start_date = None
    if not hasattr(Configuration, 'end_date'):
        Configuration.end_date = None

