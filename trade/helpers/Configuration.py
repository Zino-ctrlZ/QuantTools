import threading

## Using thread-local storage to store configuration variables
## We can access these variables from any module in the project
## Point of Configuration is to store the configuration variables, especially when used in Context
## The problem we were facing was two folds:
##      1. Configuration was a simple class. Therefore, any thread that accessed the Configuration class would have the same configuration settings. And changed it for all threads.
            # We don't want that. We want each thread to have its own configuration settings. Because each thread could be running a different context
##      2. Use thread local storage means we have achieved the thread specific configuration settings. BUT, we need to initialize the thread-local storage for each thread that is created.
            # We can do this by using the initialize_configuration() function. This function initializes the thread-local storage for each thread that is created. But it is a bit cumbersome to do this for each thread.
            # Therefore, we introduce the ConfigProxy class. This class initiates the variables when a thread is created and within the thread, Configuration is access. 
            # And we can access the configuration settings using the Configuration class.

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

class ConfigProxy:
    def __getattr__(self, name):
        """
        Automatically initialize and access thread-local variables.
        """
        if not hasattr(Configuration, name):
            initialize_configuration()
        return getattr(Configuration, name)

    def __setattr__(self, name, value):
        """
        Automatically initialize and set thread-local variables.
        """
        if not hasattr(Configuration, name):
            initialize_configuration()
        setattr(Configuration, name, value)



initialize_configuration()