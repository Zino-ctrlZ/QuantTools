import logging
import sys
import os
# FILENAME = 'Logging.ipynb'
from logging.handlers import TimedRotatingFileHandler



def setup_logger(filename,stream_log_level=logging.ERROR, file_log_level=logging.INFO, log_file=None, remove_root = True, custom_logger_name = None):
    # If custom logger name is None, use filename:
    if custom_logger_name == None:
        custom_logger_name = filename
    # Remove all Root Handlers
    if remove_root:

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.root.handlers = []
        date_strftime_format = '%Y-%m-%d %H:%M:%S'
        logging.basicConfig(stream = sys.stdout,format="%(asctime)s %(levelname)s: %(message)s", datefmt=date_strftime_format, level = logging.CRITICAL)
    

    # Create a custom logger (root logger)
    logger = logging.getLogger(custom_logger_name)
    
    ## Ensure file name - to some capacity - exists.
    assert filename, f'Please Create a FILENAME Variable'
    notebook_name = filename


    # Define the log file path
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{notebook_name}.log')


    # Remove all existing handlers (in case the logger was already configured)
    logger.handlers = []


    # Set the log level for the root logger
    logger.setLevel(logging.DEBUG)

    # Create a formatter for log messages
    formatter = logging.Formatter(
        f"%(asctime)s {notebook_name} %(levelname)s: %(message)s", 
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create a console handler (logs to stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(stream_log_level)
    logger.addHandler(console_handler)

    # Optional: Create a file handler (logs to a file)
    if log_file:
        file_handler = TimedRotatingFileHandler(log_file, when = 'midnight', interval = 1, backupCount= 3)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_log_level)
        logger.addHandler(file_handler)

    # Ensure the logger does not propagate messages to avoid duplicate logs
    logger.propagate = False

    return logger