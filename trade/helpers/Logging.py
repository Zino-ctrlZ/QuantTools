import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo 
from dotenv import load_dotenv
load_dotenv()
print("""
Console Logging & File Logging Can be configured using STREAM_LOG_LEVEL and FILE_LOG_LEVEL in environment variables.
Propagate to root logger can be set using PROPAGATE_TO_ROOT_LOGGER in environment variables.
Example:
STREAM_LOG_LEVEL = 'DEBUG'
FILE_LOG_LEVEL = 'INFO'
PROPAGATE_TO_ROOT_LOGGER = 'False'
""")
from logging.handlers import TimedRotatingFileHandler

class TimezoneFormatter(logging.Formatter):
    """Custom formatter that converts timestamps to a specific timezone."""
    def __init__(self, fmt=None, datefmt=None, tz=None):
        super().__init__(fmt, datefmt)
        self.tz = ZoneInfo(tz) if tz else None

    def converter(self, timestamp):
        """Convert timestamp to timezone-aware datetime."""
        dt = datetime.fromtimestamp(timestamp, tz=ZoneInfo('UTC'))
        if self.tz:
            dt = dt.astimezone(self.tz)
        return dt.timetuple()

def find_project_root(current_path: Path, marker=".git"):
    """
    Find the current project root by looking for a marker file in the parent directories.
    """
    if isinstance(current_path, str):
        current_path = Path(current_path)
        
    if (current_path / marker).exists():
        return current_path

    for parent in current_path.parents:
        if (parent / marker).exists():
            return parent
    return os.environ['WORK_DIR']  # Default to current path if no marker is found


def change_logger_stream_level(logger: logging.Logger, level: int):
    """
    Change the logger stream level.

    params:
    --------
    logger: Logger object to change the stream level for.
    level: New logging level (e.g., logging.INFO, logging.DEBUG).
    
    returns:
    --------
    None
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(level)

def get_logger_base_location() -> Path:
    """
    Get the base location for log files.
    """
    return Path(find_project_root(os.getcwd()))/"logs"

def setup_logger(filename,stream_log_level = None, 
                 file_log_level = None, 
                 log_file=None, 
                 remove_root = True, 
                 custom_logger_name = None, timezone = None) -> logging.Logger:


    project_root_log_dir = get_logger_base_location()

    # If custom logger name is None, use filename:

    stream_log_level = getattr(logging, os.getenv('STREAM_LOG_LEVEL', 'ERROR')) if stream_log_level is None else stream_log_level
    file_log_level = getattr(logging, os.getenv('FILE_LOG_LEVEL', 'INFO')) if file_log_level is None else file_log_level
    propagate_to_root_logger = (os.getenv('PROPAGATE_TO_ROOT_LOGGER', 'False')).strip().lower() == 'true'

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
    assert filename, 'Please Create a FILENAME Variable'
    notebook_name = filename

    # Always remove existing handlers to prevent duplicates on autoreload
    # This ensures clean state even when modules are reloaded
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    logger.handlers = []

    # Define the log file path
    os.makedirs(project_root_log_dir, exist_ok=True)
    log_file = os.path.join(project_root_log_dir, f'{notebook_name}.log')

    ## Create the log file if it doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, 'w'):
            pass  # Just create the file


    # Set the log level for the root logger
    logger.setLevel(logging.DEBUG)

    # Create a formatter for log messages
    formatter = logging.Formatter(
        f"%(asctime)s {notebook_name} %(levelname)s: %(message)s", 
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if timezone is not None: 
        formatter = TimezoneFormatter(fmt=f"%(asctime)s {notebook_name} %(levelname)s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S', tz=timezone)

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
    logger.propagate = propagate_to_root_logger

    return logger


_logger = setup_logger('trade.helpers.Logging', stream_log_level = logging.INFO)
_logger.info(f'Logging Root Directory: {Path(find_project_root(os.getcwd()))/"logs"}')