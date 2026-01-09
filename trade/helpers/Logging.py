import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from logging.handlers import TimedRotatingFileHandler

load_dotenv()


class TimezoneFormatter(logging.Formatter):
    """Custom formatter that converts timestamps to a specific timezone."""

    def __init__(self, fmt=None, datefmt=None, tz=None):
        super().__init__(fmt, datefmt)
        self.tz = ZoneInfo(tz) if tz else None

    def converter(self, timestamp):
        """Convert timestamp to timezone-aware datetime."""
        dt = datetime.fromtimestamp(timestamp, tz=ZoneInfo("UTC"))
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
    return os.environ["WORK_DIR"]  # Default to current path if no marker is found


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
    return Path(find_project_root(os.getcwd())) / "logs"


def _get_current_environment() -> str | None:
    """
    Get current environment from shared database context.

    Returns:
        Environment string ('prod', 'test', 'test-{name}') or None if not set
    """
    try:
        from dbase.database.db_utils import ENVIRONMENT_CONTEXT

        return ENVIRONMENT_CONTEXT.get("environment")
    except ImportError:
        return None


def setup_logger(
    filename,
    stream_log_level=None,
    file_log_level=None,
    log_file=None,
    remove_root=True,
    custom_logger_name=None,
    timezone=None,
) -> logging.Logger:
    """
    Set up a logger with console and file handlers, with environment-aware configuration.

    Args:
        filename: Base name for logger and log file. In test environments, suffixed with env name.
        stream_log_level: Console log level. If None, uses STREAM_LOG_LEVEL env var (default: ERROR).
        file_log_level: File log level. If None, uses FILE_LOG_LEVEL env var (default: INFO).
        log_file: Custom log file path. If None, uses {project_root}/logs/{filename}.log.
        remove_root: If True, removes all root logger handlers before setup.
        custom_logger_name: Custom logger name. If None, uses filename.
        timezone: Timezone string (e.g., 'America/New_York') for timezone-aware timestamps.

    Returns:
        logging.Logger: Configured logger with console and file handlers.

    Environment-Aware:
        - Production: {filename}.log with format "{timestamp} {filename} {level}: {message}"
        - Test: {filename}_{env}.log with format "{timestamp} [{env}] {filename} {level}: {message}"
        - Environment detected from dbase.database.SQLHelpers.ENVIRONMENT_CONTEXT if available

    Note:
        Log files rotate daily at midnight (3 backups). Handlers are cleaned on autoreload.
        Console Logging & File Logging Can be configured using STREAM_LOG_LEVEL and FILE_LOG_LEVEL in environment variables.
        Propagate to root logger can be set using PROPAGATE_TO_ROOT_LOGGER in environment variables.
        Example:
        STREAM_LOG_LEVEL = 'DEBUG'
        FILE_LOG_LEVEL = 'INFO'
        PROPAGATE_TO_ROOT_LOGGER = 'False'
    """
    project_root_log_dir = get_logger_base_location()

    # If custom logger name is None, use filename:

    stream_log_level = (
        getattr(logging, os.getenv("STREAM_LOG_LEVEL", "ERROR")) if stream_log_level is None else stream_log_level
    )
    file_log_level = getattr(logging, os.getenv("FILE_LOG_LEVEL", "INFO")) if file_log_level is None else file_log_level
    propagate_to_root_logger = (os.getenv("PROPAGATE_TO_ROOT_LOGGER", "False")).strip().lower() == "true"

    if custom_logger_name is None:
        custom_logger_name = filename
    # Remove all Root Handlers
    if remove_root:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.root.handlers = []
        date_strftime_format = "%Y-%m-%d %H:%M:%S"
        logging.basicConfig(
            stream=sys.stdout,
            format="%(asctime)s %(levelname)s: %(message)s",
            datefmt=date_strftime_format,
            level=logging.CRITICAL,
        )

    # Create a custom logger (root logger)
    logger = logging.getLogger(custom_logger_name)

    ## Ensure file name - to some capacity - exists.
    assert filename, "Please Create a FILENAME Variable"

    # Get current environment
    current_env = _get_current_environment()

    # Modify filename based on environment
    # Production: use base name (trading_bot.log)
    # Test: append environment suffix (trading_bot_test.log, trading_bot_test-mean-reversion.log)
    if current_env and current_env != "prod":
        notebook_name = f"{filename}_{current_env}"
    else:
        notebook_name = filename

    # Always remove existing handlers to prevent duplicates on autoreload
    # This ensures clean state even when modules are reloaded
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    logger.handlers = []

    # Define the log file path and create the directory if it doesn't exist
    os.makedirs(project_root_log_dir, exist_ok=True)
    log_file = os.path.join(project_root_log_dir, f"{notebook_name}.log")

    ## Create the log file if it doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, "w"):
            pass  # Just create the file

    # Set the log level for the root logger
    logger.setLevel(logging.DEBUG)

    # Create a formatter for log messages
    # Use original filename in formatter (not notebook_name which has environment suffix)
    # Add environment prefix for test environments only
    if current_env and current_env != "prod":
        formatter = logging.Formatter(
            f"%(asctime)s [{current_env}] {filename} %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    else:
        formatter = logging.Formatter(f"%(asctime)s {filename} %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    if timezone is not None:
        if current_env and current_env != "prod":
            formatter = TimezoneFormatter(
                fmt=f"%(asctime)s [{current_env}] {filename} %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                tz=timezone,
            )
        else:
            formatter = TimezoneFormatter(
                fmt=f"%(asctime)s {filename} %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", tz=timezone
            )

    # Create a console handler (logs to stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(stream_log_level)
    logger.addHandler(console_handler)

    # Optional: Create a file handler (logs to a file)
    if log_file:
        file_handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=3)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_log_level)
        logger.addHandler(file_handler)

    # Ensure the logger does not propagate messages to avoid duplicate logs
    logger.propagate = propagate_to_root_logger

    return logger


_logger = setup_logger("trade.helpers.Logging", stream_log_level=logging.INFO)
_logger.info(f'Logging Root Directory: {Path(find_project_root(os.getcwd()))/"logs"}')
