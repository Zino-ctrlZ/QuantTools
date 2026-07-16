"""Environment-aware logger setup with refreshable handler registry.

Provides ``setup_logger`` for console/file handlers whose log file name and
format track ``ENVIRONMENT_CONTEXT``. Loggers created via ``setup_logger`` are
tracked so ``refresh_all_loggers`` can rebuild handlers in place after the
environment changes (same ``logging.getLogger`` singletons modules already hold).

Core Functions:
    setup_logger: Create or rebuild a named logger with env-aware handlers.
    refresh_all_loggers: Rebuild every registered logger for the current env.
    find_loggers_by_pattern / find_logger_names_by_pattern: Lookup helpers.

Caching Strategy:
    ``_LOGGER_REGISTRY`` maps logger name -> rebuild kwargs. Handlers are not
    cached across env changes; refresh closes and recreates them.

Usage:
    >>> from trade.helpers.Logging import setup_logger
    >>> logger = setup_logger("my_module")
    >>> # set_environment_context(...) triggers refresh_all_loggers via dbase listener
"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Union
from logging.handlers import TimedRotatingFileHandler

load_dotenv()

## logger_name -> kwargs needed to rebuild handlers after ENVIRONMENT_CONTEXT changes
_LOGGER_REGISTRY: Dict[str, Dict[str, Any]] = {}


class TimezoneFormatter(logging.Formatter):
    """Custom formatter that converts timestamps to a specific timezone."""

    def __init__(self, fmt=None, datefmt=None, tz=None):
        """Initialize formatter with optional IANA timezone name.

        Args:
            fmt: Log message format string.
            datefmt: Date format string for ``asctime``.
            tz: IANA timezone name (e.g. ``America/New_York``), or None for local.
        """
        super().__init__(fmt, datefmt)
        self.tz = ZoneInfo(tz) if tz else None

    def converter(self, timestamp):
        """Convert timestamp to timezone-aware datetime.

        Args:
            timestamp: Unix timestamp from the log record.

        Returns:
            ``time.struct_time`` in the configured timezone (or UTC if unset).
        """
        dt = datetime.fromtimestamp(timestamp, tz=ZoneInfo("UTC"))
        if self.tz:
            dt = dt.astimezone(self.tz)
        return dt.timetuple()


def find_logger_names_by_pattern(pattern: str) -> List[str]:
    """Find all logger names that start with the given pattern.

    Args:
        pattern: Prefix matched against ``logging.Logger.manager.loggerDict`` keys.

    Returns:
        Logger names that start with ``pattern``.
    """
    return [
        name
        for name in logging.Logger.manager.loggerDict.keys()
        if name.startswith(pattern)
    ]


def find_loggers_by_pattern(pattern: str) -> List[logging.Logger]:
    """Find all loggers whose names start with the given pattern.

    Args:
        pattern: Prefix matched against registered logger names.

    Returns:
        Logger instances for names that start with ``pattern``.
    """
    return [
        logging.getLogger(name)
        for name in logging.Logger.manager.loggerDict.keys()
        if name.startswith(pattern)
    ]


def find_project_root(current_path: Path, marker: str = ".git") -> Union[Path, str]:
    """Find the current project root by looking for a marker file in parent directories.

    Args:
        current_path: Path to start searching from.
        marker: Directory/file name that marks the project root.

    Returns:
        Project root ``Path``, or ``WORK_DIR`` env value if no marker is found.
    """
    if isinstance(current_path, str):
        current_path = Path(current_path)

    if (current_path / marker).exists():
        return current_path

    for parent in current_path.parents:
        if (parent / marker).exists():
            return parent
    return os.environ["WORK_DIR"]  # Default to current path if no marker is found


def change_logger_stream_level(logger: logging.Logger, level: int) -> None:
    """Change the logger and its stream handlers to a new level.

    Args:
        logger: Logger object to change the stream level for.
        level: New logging level (e.g., logging.INFO, logging.DEBUG).
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(level)


def get_logger_base_location() -> Path:
    """Get the base location for log files.

    Returns:
        ``{project_root}/logs`` as a ``Path``.
    """
    return Path(find_project_root(os.getcwd())) / "logs"


def _get_current_environment() -> Optional[str]:
    """Get current environment from shared database context.

    Returns:
        Environment string ('prod', 'test', 'test-{name}') or None if not set.
    """
    try:
        from dbase.database.db_utils import ENVIRONMENT_CONTEXT

        return ENVIRONMENT_CONTEXT.get("environment")
    except ImportError:
        return None


def refresh_all_loggers(*args: Any, **kwargs: Any) -> None:
    """Rebuild handlers on every logger previously created via ``setup_logger``.

    Mutates existing ``logging.getLogger`` singletons in place so module-level
    ``logger`` references keep writing with the current environment file/format.
    Extra ``*args`` / ``**kwargs`` are ignored so this matches
    ``register_on_environment_changed``'s ``(old_env, new_env)`` signature.

    Args:
        *args: Ignored; accepted for callback compatibility.
        **kwargs: Ignored; accepted for callback compatibility.
    """
    for name, spec in list(_LOGGER_REGISTRY.items()):
        setup_logger(
            spec["filename"],
            stream_log_level=spec["stream_log_level"],
            file_log_level=spec["file_log_level"],
            custom_logger_name=name,
            timezone=spec["timezone"],
            dir=spec["dir"],
            remove_root=False,
        )


def _register_env_change_refresh() -> None:
    """Register ``refresh_all_loggers`` with dbase when that package is available.

    Soft-depends on FinanceDatabase so QuantTools still imports without dbase.
    ``register_on_environment_changed`` de-dupes, so repeat calls are safe.
    """
    try:
        from dbase.database.db_utils import register_on_environment_changed
    except ImportError:
        return
    register_on_environment_changed(refresh_all_loggers)


def setup_logger(
    filename: str,
    stream_log_level: Optional[int] = None,
    file_log_level: Optional[int] = None,
    log_file: Optional[str] = None,
    remove_root: bool = True,
    custom_logger_name: Optional[str] = None,
    timezone: Optional[str] = None,
    dir: Optional[Union[str, Path]] = None,
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
        dir: Optional log directory override. Defaults to project ``logs/``.

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
    project_root_log_dir = dir or get_logger_base_location()

    # If custom logger name is None, use filename:

    stream_log_level = (
        getattr(logging, os.getenv("STREAM_LOG_LEVEL", "ERROR")) if stream_log_level is None else stream_log_level
    )
    file_log_level = getattr(logging, os.getenv("FILE_LOG_LEVEL", "INFO")) if file_log_level is None else file_log_level
    propagate_to_root_logger = (os.getenv("PROPAGATE_TO_ROOT_LOGGER", "False")).strip().lower() == "true"

    if custom_logger_name is None:
        custom_logger_name = filename

    ## Track rebuild kwargs so refresh_all_loggers can re-apply current env in place
    _LOGGER_REGISTRY[custom_logger_name] = {
        "filename": filename,
        "stream_log_level": stream_log_level,
        "file_log_level": file_log_level,
        "timezone": timezone,
        "dir": dir,
    }

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


_register_env_change_refresh()

_logger = setup_logger("trade.helpers.Logging", stream_log_level=logging.INFO)
_logger.info(f'Logging Root Directory: {Path(find_project_root(os.getcwd()))/"logs"}')
