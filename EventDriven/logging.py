import logging

from trade.helpers.Logging import setup_logger, find_loggers_by_pattern, change_logger_stream_level

_BASE_LOGGING_LEVEL = "WARNING"
LOGGING_LEVEL = "WARNING"

logger = setup_logger("EventDriven.logging", stream_log_level=LOGGING_LEVEL)


def get_logging_level() -> str:
    """Return the current module-level logging level."""
    return LOGGING_LEVEL


def set_logging_level(level: str) -> None:
    """Set the module-level logging level (does not propagate to loggers)."""
    global LOGGING_LEVEL
    LOGGING_LEVEL = level


def get_eventdriven_loggers() -> list[logging.Logger]:
    """Retrieve all loggers under 'EventDriven'."""
    return find_loggers_by_pattern("EventDriven")


def change_all_eventdriven_loggers_level(level: str) -> None:
    """Change the stream log level for every logger under 'EventDriven'."""
    int_level = getattr(logging, level.upper(), logging.WARNING)
    for _logger in find_loggers_by_pattern("EventDriven"):
        change_logger_stream_level(_logger, int_level)


def reset_base_logging_level() -> None:
    """Reset LOGGING_LEVEL back to the package default ('WARNING')."""
    global LOGGING_LEVEL
    LOGGING_LEVEL = _BASE_LOGGING_LEVEL


def change_all_to_base_logging_level() -> None:
    """Apply the current base LOGGING_LEVEL to all EventDriven loggers."""
    change_all_eventdriven_loggers_level(_BASE_LOGGING_LEVEL)


def change_specific_logger(name: str, level: str) -> None:
    """Change the stream log level for a single EventDriven logger.

    Args:
        name: Logger name, either fully-qualified (e.g. ``"EventDriven.types"``)
            or short form (e.g. ``"types"``).  If the name does not start with
            ``"EventDriven."`` it is automatically prefixed.
        level: Target log level string, e.g. ``"DEBUG"``, ``"WARNING"``.

    Raises:
        ValueError: If no logger matching *name* has been registered yet.
    """
    if not name.startswith("EventDriven."):
        name = f"EventDriven.{name}"

    existing = find_loggers_by_pattern(name)
    if not existing:
        raise ValueError(
            f"No logger named '{name}' has been registered. Ensure the corresponding module has been imported."
        )

    int_level = getattr(logging, level.upper(), logging.WARNING)
    for _logger in existing:
        change_logger_stream_level(_logger, int_level)
