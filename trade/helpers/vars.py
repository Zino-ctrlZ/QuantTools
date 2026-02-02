from typing import List, Callable
_CUSTOM_ON_EXIT_BUCKET: List[Callable] = []

def register_on_exit(handler: Callable) -> None:
    """
    Register a function to be called upon program exit.

    Parameters:
    ----------
    handler : Callable
        The function to be called on exit.
    """
    global _CUSTOM_ON_EXIT_BUCKET
    _CUSTOM_ON_EXIT_BUCKET.append(handler)

def run_on_exit_handlers() -> None:
    """Run all registered on-exit handlers."""
    global _CUSTOM_ON_EXIT_BUCKET
    for handler in _CUSTOM_ON_EXIT_BUCKET:
        try:
            handler()
        except Exception as e:
            print(f"Error running on-exit handler {handler.__name__}: {e}")

def get_on_exit_bucket() -> List[Callable]:
    """Get the list of registered on-exit handlers."""
    global _CUSTOM_ON_EXIT_BUCKET
    return _CUSTOM_ON_EXIT_BUCKET

def clear_on_exit_bucket() -> None:
    """Clear the list of registered on-exit handlers."""
    global _CUSTOM_ON_EXIT_BUCKET
    _CUSTOM_ON_EXIT_BUCKET = []

SECONDS_IN_YEAR = 365.0 * 24.0 * 3600.0
SECONDS_IN_DAY = 24.0 * 3600.0