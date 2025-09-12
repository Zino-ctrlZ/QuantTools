import platform
import multiprocessing
from multiprocessing.context import BaseContext
from typing import Tuple

## OS Type
_SYS = platform.system()
IS_WINDOWS = _SYS == "Windows"
IS_LINUX   = _SYS == "Linux"
IS_MAC     = _SYS == "Darwin"
UNKNOWN_OS = not (IS_WINDOWS or IS_LINUX or IS_MAC)

## MULTI-PROCESSING CONTEXT
def _default_mp_context() -> Tuple[BaseContext, str]:
    """
    Get the default multiprocessing context based on the operating system.
    Returns:
        A tuple containing the multiprocessing context and its string representation.
    """

    if IS_WINDOWS:
        MP_CONTEXT = multiprocessing.get_context('spawn')
        MP_CTX_STR = 'spawn'
    else:
        MP_CONTEXT = multiprocessing.get_context('fork')
        MP_CTX_STR = 'fork'

    return MP_CONTEXT, MP_CTX_STR

MP_CONTEXT, MP_CTX_STR = _default_mp_context()


def ensure_global_start_method():
    """
    Optional: set the global start method once, in __main__ only.
    Safe no-op if already set to the same method.
    """
    current = multiprocessing.get_start_method(allow_none=True)
    if current is None:
        multiprocessing.set_start_method(MP_CTX_STR, force=False)

Process = MP_CONTEXT.Process
Pool = MP_CONTEXT.Pool
Queue = MP_CONTEXT.Queue
Pipe = MP_CONTEXT.Pipe
Event = MP_CONTEXT.Event
Lock = MP_CONTEXT.Lock

