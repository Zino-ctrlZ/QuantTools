"""Compatibility shim for legacy imports.

Importing from EventDriven.portfolio forwards to new_portfolio.
"""

from .new_portfolio import *  # noqa: F403
