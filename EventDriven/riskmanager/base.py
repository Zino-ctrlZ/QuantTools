"""Compatibility shim for legacy imports.

Importing from EventDriven.riskmanager.base forwards to new_base.
"""

from .new_base import *  # noqa: F403
