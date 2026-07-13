"""Persistence stores for position limits and sizing metadata.

Provides in-memory and database-backed implementations used by limits cogs
to save and retrieve ``PositionLimits`` and position metadata independently
from cog business logic.

Core Classes:
    InMemoryLimitsStore: Dict-backed limits storage for backtests.
    InMemoryMetadataStore: Dict-backed metadata storage for backtests.
    InMemoryPositionStore: Combined in-memory limits and metadata store.
    DatabaseLimitsStore: Limits persistence via the ``limits`` table.
    DatabaseMetadataStore: Metadata persistence via ``position_metadata``.
    DatabasePositionStore: Combined database-backed store with optional cache.

Usage:
    >>> from EventDriven.riskmanager.position.stores import InMemoryPositionStore
    >>> store = InMemoryPositionStore()
"""

from EventDriven.riskmanager.position.stores.limits_store import (
    DatabaseLimitsStore,
    DatabaseMetadataStore,
    DatabasePositionStore,
    InMemoryLimitsStore,
    InMemoryMetadataStore,
    InMemoryPositionStore,
    LimitsStore,
    MetadataStore,
    PositionStore,
    PositionStoreKey,
    StoreVerificationError,
    build_position_store,
    disable_storing_to_db,
    enable_storing_to_db,
    reset_storing_to_db,
    ticker_from_trade_id,
)
from EventDriven.riskmanager.position.stores.tmp_limits_cogs import (
    TmpLimitsAndSizingCog,
)

__all__ = [
    "DatabaseLimitsStore",
    "DatabaseMetadataStore",
    "DatabasePositionStore",
    "InMemoryLimitsStore",
    "InMemoryMetadataStore",
    "InMemoryPositionStore",
    "LimitsStore",
    "MetadataStore",
    "PositionStore",
    "PositionStoreKey",
    "StoreVerificationError",
    "TmpLimitsAndSizingCog",
    "TmpLiveLimitsAndSizingCog",
    "build_position_store",
    "disable_storing_to_db",
    "enable_storing_to_db",
    "reset_storing_to_db",
    "ticker_from_trade_id",
]


def __getattr__(name: str):
    """Lazily resolve live-cog aliases to avoid circular imports.

    Args:
        name: Attribute name requested on this package.

    Returns:
        ``LiveCOGLimitsAndSizingCog`` when ``TmpLiveLimitsAndSizingCog`` is requested.

    Raises:
        AttributeError: If ``name`` is not defined on this package.
    """
    if name == "TmpLiveLimitsAndSizingCog":
        ## Lazy: live_cogs.limits imports this package's limits_store submodule
        from EventDriven.riskmanager.position.live_cogs.limits import LiveCOGLimitsAndSizingCog

        return LiveCOGLimitsAndSizingCog
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
