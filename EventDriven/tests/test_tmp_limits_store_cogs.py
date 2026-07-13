"""Tests for limits cogs backed by position stores."""

from __future__ import annotations

import math
import uuid
from datetime import datetime

import pytest

from EventDriven.configs.core import LimitsEnabledConfig
from EventDriven.dataclasses.limits import PositionLimits
from EventDriven.riskmanager.position.cogs.limits import _LimitsMetaData, metadata_from_store_payload
from EventDriven.riskmanager.position.live_cogs.limits import LiveCOGLimitsAndSizingCog
from EventDriven.riskmanager.position.stores.limits_store import (
    DatabaseLimitsStore,
    DatabaseMetadataStore,
    disable_storing_to_db,
    reset_storing_to_db,
)
from EventDriven.riskmanager.position.stores.tmp_limits_cogs import TmpLimitsAndSizingCog

TEST_STRATEGY = "tmp_limits_store_test"
TRADE_ID = "&L:AAPL20261218C380"


@pytest.fixture(autouse=True)
def reset_db_persistence_flags() -> None:
    """Reset class-level DB persistence flags after each test."""
    yield
    reset_storing_to_db()


@pytest.fixture
def temp_db_env(monkeypatch: pytest.MonkeyPatch):
    """Use the ``temp`` database environment for isolated DB writes."""
    monkeypatch.setenv("ENVIRONMENT", "temp")
    from dbase.database.db_utils import set_environment_context

    set_environment_context("temp")

    import EventDriven.riskmanager.position.live_cogs.save_utils as save_utils

    save_utils.LIMITS_DF = None
    save_utils._ACCESS_COUNTER = 0

    yield

    save_utils.LIMITS_DF = None
    save_utils._ACCESS_COUNTER = 0


def _unique_signal_id() -> str:
    """Return a unique signal id for test isolation."""
    return f"{TEST_STRATEGY}::AAPL{uuid.uuid4().hex[:8]}LONG"


def _sample_limits(signal_id: str) -> PositionLimits:
    """Build sample limits for store tests."""
    return PositionLimits(
        delta=0.42,
        gamma=None,
        vega=None,
        theta=None,
        dte=30,
        moneyness=0.05,
        creation_date=datetime(2026, 6, 11),
    )


def _sample_metadata(trade_id: str, signal_id: str) -> _LimitsMetaData:
    """Build sample metadata for store tests."""
    return _LimitsMetaData(
        trade_id=trade_id,
        date=datetime(2026, 6, 11),
        signal_id=signal_id,
        scalar=1.0,
        sizing_lev=1.0,
        delta_lmt=0.42,
        delta_per_contract=0.21,
        option_price=1.25,
        undl_price=180.0,
        new_quantity=2,
    )


def _cleanup_db_rows(signal_id: str, trade_id: str) -> None:
    """Delete test rows from temp portfolio_data tables."""
    from dbase.database.SQLHelpers import get_engine
    from sqlalchemy import text

    engine = get_engine("portfolio_data")
    params = {
        "signal_id": signal_id,
        "trade_id": trade_id,
        "strategy_name": TEST_STRATEGY,
    }
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                DELETE FROM `limits`
                WHERE `signal_id` = :signal_id
                  AND `trade_id` = :trade_id
                  AND `strategy_name` = :strategy_name
                """
            ),
            params,
        )
        conn.execute(
            text(
                """
                DELETE FROM `position_metadata`
                WHERE `signal_id` = :signal_id
                  AND `trade_id` = :trade_id
                  AND `strategy_name` = :strategy_name
                """
            ),
            params,
        )


def test_tmp_in_memory_cog_roundtrips_limits_and_metadata() -> None:
    """In-memory tmp cog should save and reload limits and metadata."""
    cog = TmpLimitsAndSizingCog(config=LimitsEnabledConfig())
    signal_id = _unique_signal_id()
    limits = _sample_limits(signal_id)
    metadata = _sample_metadata(TRADE_ID, signal_id)

    cog._save_position_limits(TRADE_ID, signal_id, limits)
    cog._store_metadata(metadata)

    loaded_limits = cog._get_position_limits(TRADE_ID, signal_id)
    loaded_metadata = metadata_from_store_payload(
        cog._position_store.get_metadata(TRADE_ID, signal_id)
    )

    assert loaded_limits is not None
    assert math.isclose(loaded_limits.delta, limits.delta)
    assert loaded_limits.dte == limits.dte
    assert loaded_metadata is not None
    assert loaded_metadata.trade_id == TRADE_ID
    assert loaded_metadata.signal_id == signal_id
    assert math.isclose(loaded_metadata.delta_lmt, metadata.delta_lmt)


def test_live_cog_writes_limits_to_temp_db(temp_db_env: None) -> None:
    """Live cog should persist limits to portfolio_data_temp via DatabasePositionStore."""
    signal_id = _unique_signal_id()
    config = LimitsEnabledConfig(run_name=TEST_STRATEGY, default_dte=30, default_moneyness=0.05)
    cog = LiveCOGLimitsAndSizingCog(config=config)
    limits = _sample_limits(signal_id)

    try:
        cog._save_position_limits(TRADE_ID, signal_id, limits)
        cog._position_store.limits._cache.clear()

        loaded = cog._get_position_limits(TRADE_ID, signal_id)
        assert loaded is not None
        assert math.isclose(loaded.delta, limits.delta)
        assert loaded.dte == config.default_dte
        assert loaded.moneyness == config.default_moneyness
    finally:
        _cleanup_db_rows(signal_id, TRADE_ID)


def test_live_cog_writes_metadata_to_temp_db(temp_db_env: None) -> None:
    """Live cog should persist metadata to portfolio_data_temp."""
    signal_id = _unique_signal_id()
    config = LimitsEnabledConfig(run_name=TEST_STRATEGY)
    cog = LiveCOGLimitsAndSizingCog(config=config)
    metadata = _sample_metadata(TRADE_ID, signal_id)

    try:
        cog._store_metadata(metadata)
        cog._position_store.metadata._cache.clear()

        loaded = cog.get_stored_metadata(TRADE_ID, signal_id)
        assert loaded is not None
        assert loaded.trade_id == TRADE_ID
        assert loaded.signal_id == signal_id
        assert math.isclose(float(loaded.delta_lmt), metadata.delta_lmt)
    finally:
        _cleanup_db_rows(signal_id, TRADE_ID)


def test_live_cog_updates_existing_metadata_row(temp_db_env: None) -> None:
    """Live cog should update an existing metadata row for the same key."""
    signal_id = _unique_signal_id()
    config = LimitsEnabledConfig(run_name=TEST_STRATEGY)
    cog = LiveCOGLimitsAndSizingCog(config=config)
    metadata = _sample_metadata(TRADE_ID, signal_id)
    updated = _LimitsMetaData(
        trade_id=TRADE_ID,
        date=metadata.date,
        signal_id=signal_id,
        scalar=1.0,
        sizing_lev=1.0,
        delta_lmt=0.55,
        delta_per_contract=0.21,
        option_price=1.25,
        undl_price=180.0,
        new_quantity=3,
    )

    try:
        cog._store_metadata(metadata)
        cog._store_metadata(updated)
        cog._position_store.metadata._cache.clear()

        loaded = cog.get_stored_metadata(TRADE_ID, signal_id)
        assert loaded is not None
        assert math.isclose(float(loaded.delta_lmt), updated.delta_lmt)
        assert loaded.new_quantity == 3
    finally:
        _cleanup_db_rows(signal_id, TRADE_ID)


def test_live_cog_in_process_metadata_survives_failsafe_lookup(temp_db_env: None) -> None:
    """Same-session ``_get_metadata(trade_id)`` should see the in-process cache."""
    signal_id = _unique_signal_id()
    config = LimitsEnabledConfig(run_name=TEST_STRATEGY)
    cog = LiveCOGLimitsAndSizingCog(config=config)
    metadata = _sample_metadata(TRADE_ID, signal_id)

    try:
        cog._store_metadata(metadata)
        loaded = cog._get_metadata(TRADE_ID)
        assert loaded is not None
        assert loaded.signal_id == signal_id
        assert math.isclose(loaded.delta_lmt, metadata.delta_lmt)
    finally:
        _cleanup_db_rows(signal_id, TRADE_ID)


def test_disable_storing_to_db_skips_db_writes(temp_db_env: None) -> None:
    """When disabled, database stores should write to cache only."""
    signal_id = _unique_signal_id()
    config = LimitsEnabledConfig(run_name=TEST_STRATEGY, default_dte=30, default_moneyness=0.05)
    disable_storing_to_db()
    assert DatabaseLimitsStore.PERSIST_TO_DB is False
    assert DatabaseMetadataStore.PERSIST_TO_DB is False

    cog = LiveCOGLimitsAndSizingCog(config=config)
    limits = _sample_limits(signal_id)
    metadata = _sample_metadata(TRADE_ID, signal_id)

    cog._save_position_limits(TRADE_ID, signal_id, limits)
    cog._store_metadata(metadata)
    cog._position_store.limits._cache.clear()
    cog._position_store.metadata._cache.clear()

    assert cog._position_store.get_limits(TRADE_ID, signal_id) is None
    assert cog._position_store.get_metadata(TRADE_ID, signal_id) is None

    ## Live cog falls back to defaults-only limits when the store has no row
    fallback = cog._get_position_limits(TRADE_ID, signal_id)
    assert fallback is not None
    assert fallback.delta is None
    assert fallback.dte == config.default_dte
    assert fallback.moneyness == config.default_moneyness


def test_build_position_store_live_flag() -> None:
    """``build_position_store`` should select in-memory vs database from ``live``."""
    from EventDriven.riskmanager.position.stores.limits_store import (
        DatabasePositionStore,
        InMemoryPositionStore,
        build_position_store,
    )

    assert isinstance(build_position_store(live=False), InMemoryPositionStore)
    db_store = build_position_store(live=True, strategy_name=TEST_STRATEGY)
    assert isinstance(db_store, DatabasePositionStore)


def test_plain_sizing_cog_uses_store_when_live_false() -> None:
    """PlainSizingCog should persist through InMemoryPositionStore when not live."""
    from EventDriven.configs.core import PlainSizingCogConfig
    from EventDriven.riskmanager.position.cogs.plain_sizing import PlainSizingCog
    from EventDriven.riskmanager.position.stores.limits_store import InMemoryPositionStore

    config = PlainSizingCogConfig(run_name=TEST_STRATEGY, dte_threshold=15)
    cog = PlainSizingCog(config=config, live=False)
    signal_id = _unique_signal_id()
    limits = _sample_limits(signal_id)
    metadata = _sample_metadata(TRADE_ID, signal_id)

    assert isinstance(cog._position_store, InMemoryPositionStore)
    cog._save_position_limits(TRADE_ID, signal_id, limits)
    cog._store_metadata(metadata)

    loaded = cog._get_position_limits(TRADE_ID, signal_id)
    assert loaded is not None
    assert math.isclose(loaded.delta, limits.delta)
    assert cog.position_metadata[TRADE_ID].signal_id == signal_id


def test_donchian_cog_uses_database_store_when_live(temp_db_env: None) -> None:
    """DonchianMomentumCog with live=True should write limits to temp DB."""
    from EventDriven.configs.core import DonchianMomentumCogConfig
    from EventDriven.riskmanager.position.cogs.donchian_cog import DonchianMomentumCog
    from EventDriven.riskmanager.position.stores.limits_store import DatabasePositionStore

    class _StubStrategy:
        """Minimal strategy stub; store tests do not call sizing."""

        pass

    signal_id = _unique_signal_id()
    config = DonchianMomentumCogConfig(run_name=TEST_STRATEGY, dte_threshold=15)
    cog = DonchianMomentumCog(eq_strategy=_StubStrategy(), config=config, live=True)
    limits = _sample_limits(signal_id)

    assert isinstance(cog._position_store, DatabasePositionStore)
    try:
        cog._save_position_limits(TRADE_ID, signal_id, limits)
        cog.position_limits.clear()
        cog._position_store.limits._cache.clear()
        loaded = cog._get_position_limits(TRADE_ID, signal_id)
        assert loaded is not None
        assert math.isclose(loaded.delta, limits.delta)
        assert loaded.dte == config.dte_threshold
    finally:
        _cleanup_db_rows(signal_id, TRADE_ID)
