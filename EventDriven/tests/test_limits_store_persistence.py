"""Comprehensive persistence tests for in-memory and database position stores.

Covers save/get roundtrips for limits and metadata on both backends, DB
verification under truncation, NaN-safe metadata JSON, overwrite behavior,
and live soft-fail vs raise-on-verify. Database cases use the ``temp``
environment (``portfolio_data_temp``).

Usage:
    >>> pytest EventDriven/tests/test_limits_store_persistence.py -q
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import datetime
from typing import Iterator, Optional

import pytest

from EventDriven.dataclasses.limits import PositionLimits
from EventDriven.riskmanager.position.cogs.limits import (
    _LimitsMetaData,
    metadata_from_store_payload,
)
from EventDriven.riskmanager.position.stores.limits_store import (
    DatabasePositionStore,
    InMemoryPositionStore,
    StoreVerificationError,
    _dumps_metadata_json,
    _limit_values_equal,
    build_position_store,
    metadata_to_dict,
)

TEST_STRATEGY = "limits_store_persistence_test"
TRADE_ID = "&L:AAPL20261218C380"
TRADE_DATE = datetime(2026, 6, 11)


@pytest.fixture(autouse=True)
def _reset_save_utils_cache() -> Iterator[None]:
    """Clear save_utils batch cache around each test."""
    import EventDriven.riskmanager.position.live_cogs.save_utils as save_utils

    save_utils.LIMITS_DF = None
    save_utils._ACCESS_COUNTER = 0
    yield
    save_utils.LIMITS_DF = None
    save_utils._ACCESS_COUNTER = 0


@pytest.fixture
def temp_db_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin database resolution to the ``temp`` environment."""
    monkeypatch.setenv("ENVIRONMENT", "temp")
    from dbase.database.db_utils import set_environment_context

    set_environment_context("temp")
    yield


def _unique_signal_id(suffix: str = "") -> str:
    """Return an isolated signal id for parallel-safe DB keys.

    Args:
        suffix: Optional label appended before the random token.

    Returns:
        Unique ``TEST_STRATEGY``-scoped signal identifier.
    """
    token = uuid.uuid4().hex[:10]
    tag = f"{suffix}_" if suffix else ""
    return f"{TEST_STRATEGY}::{tag}{token}LONG"


def _make_limits(
    *,
    delta: Optional[float] = 0.42,
    creation_date: datetime = TRADE_DATE,
) -> PositionLimits:
    """Build sample position limits.

    Args:
        delta: Delta limit value.
        creation_date: Limits creation timestamp.

    Returns:
        Populated ``PositionLimits`` instance.
    """
    return PositionLimits(
        delta=delta,
        gamma=None,
        vega=None,
        theta=None,
        dte=30,
        moneyness=0.05,
        creation_date=creation_date,
    )


def _make_metadata(
    trade_id: str,
    signal_id: str,
    *,
    delta_lmt: float = 0.42,
    delta_per_contract: Optional[float] = 0.21,
    new_quantity: int = 2,
    rvol: Optional[float] = 0.55,
) -> _LimitsMetaData:
    """Build sample sizing metadata.

    Args:
        trade_id: Trade identifier.
        signal_id: Signal identifier.
        delta_lmt: Position delta limit.
        delta_per_contract: Per-contract delta (may be NaN in NaN cases).
        new_quantity: Sized quantity.
        rvol: Relative volume field.

    Returns:
        Populated ``_LimitsMetaData`` instance.
    """
    return _LimitsMetaData(
        trade_id=trade_id,
        date=TRADE_DATE,
        signal_id=signal_id,
        scalar=1.0,
        sizing_lev=1.0,
        delta_lmt=delta_lmt,
        delta_per_contract=delta_per_contract,
        option_price=1.25,
        undl_price=180.0,
        new_quantity=new_quantity,
        rvol=rvol,
    )


def _cleanup_db_rows(signal_id: str, trade_id: str) -> None:
    """Delete ``limits`` and ``position_metadata`` rows for a test key.

    Args:
        signal_id: Signal identifier used in writes.
        trade_id: Trade identifier used in writes.
    """
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


def _assert_limits_roundtrip(
    stored: Optional[PositionLimits],
    expected: PositionLimits,
    *,
    compare_dte: bool = True,
) -> None:
    """Assert loaded limits match the values that were saved.

    Args:
        stored: Limits returned from the store.
        expected: Limits originally saved.
        compare_dte: When ``True``, also assert DTE equality.

    Raises:
        AssertionError: If stored limits are missing or mismatched.
    """
    assert stored is not None
    assert expected.delta is not None
    assert stored.delta is not None
    assert _limit_values_equal(expected.delta, stored.delta)
    assert stored.gamma is None
    assert stored.vega is None
    assert stored.theta is None
    if compare_dte:
        assert stored.dte == expected.dte or stored.dte is None


def _assert_metadata_roundtrip(
    payload: Optional[object],
    expected: _LimitsMetaData,
) -> None:
    """Assert loaded metadata matches the values that were saved.

    Args:
        payload: Raw store payload.
        expected: Metadata originally saved.

    Raises:
        AssertionError: If stored metadata is missing or mismatched.
    """
    loaded = metadata_from_store_payload(payload)
    assert loaded is not None
    assert loaded.trade_id == expected.trade_id
    assert loaded.signal_id == expected.signal_id
    assert math.isclose(float(loaded.delta_lmt), float(expected.delta_lmt))
    assert loaded.new_quantity == expected.new_quantity
    if expected.delta_per_contract is None or (
        isinstance(expected.delta_per_contract, float)
        and math.isnan(expected.delta_per_contract)
    ):
        assert loaded.delta_per_contract is None
    else:
        assert loaded.delta_per_contract is not None
        assert math.isclose(
            float(loaded.delta_per_contract),
            float(expected.delta_per_contract),
        )


class TestInMemoryPositionStore:
    """In-memory store behavior for limits and metadata."""

    def test_limits_and_metadata_roundtrip(self) -> None:
        """Save and reload both payloads through ``InMemoryPositionStore``."""
        store = InMemoryPositionStore()
        signal_id = _unique_signal_id("mem")
        limits = _make_limits()
        metadata = _make_metadata(TRADE_ID, signal_id)

        store.save_limits(TRADE_ID, signal_id, limits)
        store.save_metadata(TRADE_ID, metadata)

        _assert_limits_roundtrip(store.get_limits(TRADE_ID, signal_id), limits)
        _assert_metadata_roundtrip(store.get_metadata(TRADE_ID, signal_id), metadata)

    def test_missing_keys_return_none(self) -> None:
        """Unread keys should return ``None`` for both stores."""
        store = InMemoryPositionStore()
        signal_id = _unique_signal_id("miss")
        assert store.get_limits(TRADE_ID, signal_id) is None
        assert store.get_metadata(TRADE_ID, signal_id) is None

    def test_overwrite_limits_and_metadata(self) -> None:
        """Second save should replace prior in-memory values."""
        store = InMemoryPositionStore()
        signal_id = _unique_signal_id("ow")
        store.save_limits(TRADE_ID, signal_id, _make_limits(delta=0.10))
        store.save_limits(TRADE_ID, signal_id, _make_limits(delta=0.55))
        loaded = store.get_limits(TRADE_ID, signal_id)
        assert loaded is not None
        assert math.isclose(loaded.delta, 0.55)

        first = _make_metadata(TRADE_ID, signal_id, delta_lmt=0.10, new_quantity=1)
        second = _make_metadata(TRADE_ID, signal_id, delta_lmt=0.55, new_quantity=3)
        store.save_metadata(TRADE_ID, first)
        store.save_metadata(TRADE_ID, second)
        _assert_metadata_roundtrip(store.get_metadata(TRADE_ID, signal_id), second)

    def test_nan_metadata_serializes_safely_in_memory(self) -> None:
        """NaN fields survive save via null coercion when converted to dict."""
        store = InMemoryPositionStore()
        signal_id = _unique_signal_id("nan")
        metadata = _make_metadata(
            TRADE_ID,
            signal_id,
            delta_per_contract=float("nan"),
        )
        store.save_metadata(TRADE_ID, metadata)
        payload = store.get_metadata(TRADE_ID, signal_id)
        assert payload is not None
        as_dict = metadata_to_dict(payload) if not isinstance(payload, dict) else payload
        assert as_dict["delta_per_contract"] is None
        encoded = _dumps_metadata_json(as_dict)
        assert "NaN" not in encoded


class TestDatabasePositionStoreTemp:
    """Database store behavior against the ``temp`` portfolio_data environment."""

    def test_limits_and_metadata_db_roundtrip(self, temp_db_env: None) -> None:
        """Persist both payloads to temp DB and reload after clearing caches."""
        signal_id = _unique_signal_id("db")
        store = DatabasePositionStore(
            strategy_name=TEST_STRATEGY,
            default_dte=30,
            default_moneyness=0.05,
            verify_after_save=True,
            raise_on_verification_failure=True,
        )
        limits = _make_limits(delta=0.011384508341273985)
        metadata = _make_metadata(TRADE_ID, signal_id, delta_lmt=0.011384508341273985)

        try:
            store.save_limits(TRADE_ID, signal_id, limits)
            store.save_metadata(TRADE_ID, metadata)

            ## Force DB read path
            store.limits._cache.clear()
            store.metadata._cache.clear()

            loaded_limits = store.get_limits(TRADE_ID, signal_id)
            _assert_limits_roundtrip(loaded_limits, limits, compare_dte=False)
            assert loaded_limits is not None
            assert loaded_limits.dte == 30
            assert loaded_limits.moneyness == 0.05

            loaded_meta = store.get_metadata(TRADE_ID, signal_id)
            _assert_metadata_roundtrip(loaded_meta, metadata)
            assert loaded_meta is not None
            assert "NaN" not in json.dumps(loaded_meta, default=str)
        finally:
            _cleanup_db_rows(signal_id, TRADE_ID)

    def test_high_precision_delta_passes_verification(self, temp_db_env: None) -> None:
        """Truncation-tolerant compare should accept DB-rounded delta values."""
        signal_id = _unique_signal_id("prec")
        store = DatabasePositionStore(
            strategy_name=TEST_STRATEGY,
            verify_after_save=True,
            raise_on_verification_failure=True,
        )
        limits = _make_limits(delta=0.011384508341273985)
        try:
            ## Must not raise StoreVerificationError on precision truncation
            store.save_limits(TRADE_ID, signal_id, limits)
            store.limits._cache.clear()
            loaded = store.get_limits(TRADE_ID, signal_id)
            assert loaded is not None
            assert _limit_values_equal(limits.delta, loaded.delta)
        finally:
            _cleanup_db_rows(signal_id, TRADE_ID)

    def test_nan_metadata_persists_as_null_json(self, temp_db_env: None) -> None:
        """NaN metadata fields should save as JSON null without raising."""
        signal_id = _unique_signal_id("dbnan")
        store = DatabasePositionStore(
            strategy_name=TEST_STRATEGY,
            verify_after_save=True,
            raise_on_verification_failure=True,
        )
        metadata = _make_metadata(
            TRADE_ID,
            signal_id,
            delta_per_contract=float("nan"),
            rvol=float("inf"),
        )
        try:
            store.save_metadata(TRADE_ID, metadata)
            store.metadata._cache.clear()
            loaded = store.get_metadata(TRADE_ID, signal_id)
            assert loaded is not None
            assert loaded.get("delta_per_contract") is None
            assert loaded.get("rvol") is None
            raw = _fetch_raw_metadata_json(signal_id, TRADE_ID)
            assert raw is not None
            assert "NaN" not in raw
            parsed = json.loads(raw)
            assert parsed["delta_per_contract"] is None
            assert parsed["rvol"] is None
        finally:
            _cleanup_db_rows(signal_id, TRADE_ID)

    def test_overwrite_limits_and_metadata(self, temp_db_env: None) -> None:
        """Second DB save should update existing limits and metadata rows."""
        signal_id = _unique_signal_id("dbow")
        store = DatabasePositionStore(
            strategy_name=TEST_STRATEGY,
            verify_after_save=True,
            raise_on_verification_failure=True,
        )
        try:
            store.save_limits(TRADE_ID, signal_id, _make_limits(delta=0.10))
            store.save_limits(TRADE_ID, signal_id, _make_limits(delta=0.77))
            store.limits._cache.clear()
            loaded_limits = store.get_limits(TRADE_ID, signal_id)
            assert loaded_limits is not None
            assert _limit_values_equal(0.77, loaded_limits.delta)

            first = _make_metadata(TRADE_ID, signal_id, delta_lmt=0.10, new_quantity=1)
            second = _make_metadata(TRADE_ID, signal_id, delta_lmt=0.77, new_quantity=5)
            store.save_metadata(TRADE_ID, first)
            store.save_metadata(TRADE_ID, second)
            store.metadata._cache.clear()
            _assert_metadata_roundtrip(store.get_metadata(TRADE_ID, signal_id), second)
        finally:
            _cleanup_db_rows(signal_id, TRADE_ID)

    def test_live_metadata_save_exception_is_logged_not_raised(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Live metadata DB write failures should log payload and continue."""
        store = build_position_store(live=True, strategy_name=TEST_STRATEGY)
        logged: list[str] = []

        def _boom(**kwargs: object) -> None:
            raise RuntimeError("simulated metadata insert failure")

        monkeypatch.setattr(
            store.metadata,
            "_persist_metadata_row",
            _boom,
        )
        monkeypatch.setattr(
            "EventDriven.riskmanager.position.stores.limits_store.verification_logger.error",
            lambda msg, *args, **kwargs: logged.append(
                (msg % args) if args else str(msg)
            ),
        )

        signal_id = _unique_signal_id("metasave")
        metadata = _make_metadata(TRADE_ID, signal_id)
        store.save_metadata(TRADE_ID, metadata)

        assert logged
        assert "metadata store save failed" in logged[0]
        assert "simulated metadata insert failure" in logged[0]
        ## In-memory cache retained so failsafe / same-session lookup still works
        cached = store.get_metadata(TRADE_ID, signal_id)
        assert cached is not None
        assert cached["signal_id"] == signal_id

    def test_raising_metadata_save_exception_still_propagates(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Non-live raise mode should still surface metadata DB write errors."""
        from EventDriven.riskmanager.position.stores.limits_store import (
            DatabaseMetadataStore,
        )

        store = DatabaseMetadataStore(
            strategy_name=TEST_STRATEGY,
            raise_on_verification_failure=True,
        )

        def _boom(**kwargs: object) -> None:
            raise RuntimeError("simulated metadata insert failure")

        monkeypatch.setattr(store, "_persist_metadata_row", _boom)
        signal_id = _unique_signal_id("metareraise")
        with pytest.raises(RuntimeError, match="simulated metadata insert failure"):
            store.save(TRADE_ID, _make_metadata(TRADE_ID, signal_id))

    def test_live_builder_does_not_raise_on_verification_failure(
        self,
        temp_db_env: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``build_position_store(live=True)`` logs verify failures instead of raising."""
        store = build_position_store(live=True, strategy_name=TEST_STRATEGY)
        assert isinstance(store, DatabasePositionStore)
        assert store.limits.raise_on_verification_failure is False
        assert store.metadata.raise_on_verification_failure is False

        logged: list[str] = []
        monkeypatch.setattr(
            "EventDriven.riskmanager.position.stores.limits_store._collect_limits_verification_errors",
            lambda **kwargs: ["forced mismatch"],
        )
        monkeypatch.setattr(
            "EventDriven.riskmanager.position.stores.limits_store.verification_logger.error",
            lambda msg, *args, **kwargs: logged.append(msg % args if args else msg),
        )

        signal_id = _unique_signal_id("livevf")
        limits = _make_limits(delta=0.33)
        try:
            store.save_limits(TRADE_ID, signal_id, limits)
            assert logged
            assert "limits store verification failed" in logged[0]
            ## Cache kept so live session retains computed limits
            assert store.get_limits(TRADE_ID, signal_id) is limits
        finally:
            _cleanup_db_rows(signal_id, TRADE_ID)

    def test_raising_store_surfaces_verification_error(
        self,
        temp_db_env: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Explicit raise mode should propagate ``StoreVerificationError``."""
        store = DatabasePositionStore(
            strategy_name=TEST_STRATEGY,
            verify_after_save=True,
            raise_on_verification_failure=True,
        )
        monkeypatch.setattr(
            "EventDriven.riskmanager.position.stores.limits_store._collect_metadata_verification_errors",
            lambda **kwargs: ["no metadata row found after save"],
        )
        signal_id = _unique_signal_id("raise")
        metadata = _make_metadata(TRADE_ID, signal_id)
        try:
            with pytest.raises(StoreVerificationError, match="metadata store verification failed"):
                store.save_metadata(TRADE_ID, metadata)
        finally:
            _cleanup_db_rows(signal_id, TRADE_ID)


def _fetch_raw_metadata_json(signal_id: str, trade_id: str) -> Optional[str]:
    """Read the raw ``metadata_dict`` JSON text from temp DB.

    Args:
        signal_id: Signal identifier.
        trade_id: Trade identifier.

    Returns:
        Raw JSON string, or ``None`` when no row exists.
    """
    import pandas as pd
    from dbase.database.SQLHelpers import get_engine
    from sqlalchemy import text

    engine = get_engine("portfolio_data")
    df = pd.read_sql(
        text(
            """
            SELECT CAST(`metadata_dict` AS CHAR) AS metadata_json
            FROM `position_metadata`
            WHERE `signal_id` = :signal_id
              AND `trade_id` = :trade_id
              AND `strategy_name` = :strategy_name
            LIMIT 1
            """
        ),
        engine,
        params={
            "signal_id": signal_id,
            "trade_id": trade_id,
            "strategy_name": TEST_STRATEGY,
        },
    )
    if df.empty:
        return None
    value = df.iloc[0]["metadata_json"]
    return None if value is None else str(value)


def test_build_position_store_backends() -> None:
    """Factory should return the backend matching the ``live`` flag."""
    mem = build_position_store(live=False)
    db = build_position_store(live=True, strategy_name=TEST_STRATEGY)
    assert isinstance(mem, InMemoryPositionStore)
    assert isinstance(db, DatabasePositionStore)


def test_in_memory_and_db_parity_shape(temp_db_env: None) -> None:
    """Both backends should expose the same save/get surface for one trade.

    This is the end-to-end smoke: identical payloads written to memory and temp
    DB should round-trip comparable limits and metadata fields.
    """
    signal_id = _unique_signal_id("parity")
    limits = _make_limits(delta=0.038147352071577594)
    metadata = _make_metadata(
        TRADE_ID,
        signal_id,
        delta_lmt=0.038147352071577594,
        delta_per_contract=0.033171610497023996,
        new_quantity=1,
        rvol=0.5585154860100335,
    )

    mem = InMemoryPositionStore()
    mem.save_limits(TRADE_ID, signal_id, limits)
    mem.save_metadata(TRADE_ID, metadata)

    db = DatabasePositionStore(
        strategy_name=TEST_STRATEGY,
        default_dte=30,
        default_moneyness=0.05,
        verify_after_save=True,
        raise_on_verification_failure=True,
    )
    try:
        db.save_limits(TRADE_ID, signal_id, limits)
        db.save_metadata(TRADE_ID, metadata)
        db.limits._cache.clear()
        db.metadata._cache.clear()

        mem_limits = mem.get_limits(TRADE_ID, signal_id)
        db_limits = db.get_limits(TRADE_ID, signal_id)
        assert mem_limits is not None and db_limits is not None
        assert _limit_values_equal(mem_limits.delta, db_limits.delta)

        mem_meta = metadata_from_store_payload(mem.get_metadata(TRADE_ID, signal_id))
        db_meta = metadata_from_store_payload(db.get_metadata(TRADE_ID, signal_id))
        assert mem_meta is not None and db_meta is not None
        assert mem_meta.signal_id == db_meta.signal_id
        assert math.isclose(float(mem_meta.delta_lmt), float(db_meta.delta_lmt))
        assert mem_meta.new_quantity == db_meta.new_quantity
        assert mem_meta.delta_per_contract is not None
        assert db_meta.delta_per_contract is not None
        assert math.isclose(
            float(mem_meta.delta_per_contract),
            float(db_meta.delta_per_contract),
        )
    finally:
        _cleanup_db_rows(signal_id, TRADE_ID)
