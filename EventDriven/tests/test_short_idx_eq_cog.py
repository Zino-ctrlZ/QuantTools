"""Tests for ShortIdxEqCog sizing, metadata, and optional profit rolls."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from EventDriven.configs.core import ShortIdxEqCogConfig
from EventDriven.dataclasses.orders import OrderRequest
from EventDriven.dataclasses.timeseries import AtTimePositionData
from EventDriven.riskmanager.actions import CLOSE, ROLL
from EventDriven.riskmanager.position.cogs.short_idx_eq import (
    ShortIdxEqCog,
    metadata_from_store_payload,
)
from EventDriven.dataclasses.states import (
    NewPositionState,
    PortfolioMetaInfo,
    PortfolioState,
    PositionAnalysisContext,
    PositionState,
)
from EventDriven.types import Order, OrderData
from EventDriven.trade import Trade
from trade.datamanager.market_data import AtIndexResult


REQUIRED_SETUP_FEATURES = (
    "p_down_given_down_10d_126d",
    "vix_spy_rvol_20_ratio",
    "vix_vs_vix3m",
    "vix_level",
    "vix_pct_200ma",
    "xlu_spy_zscore",
    "xly_xlp_relative_momentum_20d",
    "gld_spy_relative_momentum_20d",
    "fxi_iv_to_asset_rvol_20d_ratio",
    "eem_iv_to_asset_rvol_20d_ratio",
    "uup_close_vs_sma_200",
    "usdcnh_ret_20d",
)

SIGNAL_DATE = "2018-06-15"
ORDER_DATE = date(2018, 6, 20)
TICKER = "EEM"
TRADE_ID = "&L:EEM20180921P40"
SIGNAL_ID = f"short_donchian_equity::{TICKER}20180615SHORT"
OTHER_SIGNAL_ID = f"plain_sizing::{TICKER}20180615SHORT"


class FakeIndicator:
    """Minimal indicator container with a values array."""

    def __init__(self, values: List[float]) -> None:
        """Store indicator values.

        Args:
            values: Per-date indicator values aligned to the fake strategy index.
        """
        self.values = np.asarray(values, dtype=float)


class FakeAssetStrategy:
    """Stand-in for ShortDonchianEquityFlat used by ShortIdxEqCog tests."""

    REQUIRED_SETUP_FEATURES = REQUIRED_SETUP_FEATURES
    MULTIPLIER_VERSION = 1

    def __init__(
        self,
        *,
        dates: Optional[List[str]] = None,
        feature_rows: Optional[List[Dict[str, float]]] = None,
        multiplier: float = 3,
    ) -> None:
        """Initialize a fake per-ticker strategy.

        Args:
            dates: Calendar dates present on the fake bar index.
            feature_rows: Per-date feature maps aligned with ``dates``.
            multiplier: Value returned by ``assign_dollar_multiplier``.
        """
        dates = dates or [SIGNAL_DATE, "2018-06-18", "2018-06-20"]
        self._index = [pd.Timestamp(d) for d in dates]
        self._dates_map = {ts: i for i, ts in enumerate(self._index)}
        self._n = len(self._index)
        self.multiplier = multiplier
        self.last_multiplier_date: Optional[pd.Timestamp] = None
        self.feature_rows = feature_rows or [
            {
                "uup_close_vs_sma_200": 0.03,
                "usdcnh_ret_20d": 0.02,
                "vix_level": 18.5,
                "p_down_given_down_10d_126d": 0.4,
            },
            {},
            {},
        ]
        self.indicators = {
            name: FakeIndicator([row.get(name, np.nan) for row in self.feature_rows])
            for name in REQUIRED_SETUP_FEATURES
        }

    def _resolve(self, *, date: pd.Timestamp = None, index: int = None):
        """Resolve date or index the same way StrategyBase does.

        Args:
            date: Optional lookup date.
            index: Optional integer index.

        Returns:
            Tuple of ``(index, timestamp)``.
        """
        if index is not None:
            return index, self._index[index]
        ts = pd.Timestamp(date)
        idx = self._dates_map.get(ts)
        if idx is None:
            raise KeyError(f"date {ts} not found in dataset index.")
        return idx, ts

    def assign_dollar_multiplier(self, date: pd.Timestamp = None, index: int = None) -> float:
        """Return the configured multiplier and record the lookup date.

        Args:
            date: Lookup date forwarded by the cog.
            index: Optional integer index.

        Returns:
            Configured multiplier, or 1.5 when version is 2.
        """
        _, ts = self._resolve(date=date, index=index)
        self.last_multiplier_date = ts
        if self.MULTIPLIER_VERSION == 2:
            return 1.5
        if self.MULTIPLIER_VERSION == 3:
            return 3
        return self.multiplier


class FakeMultiAssetStrategy:
    """Minimal MultiAssetStrategy stand-in exposing ``asset_strategies``."""

    def __init__(self, strategies: Dict[str, FakeAssetStrategy]) -> None:
        """Store per-ticker fake strategies.

        Args:
            strategies: Mapping of ticker to fake asset strategy.
        """
        self.asset_strategies = strategies


def _option_data(price: float = 2.5) -> AtTimePositionData:
    """Build option at-time data with a midpoint equal to ``price``.

    Args:
        price: Option midpoint / close used for sizing.

    Returns:
        At-time position data payload.
    """
    return AtTimePositionData(
        date=datetime(2018, 6, 20),
        close=price,
        bid=price - 0.05,
        ask=price + 0.05,
        midpoint=price,
        delta=-0.3,
        gamma=0.01,
        vega=0.1,
        theta=-0.02,
        position_id=TRADE_ID,
        use_price="midpoint",
    )


def _undl_data() -> AtIndexResult:
    """Build a dummy underlying snapshot.

    Returns:
        At-index underlying result used by PositionState.
    """
    close = pd.Series({"close": 40.0})
    return AtIndexResult(
        sym=TICKER,
        date=pd.Timestamp("2018-06-20"),
        spot=close,
        chain_spot=close,
        rates=pd.Series({"rate": 0.02}),
        dividends=0.0,
        dividend_yield=0.0,
        split_factor=1.0,
    )


def _new_position_state(
    *,
    signal_id: str = SIGNAL_ID,
    ticker: str = TICKER,
    option_price: float = 2.5,
    quantity: int = 0,
    tick_cash: float = 3000.0,
    is_tick_cash_scaled: bool = True,
) -> NewPositionState:
    """Build a NewPositionState for sizing tests.

    Args:
        signal_id: Signal identifier, optionally slug-prefixed.
        ticker: Underlying ticker stored on the state.
        option_price: Option midpoint used for sizing.
        quantity: Initial order quantity.
        tick_cash: Request tick cash. Defaults to dollars when scaled.
        is_tick_cash_scaled: Whether ``tick_cash`` is already in dollars.

    Returns:
        New position state ready for ``on_new_position``.
    """
    order = Order(
        result="SUCCESSFUL",
        signal_id=signal_id,
        map_signal_id=signal_id,
        date=ORDER_DATE,
        data=OrderData(
            trade_id=TRADE_ID,
            long=[TRADE_ID.replace("&L:", "")],
            short=[],
            close=option_price,
            quantity=quantity,
        ),
    )
    request = OrderRequest(
        date=datetime(2018, 6, 20),
        symbol=ticker,
        option_type="p",
        max_close=5,
        tick_cash=tick_cash,
        direction="SHORT",
        signal_id=signal_id,
        is_tick_cash_scaled=is_tick_cash_scaled,
    )
    return NewPositionState(
        trade_id=TRADE_ID,
        order=order,
        request=request,
        symbol=ticker,
        at_time_data=_option_data(option_price),
        undl_at_time_data=_undl_data(),
    )


def _make_cog(
    *,
    asset: Optional[FakeAssetStrategy] = None,
    tickers: Optional[Dict[str, FakeAssetStrategy]] = None,
    config: Optional[ShortIdxEqCogConfig] = None,
    calculator=None,
    trade_size: float = 3000.0,
    **config_kwargs: Any,
) -> tuple[ShortIdxEqCog, FakeAssetStrategy]:
    """Construct a cog with a fake multi-asset strategy.

    Args:
        asset: Optional explicit fake asset strategy for ``TICKER``.
        tickers: Optional full ``asset_strategies`` mapping.
        config: Optional config override.
        calculator: Optional quantity calculator override.
        trade_size: Default trade size when ``config`` is omitted.
        **config_kwargs: Extra ``ShortIdxEqCogConfig`` fields.

    Returns:
        Tuple of ``(cog, asset_strategy_for_ticker)``.
    """
    asset = asset or FakeAssetStrategy()
    multi = FakeMultiAssetStrategy(tickers if tickers is not None else {TICKER: asset})
    if config is None:
        config = ShortIdxEqCogConfig(trade_size=trade_size, **config_kwargs)
    cog = ShortIdxEqCog(eq_strategy=multi, config=config, calculator=calculator)
    return cog, asset


def test_config_requires_positive_trade_size() -> None:
    """trade_size must be provided and greater than zero."""
    with pytest.raises(ValueError, match="trade_size"):
        ShortIdxEqCogConfig()
    with pytest.raises(ValueError, match="trade_size"):
        ShortIdxEqCogConfig(trade_size=0)
    with pytest.raises(ValueError, match="version 2"):
        ShortIdxEqCogConfig(trade_size=1000, multiplier_version=2)


def test_config_rejects_both_profit_flags() -> None:
    """enable_profit_roll and enable_profit_waterfall cannot both be True."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        ShortIdxEqCogConfig(trade_size=1000, enable_profit_roll=True, enable_profit_waterfall=True)


def test_cog_requires_config() -> None:
    """Cog construction without config should fail because trade_size is required."""
    with pytest.raises(TypeError, match="trade_size"):
        ShortIdxEqCog(eq_strategy=FakeMultiAssetStrategy({TICKER: FakeAssetStrategy()}))


def test_default_calculator_sizes_from_multiplier_and_close() -> None:
    """Default qty is trade_size * multiplier / 3 / (close * 100)."""
    cog, asset = _make_cog(asset=FakeAssetStrategy(multiplier=3), multiplier_version=3)
    state = _new_position_state(option_price=2.5)
    cog.on_new_position(state)

    assert state.order["data"]["quantity"] == 12
    assert asset.last_multiplier_date == pd.Timestamp(SIGNAL_DATE)
    meta = cog.position_metadata[TRADE_ID]
    assert meta.multiplier == 3
    assert meta.tick_cash == pytest.approx(3000.0)
    assert meta.config_trade_size == pytest.approx(3000.0)
    assert meta.trade_size == pytest.approx(3000.0)
    assert meta.allowed_trade_size == pytest.approx(3000.0)
    assert meta.new_quantity == 12
    assert meta.signal_date == SIGNAL_DATE
    stored = cog._position_store.get_metadata(TRADE_ID, SIGNAL_ID)
    assert stored is not None
    assert stored["new_quantity"] == 12
    assert stored["setup_features_at_date"]["uup_close_vs_sma_200"] == pytest.approx(0.03)
    assert stored["setup_features_at_date"]["usdcnh_ret_20d"] == pytest.approx(0.02)
    assert stored["setup_features_at_date"]["vix_level"] == pytest.approx(18.5)


def test_default_calculator_scales_down_for_multiplier_one() -> None:
    """Multiplier 1 should use one-third of trade_size."""
    cog, _ = _make_cog(asset=FakeAssetStrategy(multiplier=1))
    state = _new_position_state(option_price=2.5)
    cog.on_new_position(state)
    assert state.order["data"]["quantity"] == 4


def test_zero_quantity_defaults_to_one() -> None:
    """When the calculator yields 0, quantity becomes 1."""
    cog, _ = _make_cog(asset=FakeAssetStrategy(multiplier=1), trade_size=100)
    state = _new_position_state(option_price=50.0)
    cog.on_new_position(state)
    assert state.order["data"]["quantity"] == 1


def test_custom_calculator_is_used() -> None:
    """A passed calculator should replace the default formula."""

    seen = {}

    def _calc(multiplier: int, option_price: float, trade_size: float) -> int:
        seen["args"] = (multiplier, option_price, trade_size)
        return multiplier * 2

    cog, _ = _make_cog(asset=FakeAssetStrategy(multiplier=3), calculator=_calc)
    state = _new_position_state()
    cog.on_new_position(state)
    assert state.order["data"]["quantity"] == 6
    assert seen["args"] == (3, 2.5, 3000.0)


def test_effective_trade_size_is_min_of_tick_cash_and_config() -> None:
    """Sizing budget should cap at the smaller of tick cash and config trade_size."""
    cog, _ = _make_cog(asset=FakeAssetStrategy(multiplier=3), trade_size=3000.0)
    state = _new_position_state(option_price=2.5, tick_cash=1500.0, is_tick_cash_scaled=True)
    cog.on_new_position(state)

    ## 1500 * 3 / 3 / (2.5 * 100) = 6
    assert state.order["data"]["quantity"] == 6
    meta = cog.position_metadata[TRADE_ID]
    assert meta.tick_cash == pytest.approx(1500.0)
    assert meta.config_trade_size == pytest.approx(3000.0)
    assert meta.trade_size == pytest.approx(1500.0)
    assert meta.allowed_trade_size == pytest.approx(1500.0)


def test_unscaled_tick_cash_is_converted_to_dollars_before_min() -> None:
    """Unscaled tick cash is * 100 before comparing to config trade_size."""
    cog, _ = _make_cog(asset=FakeAssetStrategy(multiplier=3), trade_size=3000.0)
    state = _new_position_state(option_price=2.5, tick_cash=15.0, is_tick_cash_scaled=False)
    cog.on_new_position(state)

    ## scaled tick cash = 1500; min(1500, 3000) = 1500; qty = 6
    assert state.order["data"]["quantity"] == 6
    assert cog.position_metadata[TRADE_ID].trade_size == pytest.approx(1500.0)


def test_skips_non_matching_strategy_slug() -> None:
    """Signals without the short Donchian equity slug are ignored."""
    cog, _ = _make_cog()
    state = _new_position_state(signal_id=OTHER_SIGNAL_ID)
    cog.on_new_position(state)
    assert state.order["data"]["quantity"] == 0
    assert cog.position_metadata == {}


def test_missing_ticker_raises_informative_error() -> None:
    """Missing asset_strategies ticker should name the ticker and available keys."""
    cog, _ = _make_cog(tickers={"IWM": FakeAssetStrategy()})
    state = _new_position_state(ticker="EEM")
    with pytest.raises(KeyError, match=r"ticker 'EEM' not found") as exc_info:
        cog.on_new_position(state)
    assert "IWM" in str(exc_info.value)


def test_multiplier_version_is_restored_after_call() -> None:
    """Config multiplier_version should apply for the call only, then undo."""
    asset = FakeAssetStrategy(multiplier=1)
    asset.MULTIPLIER_VERSION = 1
    cog, _ = _make_cog(asset=asset, multiplier_version=3)
    state = _new_position_state()
    cog.on_new_position(state)

    assert state.order["data"]["quantity"] == 12
    assert asset.MULTIPLIER_VERSION == 1
    assert cog.position_metadata[TRADE_ID].multiplier_version == 3


def test_class_level_multiplier_version_is_not_left_on_instance() -> None:
    """Temporary version stamp should not persist as an instance attribute."""

    class _ClassVersionStrategy(FakeAssetStrategy):
        MULTIPLIER_VERSION = 1

    asset = _ClassVersionStrategy(multiplier=1)
    assert "MULTIPLIER_VERSION" not in asset.__dict__
    cog, _ = _make_cog(asset=asset, multiplier_version=3)
    cog.on_new_position(_new_position_state())
    assert "MULTIPLIER_VERSION" not in asset.__dict__
    assert asset.MULTIPLIER_VERSION == 1


def test_version_two_on_strategy_is_rejected() -> None:
    """Effective multiplier version 2 should raise before sizing."""
    asset = FakeAssetStrategy()
    asset.MULTIPLIER_VERSION = 2
    cog, _ = _make_cog(asset=asset)
    with pytest.raises(ValueError, match="version 2"):
        cog.on_new_position(_new_position_state())
    assert asset.MULTIPLIER_VERSION == 2


def test_non_integer_multiplier_is_rejected() -> None:
    """Fractional assign_dollar_multiplier output is not allowed."""
    asset = FakeAssetStrategy(multiplier=1.5)
    cog, _ = _make_cog(asset=asset)
    with pytest.raises(ValueError, match="non-integer"):
        cog.on_new_position(_new_position_state())


def test_uses_signal_id_date_not_order_date() -> None:
    """Rolls keep the original signal id, so lookup date must come from it."""
    cog, asset = _make_cog(asset=FakeAssetStrategy(multiplier=3))
    state = _new_position_state()
    assert state.order["date"] == ORDER_DATE
    cog.on_new_position(state)
    assert asset.last_multiplier_date == pd.Timestamp(SIGNAL_DATE)


def test_analyze_does_nothing_when_profit_roll_disabled() -> None:
    """Default enable_profit_roll=False should return no opinions."""
    cog, _ = _make_cog()
    pos = _open_position(pnl=20.0, quantity=4, entry_price=2.5)
    actions = cog._analyze_impl(_analysis_context([pos]))
    assert actions.opinions == []


def test_analyze_rolls_when_pnl_above_threshold() -> None:
    """PnL% above threshold should emit ROLL when the feature is enabled."""
    cog, _ = _make_cog(enable_profit_roll=True, roll_profit_threshold=1.0)
    pos = _open_position(pnl=12.0, quantity=4, entry_price=2.5)
    actions = cog._analyze_impl(_analysis_context([pos]))
    assert len(actions.opinions) == 1
    assert isinstance(actions.opinions[0].action, ROLL)
    assert actions.opinions[0].action.action["new_quantity"] == 4


def test_analyze_skips_below_threshold_and_other_slugs() -> None:
    """Below-threshold names and non-matching slugs should not emit opinions."""
    cog, _ = _make_cog(enable_profit_roll=True, roll_profit_threshold=1.0)
    below = _open_position(pnl=8.0, quantity=4, entry_price=2.5)
    other = _open_position(pnl=20.0, quantity=4, entry_price=2.5, signal_id=OTHER_SIGNAL_ID)
    actions = cog._analyze_impl(_analysis_context([below, other]))
    assert actions.opinions == []


def test_new_position_seeds_waterfall_metadata_fields() -> None:
    """Sizing should always persist initial_quantity and inactive trigger fields."""
    cog, _ = _make_cog(asset=FakeAssetStrategy(multiplier=3))
    cog.on_new_position(_new_position_state(option_price=2.5))
    meta = cog.position_metadata[TRADE_ID]
    assert meta.initial_quantity == 12
    assert meta.half_closed is False
    assert meta.threshold_triggered is False
    assert meta.threshold_triggered_pct is None
    assert meta.threshold_triggered_date is None
    assert meta.stop_triggered is False
    assert meta.stop_triggered_pct is None
    assert meta.stop_triggered_date is None
    stored = cog._position_store.get_metadata(TRADE_ID, SIGNAL_ID)
    rebuilt = metadata_from_store_payload(stored)
    assert rebuilt is not None
    assert rebuilt.initial_quantity == 12
    assert rebuilt.half_closed is False
    assert rebuilt.threshold_triggered is False
    assert rebuilt.stop_triggered is False


def test_metadata_from_store_payload_defaults_legacy_rows() -> None:
    """Older payloads without waterfall fields should default safely."""
    rebuilt = metadata_from_store_payload(
        {
            "trade_id": TRADE_ID,
            "date": "2018-06-20",
            "signal_id": SIGNAL_ID,
            "ticker": TICKER,
            "multiplier": 3,
            "multiplier_version": 1,
            "option_price": 2.5,
            "tick_cash": 3000.0,
            "config_trade_size": 3000.0,
            "trade_size": 3000.0,
            "allowed_trade_size": 3000.0,
            "new_quantity": 4,
            "signal_date": SIGNAL_DATE,
            "setup_features_at_date": {},
        }
    )
    assert rebuilt is not None
    assert rebuilt.initial_quantity == 4
    assert rebuilt.half_closed is False
    assert rebuilt.threshold_triggered is False
    assert rebuilt.stop_triggered is False


def test_metadata_from_store_payload_maps_legacy_stop_triggered() -> None:
    """Legacy waterfall_stop_triggered should map onto stop_triggered."""
    rebuilt = metadata_from_store_payload(
        {
            "trade_id": TRADE_ID,
            "date": "2018-06-20",
            "signal_id": SIGNAL_ID,
            "ticker": TICKER,
            "multiplier": 3,
            "multiplier_version": 1,
            "option_price": 2.5,
            "tick_cash": 3000.0,
            "config_trade_size": 3000.0,
            "trade_size": 3000.0,
            "allowed_trade_size": 3000.0,
            "new_quantity": 1,
            "signal_date": SIGNAL_DATE,
            "setup_features_at_date": {},
            "initial_quantity": 3,
            "half_closed": True,
            "waterfall_stop_triggered": True,
        }
    )
    assert rebuilt is not None
    assert rebuilt.stop_triggered is True
    assert rebuilt.stop_triggered_pct is None
    assert rebuilt.stop_triggered_date is None


def test_waterfall_qty1_rolls_at_threshold_and_marks_taken() -> None:
    """initial_qty=1 at threshold vs initial should ROLL and set half_closed."""
    cog, _ = _make_cog(enable_profit_waterfall=True, waterfall_profit_threshold=1.0)
    cog.on_new_position(_new_position_state(option_price=50.0, tick_cash=100.0, is_tick_cash_scaled=True))
    assert cog.position_metadata[TRADE_ID].initial_quantity == 1

    ## initial cost = 2.5 * 1 = 2.5; pnl 2.5 -> 100%
    pos = _open_position(pnl=2.5, quantity=1, entry_price=2.5)
    actions = cog._analyze_impl(_analysis_context([pos]))
    assert len(actions.opinions) == 1
    assert isinstance(actions.opinions[0].action, ROLL)
    meta = cog.position_metadata[TRADE_ID]
    assert meta.half_closed is True
    assert meta.threshold_triggered is True
    assert meta.threshold_triggered_pct == pytest.approx(1.0)
    assert meta.threshold_triggered_date == datetime(2018, 7, 1)
    stored = cog._position_store.get_metadata(TRADE_ID, SIGNAL_ID)
    assert stored["half_closed"] is True
    assert stored["threshold_triggered"] is True

    ## Do not re-fire
    assert cog._analyze_impl(_analysis_context([pos])).opinions == []


def test_waterfall_sells_half_once_and_does_not_resell() -> None:
    """qty>1 closes ceil(fraction * initial) once at threshold, then stays quiet."""
    cog, _ = _make_cog(enable_profit_waterfall=True, waterfall_profit_threshold=1.0)

    def _calc(_m: int, _p: float, _t: float) -> int:
        return 3

    cog.calculator = _calc
    cog.on_new_position(_new_position_state(option_price=2.5))
    assert cog.position_metadata[TRADE_ID].initial_quantity == 3

    ## initial cost = 2.5 * 3 = 7.5; default fraction 0.5 -> ceil(3/2)=2
    pos = _open_position(pnl=7.5, quantity=3, entry_price=2.5)
    actions = cog._analyze_impl(_analysis_context([pos]))
    assert len(actions.opinions) == 1
    assert isinstance(actions.opinions[0].action, CLOSE)
    assert actions.opinions[0].action.action["quantity_diff"] == -2
    assert actions.opinions[0].action.action["new_quantity"] == 1
    meta = cog.position_metadata[TRADE_ID]
    assert meta.half_closed is True
    assert meta.new_quantity == 1

    ## Even at higher PnL, do not sell again
    pos_after = _open_position(pnl=100.0, quantity=1, entry_price=2.5)
    assert cog._analyze_impl(_analysis_context([pos_after])).opinions == []
    stored = cog._position_store.get_metadata(TRADE_ID, SIGNAL_ID)
    assert stored["half_closed"] is True


def test_waterfall_close_fraction_one_third() -> None:
    """waterfall_close_fraction=1/3 should CLOSE ceil(initial/3)."""
    cog, _ = _make_cog(
        enable_profit_waterfall=True,
        waterfall_profit_threshold=1.0,
        waterfall_close_fraction=1.0 / 3.0,
    )

    def _calc(_m: int, _p: float, _t: float) -> int:
        return 4

    cog.calculator = _calc
    cog.on_new_position(_new_position_state(option_price=2.5))

    ## ceil(4 * 1/3) = ceil(1.333) = 2
    pos = _open_position(pnl=10.0, quantity=4, entry_price=2.5)
    actions = cog._analyze_impl(_analysis_context([pos]))
    assert len(actions.opinions) == 1
    assert isinstance(actions.opinions[0].action, CLOSE)
    assert actions.opinions[0].action.action["quantity_diff"] == -2
    assert actions.opinions[0].action.action["new_quantity"] == 2


def test_waterfall_close_fraction_one_quarter() -> None:
    """waterfall_close_fraction=0.25 should CLOSE ceil(initial/4)."""
    cog, _ = _make_cog(
        enable_profit_waterfall=True,
        waterfall_profit_threshold=1.0,
        waterfall_close_fraction=0.25,
    )

    def _calc(_m: int, _p: float, _t: float) -> int:
        return 4

    cog.calculator = _calc
    cog.on_new_position(_new_position_state(option_price=2.5))

    ## ceil(4 * 0.25) = 1
    pos = _open_position(pnl=10.0, quantity=4, entry_price=2.5)
    actions = cog._analyze_impl(_analysis_context([pos]))
    assert len(actions.opinions) == 1
    assert isinstance(actions.opinions[0].action, CLOSE)
    assert actions.opinions[0].action.action["quantity_diff"] == -1
    assert actions.opinions[0].action.action["new_quantity"] == 3


def test_config_rejects_invalid_waterfall_close_fraction() -> None:
    """waterfall_close_fraction must be in (0, 1]."""
    with pytest.raises(ValueError, match="waterfall_close_fraction"):
        ShortIdxEqCogConfig(trade_size=1000, waterfall_close_fraction=0.0)
    with pytest.raises(ValueError, match="waterfall_close_fraction"):
        ShortIdxEqCogConfig(trade_size=1000, waterfall_close_fraction=1.5)


def test_config_validates_waterfall_stop_loss_settings() -> None:
    """Waterfall stop loss requires waterfall mode and a positive offset."""
    with pytest.raises(ValueError, match="requires enable_profit_waterfall"):
        ShortIdxEqCogConfig(trade_size=1000, enable_waterfall_stop_loss=True)
    with pytest.raises(ValueError, match="waterfall_stop_loss_offset"):
        ShortIdxEqCogConfig(
            trade_size=1000,
            enable_profit_waterfall=True,
            enable_waterfall_stop_loss=True,
            waterfall_stop_loss_offset=0.0,
        )


def test_waterfall_profit_stop_uses_crossing_pnl_and_closes_remaining() -> None:
    """Crossing at 110% with a 0.5 multiplier should arm and trigger a 55% stop."""
    cog, _ = _make_cog(
        enable_profit_waterfall=True,
        waterfall_profit_threshold=1.0,
        enable_waterfall_stop_loss=True,
        waterfall_stop_loss_offset=0.5,
    )

    def _calc(_m: int, _p: float, _t: float) -> int:
        return 3

    cog.calculator = _calc
    cog.on_new_position(_new_position_state(option_price=2.5))

    ## Initial cost = 7.5; PnL 8.25 is 110%, so the frozen stop is 55%.
    crossed = _open_position(pnl=8.25, quantity=3, entry_price=2.5)
    trim_actions = cog._analyze_impl(_analysis_context([crossed]))
    assert len(trim_actions.opinions) == 1
    assert isinstance(trim_actions.opinions[0].action, CLOSE)
    metadata = cog.position_metadata[TRADE_ID]
    assert metadata.threshold_triggered is True
    assert metadata.threshold_triggered_pct == pytest.approx(1.1)
    assert metadata.threshold_triggered_date == datetime(2018, 7, 1)
    assert metadata.waterfall_stop_reference_pnl_pct == pytest.approx(1.1)
    assert metadata.waterfall_stop_pnl_pct == pytest.approx(0.55)
    assert metadata.waterfall_stop_set_date == datetime(2018, 7, 1)
    assert metadata.stop_triggered is False
    assert metadata.stop_triggered_pct is None
    assert metadata.stop_triggered_date is None

    ## Above the stored stop, the remaining contract stays open.
    above_stop = _open_position(pnl=4.2, quantity=1, entry_price=2.5)
    assert cog._analyze_impl(_analysis_context([above_stop])).opinions == []

    ## At 55% vs frozen initial cost, close all remaining quantity exactly once.
    at_stop = _open_position(pnl=4.125, quantity=1, entry_price=2.5)
    stop_actions = cog._analyze_impl(_analysis_context([at_stop]))
    assert len(stop_actions.opinions) == 1
    stop_action = stop_actions.opinions[0].action
    assert isinstance(stop_action, CLOSE)
    assert stop_action.action["quantity_diff"] == -1
    assert stop_action.action["new_quantity"] == 0
    assert "55.00%" in stop_action.reason
    assert metadata.stop_triggered is True
    assert metadata.stop_triggered_pct == pytest.approx(0.55)
    assert metadata.stop_triggered_date == datetime(2018, 7, 1)
    assert cog._analyze_impl(_analysis_context([at_stop])).opinions == []

    stored = cog._position_store.get_metadata(TRADE_ID, SIGNAL_ID)
    rebuilt = metadata_from_store_payload(stored)
    assert rebuilt is not None
    assert rebuilt.threshold_triggered is True
    assert rebuilt.threshold_triggered_pct == pytest.approx(1.1)
    assert rebuilt.threshold_triggered_date == datetime(2018, 7, 1)
    assert rebuilt.waterfall_stop_reference_pnl_pct == pytest.approx(1.1)
    assert rebuilt.waterfall_stop_pnl_pct == pytest.approx(0.55)
    assert rebuilt.stop_triggered is True
    assert rebuilt.stop_triggered_pct == pytest.approx(0.55)
    assert rebuilt.stop_triggered_date == datetime(2018, 7, 1)


def test_correct_position_pnl_prefers_ledger_and_falls_back() -> None:
    """correct_position_pnl should use trade total PnL when present, else open-book."""
    from EventDriven.riskmanager.position.cogs.pnl_utils import correct_position_pnl

    trade = _make_trade(total_pnl=825.0, avg_price=250.0, quantity=3)
    ## Misleading open-book fields after a fictional trim should be ignored.
    with_ledger = _open_position(pnl=1.0, quantity=1, entry_price=750.0, trades=trade)
    corrected = correct_position_pnl(with_ledger, fallback_initial_qty=3)
    assert corrected.used_ledger is True
    assert corrected.pnl == pytest.approx(825.0)
    assert corrected.entry_price == pytest.approx(250.0)
    assert corrected.initial_quantity == 3
    assert corrected.initial_cost == pytest.approx(750.0)
    assert corrected.pnl_pct == pytest.approx(1.1)

    no_ledger = _open_position(pnl=7.5, quantity=3, entry_price=2.5)
    fallback = correct_position_pnl(no_ledger, fallback_initial_qty=3)
    assert fallback.used_ledger is False
    assert fallback.pnl_pct == pytest.approx(1.0)


def test_waterfall_pnl_pct_prefers_trade_ledger_over_open_book() -> None:
    """Ledger total PnL / initial cost should drive %, not open-book pos_state.pnl."""
    cog, _ = _make_cog(enable_profit_waterfall=True, waterfall_profit_threshold=1.0)

    def _calc(_m: int, _p: float, _t: float) -> int:
        return 3

    cog.calculator = _calc
    cog.on_new_position(_new_position_state(option_price=2.5))

    ## Ledger says 110% (total_pnl 825 over initial cost 250*3=750) even though the
    ## misleading open-book pnl below would only read as ~1% on entry_price*qty.
    trade = _make_trade(total_pnl=825.0, avg_price=250.0, quantity=3)
    crossed = _open_position(pnl=1.0, quantity=3, entry_price=250.0, trades=trade)
    actions = cog._analyze_impl(_analysis_context([crossed]))
    assert len(actions.opinions) == 1
    assert isinstance(actions.opinions[0].action, CLOSE)
    meta = cog.position_metadata[TRADE_ID]
    assert meta.threshold_triggered_pct == pytest.approx(1.1)


def test_waterfall_stop_stable_after_trim_via_trade_ledger() -> None:
    """After a half-close the ledger keeps % near crossing; stop fires only on giveback."""
    cog, _ = _make_cog(
        enable_profit_waterfall=True,
        waterfall_profit_threshold=1.0,
        enable_waterfall_stop_loss=True,
        waterfall_stop_loss_offset=0.5,
    )

    def _calc(_m: int, _p: float, _t: float) -> int:
        return 4

    cog.calculator = _calc
    cog.on_new_position(_new_position_state(option_price=2.5))

    ## Initial cost = avg_price*qty = 250*4 = 1000. total_pnl 1100 -> 110% crossing.
    crossed = _open_position(
        pnl=1100.0, quantity=4, entry_price=250.0,
        trades=_make_trade(total_pnl=1100.0, avg_price=250.0, quantity=4),
    )
    trim = cog._analyze_impl(_analysis_context([crossed]))
    assert len(trim.opinions) == 1
    assert isinstance(trim.opinions[0].action, CLOSE)
    assert trim.opinions[0].action.action["quantity_diff"] == -2
    meta = cog.position_metadata[TRADE_ID]
    assert meta.waterfall_stop_pnl_pct == pytest.approx(0.55)

    ## Same mark after trim: buy_ledger.quantity stays 4 (initial), total_pnl still
    ## ~1100 -> 110%, comfortably above the 55% stop. No action.
    held = _open_position(
        pnl=1050.0, quantity=2, entry_price=500.0,
        trades=_make_trade(total_pnl=1100.0, avg_price=250.0, quantity=4),
    )
    assert cog._analyze_impl(_analysis_context([held])).opinions == []

    ## total_pnl falls to 550 -> 55% vs initial cost 1000 -> stop fires, closes all.
    giveback = _open_position(
        pnl=550.0, quantity=2, entry_price=500.0,
        trades=_make_trade(total_pnl=550.0, avg_price=250.0, quantity=4),
    )
    stop = cog._analyze_impl(_analysis_context([giveback]))
    assert len(stop.opinions) == 1
    assert isinstance(stop.opinions[0].action, CLOSE)
    assert stop.opinions[0].action.action["new_quantity"] == 0
    assert meta.stop_triggered is True
    assert meta.stop_triggered_pct == pytest.approx(0.55)


def test_waterfall_respects_custom_threshold() -> None:
    """Below waterfall_profit_threshold should emit nothing."""
    cog, _ = _make_cog(enable_profit_waterfall=True, waterfall_profit_threshold=1.5)

    def _calc(_m: int, _p: float, _t: float) -> int:
        return 4

    cog.calculator = _calc
    cog.on_new_position(_new_position_state(option_price=2.5))

    ## initial cost = 10; pnl 10 -> 100% < 150%
    pos = _open_position(pnl=10.0, quantity=4, entry_price=2.5)
    assert cog._analyze_impl(_analysis_context([pos])).opinions == []

    ## pnl 15 -> 150% fires ceil(4/2)=2
    pos_hit = _open_position(pnl=15.0, quantity=4, entry_price=2.5)
    actions = cog._analyze_impl(_analysis_context([pos_hit]))
    assert len(actions.opinions) == 1
    assert isinstance(actions.opinions[0].action, CLOSE)
    assert actions.opinions[0].action.action["quantity_diff"] == -2


def _make_trade(*, total_pnl: float, avg_price: float, quantity: int) -> Trade:
    """Build a real Trade with a seeded buy ledger and total PnL.

    Args:
        total_pnl: Scaled realized + unrealized PnL in dollars.
        avg_price: Scaled per-contract entry price (premium * 100).
        quantity: Total contracts bought for the trade.

    Returns:
        Trade whose ``buy_ledger`` and ``total_pnl`` drive ledger-based pnl%.
    """
    trade = Trade(TRADE_ID, TICKER, SIGNAL_ID)
    trade.buy_ledger.avg_price = avg_price
    trade.buy_ledger.quantity = quantity
    trade.total_pnl = total_pnl
    return trade


def _open_position(
    *,
    pnl: float,
    quantity: int,
    entry_price: float,
    signal_id: str = SIGNAL_ID,
    trades: Optional[Trade] = None,
) -> PositionState:
    """Build an open PositionState for analysis tests.

    Args:
        pnl: Current open-book PnL (fallback path).
        quantity: Open quantity.
        entry_price: Entry premium per contract share.
        signal_id: Signal identifier on the position.
        trades: Optional fake trade ledger to exercise the ledger-based pnl%.

    Returns:
        Position state used inside a portfolio snapshot.
    """
    return PositionState(
        trade_id=TRADE_ID if signal_id == SIGNAL_ID else f"{TRADE_ID}-other",
        signal_id=signal_id,
        underlier_tick=TICKER,
        quantity=quantity,
        entry_price=entry_price,
        current_position_data=_option_data(entry_price),
        current_underlier_data=_undl_data(),
        pnl=pnl,
        last_updated=datetime(2018, 7, 1),
        trades=trades,
    )


def _analysis_context(positions: List[PositionState]) -> PositionAnalysisContext:
    """Wrap positions in a PositionAnalysisContext.

    Args:
        positions: Open positions to analyze.

    Returns:
        Analysis context for ``_analyze_impl``.
    """
    last_updated = datetime(2018, 7, 1)
    return PositionAnalysisContext(
        date=last_updated,
        portfolio=PortfolioState(cash=100_000, positions=positions, last_updated=last_updated),
        portfolio_meta=PortfolioMetaInfo(t_plus_n=1, is_backtest=True, start_date=datetime(2018, 1, 1)),
    )
