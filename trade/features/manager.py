"""FeatureDataManager: pull history, compute features, L1/L2 cache, slice.

Comment density: orchestration.

Processing Flow:
    1. Pick an endpoint (market / ticker self-query / caller OHLCV).
    2. Resolve feature names.
    3. L1 → (L2 if durable) lookup. If cache does not reach the needed end,
       compute the full available history, strip edge NaNs, write cache.
    4. Return: clip with caller ``start``/``end``, then optional ``index``.

Public endpoints:
    get_market_features: market-only (no ticker / ohlcv).
    get_features_for_ticker: self-queries asset spot; durable L1+L2.
    get_features_from_ohlcv: requires ticker+ohlcv; L1-only custom bars.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Sequence, Union

import pandas as pd

from trade.datamanager._enums import Interval, SeriesId
from trade.datamanager.base import BaseDataManager, CacheSpec
from trade.datamanager.utils.cache import (
    _CachedData,
    _data_structure_cache_it,
    _extract_data,
)
from trade.features.catalog import (
    Feature,
    get_spec,
    list_features,
    resolve_features,
)
from trade.features.compute.asset.ohlcv_util import normalize_ohlcv
from trade.features.compute.context import FeatureContext
from trade.features.compute.registry import iv_keys_for_features, symbols_for_features
from trade.features.inputs import FeatureInputs
from trade.features.keys import MARKET_SYMBOL, feature_cache_key
from trade.features.l1 import FeatureL1Cache
from trade.features.result import FeatureResult
from trade.helpers.Logging import setup_logger
from trade.helpers.helper import to_datetime

logger = setup_logger(__name__)

DATE_HINT = Union[pd.Timestamp, str]


class FeatureDataManager(BaseDataManager):
    """Named feature Series with L1, CustomCache L2, and in-manager data pulls.

    Example:
        mgr = FeatureDataManager()
        market = mgr.get_market_features(index=df.index)
        vendor = mgr.get_features_for_ticker("AAPL", features=["ret_20d"])
        custom = mgr.get_features_from_ohlcv("AAPL", aapl_ohlcv, features=["ret_20d"])
    """

    CACHE_NAME: ClassVar[str] = "feature_data_manager"
    CACHE_SPEC: ClassVar[CacheSpec] = CacheSpec(cache_fname=CACHE_NAME)
    DEFAULT_SERIES_ID: ClassVar[SeriesId] = SeriesId.HIST
    DEFAULT_INTERVAL: ClassVar[Optional[Interval]] = Interval.EOD
    INSTANCE: ClassVar[Optional["FeatureDataManager"]] = None

    def __new__(cls, *, enable_namespacing: bool = True) -> "FeatureDataManager":
        """Return the process singleton."""
        if cls.INSTANCE is not None:
            return cls.INSTANCE
        instance = object.__new__(cls)
        cls.INSTANCE = instance
        return instance

    def __init__(self, *, enable_namespacing: bool = True) -> None:
        """Initialize L1, inputs helper, and CustomCache once."""
        if getattr(self, "_initialized", False):
            return
        super().__init__(enable_namespacing=enable_namespacing)
        self._l1 = FeatureL1Cache()
        self.inputs = FeatureInputs()
        self._initialized = True
        logger.info("FeatureDataManager ready cache=%s", self.CACHE_NAME)

    # --- public endpoints ---------------------------------------------------

    def get_market_features(
        self,
        features: Optional[Sequence[str]] = None,
        *,
        index: Optional[pd.DatetimeIndex] = None,
        start: Optional[DATE_HINT] = None,
        end: Optional[DATE_HINT] = None,
        recompute: bool = False,
    ) -> FeatureResult:
        """Return **market-only** features (no ticker / OHLCV).

        Durable cache holds full market history. ``start`` / ``end`` only clip
        the returned series (and ``end`` can extend the cache if needed).

        Args:
            features: Feature names. Default ``None`` loads all active market
                features (deprecated names omitted).
            index: Optional calendar to reindex onto after the return clip.
            start: Return-window floor only.
            end: Return-window ceiling only (default last NY business day).
            recompute: Ignore L1/L2 and rewrite disk cache.

        Returns:
            ``FeatureResult`` clipped to ``start``/``end`` (aligned to ``index``
            when given).
        """
        names = resolve_features(features, include_hybrid=False, include_asset=False)
        if any(get_spec(name).scope != "market" for name in names):
            raise ValueError(
                "get_market_features is market-only; use get_features_for_ticker "
                "(self-queried spot) or get_features_from_ohlcv (caller bars)"
            )
        return self._get_features_impl(
            names,
            index=index,
            start=start,
            end=end,
            ticker=None,
            ohlcv=None,
            custom_bars=False,
            persist_l2=True,
            recompute=recompute,
        )

    def get_features_for_ticker(
        self,
        ticker: str,
        features: Optional[Sequence[str]] = None,
        *,
        index: Optional[pd.DatetimeIndex] = None,
        start: Optional[DATE_HINT] = None,
        end: Optional[DATE_HINT] = None,
        recompute: bool = False,
    ) -> FeatureResult:
        """Return features for ``ticker`` using **self-queried** vendor spot.

        Durable L1/L2 store listing→end history. ``start`` / ``end`` only clip
        the returned series.

        Args:
            ticker: Equity ticker (required).
            features: Names to compute. Default is active market + hybrid +
                asset (deprecated names omitted).
            index: Optional calendar to reindex onto after the return clip.
            start: Return-window floor only.
            end: Return-window ceiling only (default last NY business day).
            recompute: Ignore L1/L2 and rewrite disk cache.

        Returns:
            ``FeatureResult`` clipped to ``start``/``end`` (aligned to ``index``
            when given).

        Raises:
            ValueError: If ``ticker`` is empty or spot history is empty.
        """
        ticker_key = self._require_ticker(ticker)
        names = resolve_features(features, include_hybrid=True, include_asset=True)
        ## Full listing→end bars for durable compute; start never truncates this.
        asset = self.inputs.asset_ohlcv(ticker_key, end=end)
        if asset.empty:
            raise ValueError(f"No spot history for ticker={ticker_key!r}")
        logger.info(
            "get_features_for_ticker ticker=%s n=%s bars=%s→%s",
            ticker_key,
            len(names),
            pd.Timestamp(asset.index.min()).date(),
            pd.Timestamp(asset.index.max()).date(),
        )
        return self._get_features_impl(
            names,
            index=index,
            start=start,
            end=end,
            ticker=ticker_key,
            ohlcv=asset,
            custom_bars=False,
            persist_l2=True,
            recompute=recompute,
        )

    def get_features_from_ohlcv(
        self,
        ticker: str,
        ohlcv: pd.DataFrame,
        features: Optional[Sequence[str]] = None,
        *,
        index: Optional[pd.DatetimeIndex] = None,
        start: Optional[DATE_HINT] = None,
        end: Optional[DATE_HINT] = None,
        recompute: bool = False,
    ) -> FeatureResult:
        """Return features from **caller-supplied** ``ticker`` + ``ohlcv``.

        Both arguments are required. L1-only under ``bars=custom`` (no L2).
        ``start`` / ``end`` only clip the returned series.

        Args:
            ticker: Equity ticker (required; labels the L1 key).
            ohlcv: Asset OHLCV frame (required; must include ``close``).
            features: Names to compute. Default is active market + hybrid + asset.
            index: Optional calendar to reindex onto after the return clip.
            start: Return-window floor only.
            end: Return-window ceiling only.
            recompute: Ignore L1 and recompute from ``ohlcv``.

        Returns:
            ``FeatureResult`` clipped to ``start``/``end`` (aligned to ``index``
            when given).

        Raises:
            ValueError: If ``ticker`` is empty or ``ohlcv`` is missing/empty.
            TypeError: If ``ohlcv`` is not a DataFrame.
        """
        ticker_key = self._require_ticker(ticker)
        if ohlcv is None:
            raise ValueError(
                "get_features_from_ohlcv requires ohlcv= (caller bars); "
                "use get_features_for_ticker for self-queried spot"
            )
        if not isinstance(ohlcv, pd.DataFrame):
            raise TypeError(f"ohlcv must be a DataFrame, got {type(ohlcv)}")
        asset = normalize_ohlcv(ohlcv)
        if asset.empty:
            raise ValueError("ohlcv is empty")
        names = resolve_features(features, include_hybrid=True, include_asset=True)
        logger.info(
            "get_features_from_ohlcv ticker=%s n=%s bars=%s→%s",
            ticker_key,
            len(names),
            pd.Timestamp(asset.index.min()).date(),
            pd.Timestamp(asset.index.max()).date(),
        )
        return self._get_features_impl(
            names,
            index=index,
            start=start,
            end=end,
            ticker=ticker_key,
            ohlcv=asset,
            custom_bars=True,
            persist_l2=False,
            recompute=recompute,
        )

    def clear(self) -> None:
        """Clear L1, L2, and pulled market/asset inputs."""
        self._l1.clear()
        self.clear_cache()
        self.inputs.clear()
        logger.info("FeatureDataManager cleared")

    def list_features(self, scope: str = "all") -> List[str]:
        """List catalog feature names."""
        return list_features(scope=scope)

    def feature_spec(self, name: str) -> Feature:
        """Return catalog metadata for one feature."""
        return get_spec(name)

    def cache_it(self, key: str, value: Any, *, expire: Optional[int] = None) -> None:
        """Write a feature Series to L2."""
        _data_structure_cache_it(self, key, value, expire=expire)

    # --- shared orchestration -----------------------------------------------

    def _get_features_impl(
        self,
        names: Sequence[str],
        *,
        index: Optional[pd.DatetimeIndex],
        start: Optional[DATE_HINT],
        end: Optional[DATE_HINT],
        ticker: Optional[str],
        ohlcv: Optional[pd.DataFrame],
        custom_bars: bool,
        persist_l2: bool,
        recompute: bool,
    ) -> FeatureResult:
        """Lookup cache; on miss compute full available history; slice to return.

        ``start`` / ``end`` are return clips only. Cache must reach through the
        needed end (caller ``end``, else last available bar / last bus day).
        """
        start_ts = self._as_ts(start)
        end_ts = self._as_ts(end)
        ## Cache must be good through this date; missing → recompute/extend.
        need_end = end_ts if end_ts is not None else self._available_end(ohlcv)

        out: Dict[str, pd.Series] = {}
        misses: List[Feature] = []

        for name in names:
            spec = get_spec(name)
            key = self._key(spec, ticker, custom_bars=custom_bars)

            if not recompute:
                hit = self._l1.get(key)
                if hit is not None and self._covers_through(hit, need_end):
                    out[name] = self._slice(hit, index, start=start_ts, end=end_ts)
                    continue
                if persist_l2:
                    cached = self._read_l2(key, need_end)
                    if cached is not None:
                        self._l1.set(key, cached)
                        out[name] = self._slice(
                            cached, index, start=start_ts, end=end_ts
                        )
                        continue

            misses.append(spec)

        if misses:
            computed = self._compute(misses, ticker=ticker, ohlcv=ohlcv, end=need_end)
            for name, series in computed.items():
                key = self._key(get_spec(name), ticker, custom_bars=custom_bars)
                ## Drop leading/trailing NaN padding before any cache write.
                durable = self._strip_edge_nans(series)
                if persist_l2:
                    self.cache_it(key, durable)
                self._l1.set(key, durable)
                out[name] = self._slice(durable, index, start=start_ts, end=end_ts)

        return FeatureResult(series=out, ticker=ticker)

    # --- internals ----------------------------------------------------------

    @staticmethod
    def _as_ts(value: Optional[DATE_HINT]) -> Optional[pd.Timestamp]:
        """Normalize an optional date hint to a midnight timestamp."""
        if value is None:
            return None
        return pd.Timestamp(to_datetime(value)).normalize()

    def _available_end(self, ohlcv: Optional[pd.DataFrame]) -> pd.Timestamp:
        """Last date we can cover from bars or the market history default."""
        if ohlcv is not None and not ohlcv.empty:
            return pd.Timestamp(ohlcv.index.max()).normalize()
        return self.inputs.history_bounds(None, None)[1]

    @staticmethod
    def _require_ticker(ticker: str) -> str:
        """Normalize and require a non-empty ticker string."""
        if ticker is None or str(ticker).strip() == "":
            raise ValueError("ticker is required")
        return str(ticker).strip().upper()

    def _key(
        self,
        spec: Feature,
        ticker: Optional[str],
        *,
        custom_bars: bool = False,
    ) -> str:
        """Build the L1/L2 key for one feature."""
        if spec.scope == "market":
            symbol = MARKET_SYMBOL
            custom_bars = False
        else:
            if ticker is None:
                raise ValueError(f"Feature {spec.name!r} requires ticker=")
            symbol = ticker.upper()
        return feature_cache_key(
            self,
            symbol=symbol,
            feature=spec.name,
            formula_id=spec.formula_id,
            custom_bars=custom_bars,
        )

    @staticmethod
    def _strip_edge_nans(series: pd.Series) -> pd.Series:
        """Drop leading and trailing NaNs; keep the valid interior span."""
        if series is None or series.empty or series.notna().sum() == 0:
            return series.copy() if series is not None else series
        first = series.first_valid_index()
        last = series.last_valid_index()
        return series.loc[first:last].copy()

    @staticmethod
    def _covers_through(series: pd.Series, end: pd.Timestamp) -> bool:
        """True when cached feature history reaches ``end`` (warmup gaps OK)."""
        if series is None or series.empty:
            return False
        return pd.Timestamp(series.index.max()).normalize() >= end

    @staticmethod
    def _slice(
        series: pd.Series,
        index: Optional[pd.DatetimeIndex],
        *,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.Series:
        """Copy, clip to optional ``start``/``end``, then optional ``index``."""
        held = series.copy()
        held.index = to_datetime(held.index)
        if start is not None:
            held = held.loc[held.index >= start]
        if end is not None:
            held = held.loc[held.index <= end]
        if index is None:
            return held
        return held.reindex(pd.DatetimeIndex(to_datetime(index)))

    def _read_l2(self, key: str, end: pd.Timestamp) -> Optional[pd.Series]:
        """Return L2 Series when present and covering through ``end``."""
        raw = self.get(key, default=None)
        if raw is None:
            return None
        if isinstance(raw, (pd.Series, pd.DataFrame)):
            raw = _CachedData(key=key, data=raw)
        if not isinstance(raw, _CachedData):
            return None
        cached = _extract_data(raw)
        if not isinstance(cached, pd.Series) or cached.empty:
            return None
        if not self._covers_through(cached, end):
            return None
        return cached

    def _compute(
        self,
        misses: Sequence[Feature],
        *,
        ticker: Optional[str],
        ohlcv: Optional[pd.DataFrame],
        end: pd.Timestamp,
    ) -> Dict[str, pd.Series]:
        """Compute missing features on the full available bar calendar."""
        miss_names = [s.name for s in misses]
        needs_asset_bars = any(s.scope in {"hybrid", "asset"} for s in misses)
        asset_frame: Optional[pd.DataFrame] = None

        if needs_asset_bars:
            if ohlcv is None:
                raise ValueError(
                    f"Hybrid/asset features require asset bars (n_miss={len(miss_names)})"
                )
            ## Full caller/vendor bars up through need-end (never truncated by start).
            asset_frame = normalize_ohlcv(ohlcv)
            asset_frame = asset_frame.loc[asset_frame.index <= end]
            if asset_frame.empty:
                raise ValueError(f"Asset bars empty through {end.date()}")
            compute_index = pd.DatetimeIndex(asset_frame.index)
        else:
            lo, hi = self.inputs.history_bounds(None, end)
            grid = pd.bdate_range(lo, hi)
            spy = self.inputs.close("SPY", grid)
            compute_index = (
                pd.DatetimeIndex(spy.dropna().index) if spy.notna().any() else grid
            )

        close_keys = symbols_for_features(miss_names)
        closes = {sym: self.inputs.close(sym, compute_index) for sym in close_keys}
        ivs = {
            key: self.inputs.iv(key, compute_index)
            for key in iv_keys_for_features(miss_names)
        }
        ctx = FeatureContext(closes=closes, ivs=ivs, ohlcv=asset_frame)

        out: Dict[str, pd.Series] = {}
        for spec in misses:
            out[spec.name] = spec.compute(ctx)
        logger.info("compute_features n=%s rows=%s", len(out), len(compute_index))
        return out


def get_feature_data_manager() -> FeatureDataManager:
    """Return the process-global ``FeatureDataManager`` singleton."""
    return FeatureDataManager()


def reset_feature_data_manager() -> None:
    """Drop the singleton and clear caches (tests / notebooks)."""
    inst = FeatureDataManager.INSTANCE
    if inst is not None:
        try:
            inst.clear()
        except Exception as exc:  # noqa: BLE001
            logger.warning("reset_feature_data_manager cleanup err=%s", exc)
    FeatureDataManager.INSTANCE = None
