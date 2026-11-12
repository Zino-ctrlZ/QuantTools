"""FeatureDataManager: pull history, compute features, L1/L2 cache, slice.

Comment density: orchestration.

Processing Flow:
    1. Validate feature names.
    2. L1 → L2 lookup per feature.
    3. On miss: ``FeatureInputs`` pulls market data (datamanager / FRED),
       compute on ``start``→``end``, write L2 + L1.
    4. Slice onto ``index`` if given; return ``FeatureResult``.
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
    FeatureSpec,
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
        out = mgr.get_features(index=df.index)  # market; +hybrid+asset if ticker+ohlcv
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

    def get_features(
        self,
        features: Optional[Sequence[str]] = None,
        *,
        index: Optional[pd.DatetimeIndex] = None,
        start: Optional[DATE_HINT] = None,
        end: Optional[DATE_HINT] = None,
        ticker: Optional[str] = None,
        ohlcv: Optional[pd.DataFrame] = None,
        recompute: bool = False,
    ) -> FeatureResult:
        """Return features for ``start``→``end``, optionally sliced to ``index``.

        Args:
            features: Feature names. Default ``None`` loads all market features,
                and also hybrid IV−rvol plus asset-local features when both
                ``ticker`` and ``ohlcv`` are passed. Pass an explicit list only
                to subset.
            index: Optional calendar to reindex onto after load/compute.
            start: History start (default 1970-01-01).
            end: History end (default last NY business day).
            ticker: Asset id for hybrid features. Must be passed with ``ohlcv``.
            ohlcv: Asset bars for hybrid features. Must be passed with ``ticker``.
            recompute: Ignore L1/L2 and rewrite disk cache.

        Returns:
            ``FeatureResult`` with a ``series`` mapping.

        Raises:
            ValueError: If ``ticker`` / ``ohlcv`` are unpaired, or hybrids are
                requested without both.

        Note:
            Default without asset args is all market. Default with ticker+ohlcv
            is market + hybrid + asset-local. L1/L2 hits are sliced and returned;
            only misses pull data and compute.
        """
        ## ticker+ohlcv means the caller has an asset context → include hybrid
        ## and asset-local names in the default set (explicit ``features=``
        ## still wins as a subset).
        has_asset = (
            ticker is not None
            and str(ticker).strip() != ""
            and ohlcv is not None
        )
        names = resolve_features(
            features,
            include_hybrid=has_asset,
            include_asset=has_asset,
        )
        self._require_ticker_ohlcv_pair(names, ticker=ticker, ohlcv=ohlcv)
        hist_start, hist_end = self.inputs.history_bounds(
            pd.Timestamp(to_datetime(start)).normalize() if start is not None else None,
            pd.Timestamp(to_datetime(end)).normalize() if end is not None else None,
        )
        logger.info(
            "get_features n=%s %s→%s ticker=%s",
            len(names),
            hist_start.date(),
            hist_end.date(),
            ticker,
        )

        out: Dict[str, pd.Series] = {}
        misses: List[FeatureSpec] = []

        for name in names:
            spec = get_spec(name)
            key = self._key(spec, ticker)

            if not recompute:
                hit = self._l1.get(key)
                if hit is not None and self._covers(hit, hist_start, hist_end):
                    out[name] = self._slice(hit, index)
                    continue
                cached = self._read_l2(key, hist_start, hist_end)
                if cached is not None:
                    self._l1.set(key, cached)
                    out[name] = self._slice(cached, index)
                    continue

            misses.append(spec)

        if misses:
            computed = self._compute(
                misses,
                hist_start=hist_start,
                hist_end=hist_end,
                ticker=ticker,
                ohlcv=ohlcv,
            )
            for name, series in computed.items():
                key = self._key(get_spec(name), ticker)
                self.cache_it(key, series)
                self._l1.set(key, series)
                out[name] = self._slice(series, index)

        return FeatureResult(series=out, ticker=ticker)

    def clear(self) -> None:
        """Clear L1, L2, and pulled market inputs."""
        self._l1.clear()
        self.clear_cache()
        self.inputs.clear()
        logger.info("FeatureDataManager cleared")

    def list_features(self, scope: str = "all") -> List[str]:
        """List catalog feature names."""
        return list_features(scope=scope)

    def feature_spec(self, name: str) -> FeatureSpec:
        """Return catalog metadata for one feature."""
        return get_spec(name)

    def cache_it(self, key: str, value: Any, *, expire: Optional[int] = None) -> None:
        """Write a feature Series to L2."""
        _data_structure_cache_it(self, key, value, expire=expire)

    # --- internals ----------------------------------------------------------

    @staticmethod
    def _require_ticker_ohlcv_pair(
        names: Sequence[str],
        *,
        ticker: Optional[str],
        ohlcv: Optional[pd.DataFrame],
    ) -> None:
        """Enforce ticker and ohlcv as a pair; require both when hybrids appear.

        Args:
            names: Resolved feature names for this call.
            ticker: Optional asset ticker.
            ohlcv: Optional asset OHLCV frame.

        Raises:
            ValueError: If only one of ticker/ohlcv is set, or hybrids lack both.
        """
        has_ticker = ticker is not None and str(ticker).strip() != ""
        has_ohlcv = ohlcv is not None
        if has_ticker != has_ohlcv:
            raise ValueError(
                "ticker and ohlcv must be passed together "
                f"(ticker={'set' if has_ticker else 'missing'}, "
                f"ohlcv={'set' if has_ohlcv else 'missing'})"
            )
        needs_asset = any(get_spec(name).scope != "market" for name in names)
        if needs_asset and not (has_ticker and has_ohlcv):
            raise ValueError(
                "Non-market features require both ticker= and ohlcv= "
                f"(requested non-market among {len(names)} names)"
            )

    def _key(self, spec: FeatureSpec, ticker: Optional[str]) -> str:
        """Build the shared L1/L2 key for one feature."""
        if spec.scope == "market":
            symbol = MARKET_SYMBOL
        else:
            if ticker is None:
                raise ValueError(f"Feature {spec.name!r} requires ticker=")
            symbol = ticker.upper()
        return feature_cache_key(
            self,
            symbol=symbol,
            feature=spec.name,
            formula_id=spec.formula_id,
        )

    @staticmethod
    def _covers(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        """True when an in-memory (L1) Series spans ``start``→``end``.

        L1 stores bare Series, so this is a light index-span check. L2 uses
        ``_CachedData.is_fully_covered`` via ``_read_l2`` instead.
        """
        if series is None or series.empty:
            return False
        return (
            pd.Timestamp(series.index.min()).normalize() <= start
            and pd.Timestamp(series.index.max()).normalize() >= end
        )

    @staticmethod
    def _slice(series: pd.Series, index: Optional[pd.DatetimeIndex]) -> pd.Series:
        """Copy and optionally reindex."""
        held = series.copy()
        held.index = to_datetime(held.index)
        if index is None:
            return held
        return held.reindex(pd.DatetimeIndex(to_datetime(index)))

    def _read_l2(
        self,
        key: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> Optional[pd.Series]:
        """Return L2 Series when present and fully covering ``start``→``end``.

        Uses ``_CachedData.is_fully_covered`` (same coverage helper as other
        datamanagers). Legacy bare Series entries are promoted first.
        """
        raw = self.get(key, default=None)
        if raw is None:
            return None
        ## Same promotion path as ``_data_structure_cache_check_missing``.
        if isinstance(raw, (pd.Series, pd.DataFrame)):
            raw = _CachedData(key=key, data=raw)
        if not isinstance(raw, _CachedData):
            return None
        if not raw.is_fully_covered(start, end):
            return None
        cached = _extract_data(raw)
        if not isinstance(cached, pd.Series) or cached.empty:
            return None
        return cached

    def _compute(
        self,
        misses: Sequence[FeatureSpec],
        *,
        hist_start: pd.Timestamp,
        hist_end: pd.Timestamp,
        ticker: Optional[str],
        ohlcv: Optional[pd.DataFrame],
    ) -> Dict[str, pd.Series]:
        """Pull inputs and compute missing features via ``Feature.compute(ctx)``."""
        grid = pd.bdate_range(hist_start, hist_end)
        spy = self.inputs.close("SPY", grid)
        compute_index = pd.DatetimeIndex(spy.dropna().index) if spy.notna().any() else grid
        miss_names = [s.name for s in misses]

        needs_asset_bars = any(s.scope in {"hybrid", "asset"} for s in misses)
        asset_frame: Optional[pd.DataFrame] = None
        if needs_asset_bars:
            if ohlcv is None:
                raise ValueError(
                    f"Hybrid/asset features require ohlcv= (n_miss={len(miss_names)})"
                )
            asset_frame = normalize_ohlcv(ohlcv).reindex(compute_index)

        ## Load every close / IV key declared on the missed features.
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
