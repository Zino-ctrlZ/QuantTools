"""Pull and cache market OHLC / FRED IV for feature compute.

One job: given a symbol or IV key and a date index, return aligned data.
Pulls go through datamanager (spots) or FRED (IV). Results are kept in a
process-local dict so one ``get_features`` batch does not re-fetch SPY
twenty times.

Comment density: orchestration.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import pandas as pd

from trade.features.compute.symbols import (
    FRED_IV_SERIES,
    HISTORY_START,
    IV_MAX_STALENESS_DAYS,
    MARKET_SYMBOLS,
)
from trade.helpers.Logging import setup_logger
from trade.helpers.helper import ny_now_busday, to_datetime

logger = setup_logger(__name__)


class FeatureInputs:
    """Market data helper used by ``FeatureDataManager``.

    Usage:
        inputs = FeatureInputs()
        close = inputs.close("SPY", index)   # pulls + caches as needed
        iv = inputs.iv("EEM_IV", index)
    """

    def __init__(
        self,
        *,
        history_start: pd.Timestamp = HISTORY_START,
        iv_max_staleness_days: int = IV_MAX_STALENESS_DAYS,
    ) -> None:
        """Create empty process-local OHLC / IV caches.

        Args:
            history_start: Default lower bound for live vendor pulls.
            iv_max_staleness_days: Max business days to forward-fill IV.
        """
        self.history_start = pd.Timestamp(history_start).normalize()
        self.iv_max_staleness_days = int(iv_max_staleness_days)
        ## symbol → full OHLC frame
        self._ohlc: Dict[str, pd.DataFrame] = {}
        ## iv key → decimal IV series
        self._iv: Dict[str, pd.Series] = {}
        ## iv key → last non-NaN observation (staleness cutoff)
        self._iv_last_native: Dict[str, pd.Timestamp] = {}

    def clear(self) -> None:
        """Drop all cached OHLC and IV."""
        self._ohlc.clear()
        self._iv.clear()
        self._iv_last_native.clear()

    def history_bounds(
        self,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Return normalized ``(start, end)`` for a pull window.

        Defaults: ``history_start`` → last NY business day.
        """
        lo = pd.Timestamp(start).normalize() if start is not None else self.history_start
        if end is not None:
            hi = pd.Timestamp(end).normalize()
        else:
            hi = pd.Timestamp(pd.Timestamp(ny_now_busday()).strftime("%Y-%m-%d"))
        if hi < lo:
            hi = lo
        return lo, hi

    def seed_ohlc(self, frames: Mapping[str, pd.DataFrame]) -> None:
        """Install test/notebook OHLC frames (skips vendor pull for those keys)."""
        for key, frame in frames.items():
            self._ohlc[self._spot_key(key)] = self._normalize_ohlc(frame)

    def seed_iv(self, series_map: Mapping[str, pd.Series]) -> None:
        """Install test/notebook IV series (skips FRED pull for those keys)."""
        for key, series in series_map.items():
            iv_key = self._iv_key(key)
            normalized = self._normalize_iv(series)
            self._iv[iv_key] = normalized
            self._iv_last_native[iv_key] = (
                pd.Timestamp(normalized.dropna().index.max()).normalize()
                if normalized.notna().any()
                else pd.Timestamp(normalized.index.max()).normalize()
            )

    def close(self, symbol: str, index: pd.DatetimeIndex) -> pd.Series:
        """Return MarketTimeseries ``close`` reindexed to ``index``.

        Spot frames expose ``close`` (split-adjusted) and ``unadjusted_close``;
        features use ``close``. Gaps stay NaN.
        """
        return self.ohlc(symbol, index)["close"].reindex(index)

    def ohlc(self, symbol: str, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Return OHLC reindexed to ``index``."""
        key = self._spot_key(symbol)
        if key not in self._ohlc:
            start, end = self.history_bounds()
            logger.info("pull spot %s %s→%s", key, start.date(), end.date())
            self._ohlc[key] = self._pull_spot(key, start, end)
        return self._ohlc[key].reindex(index)

    def iv(self, iv_key: str, index: pd.DatetimeIndex) -> pd.Series:
        """Return decimal IV reindexed to ``index`` with short forward-fill."""
        key = self._iv_key(iv_key)
        if key not in self._iv:
            start, end = self.history_bounds()
            logger.info("pull iv %s %s→%s", key, start.date(), end.date())
            series = self._pull_iv(key, start, end)
            self._iv[key] = series
            self._iv_last_native[key] = (
                pd.Timestamp(series.dropna().index.max()).normalize()
                if series.notna().any()
                else end
            )
        aligned = self._iv[key].reindex(index).ffill()
        cutoff = self._iv_last_native[key] + pd.tseries.offsets.BDay(self.iv_max_staleness_days)
        return aligned.where(pd.DatetimeIndex(aligned.index) <= cutoff)

    # --- vendor pulls -------------------------------------------------------

    def _pull_spot(self, key: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Fetch one market symbol via ``MarketTimeseries``."""
        from trade.datamanager.vars import get_times_series

        vendor = MARKET_SYMBOLS[key]
        frame = get_times_series().get_timeseries(
            vendor,
            "spot",
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
        ).spot
        return self._normalize_ohlc(frame)

    def _pull_iv(self, key: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Fetch one FRED IV series and convert percentage points → decimal."""
        import pandas_datareader.data as web

        raw = web.DataReader(
            FRED_IV_SERIES[key],
            "fred",
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
        )
        return self._normalize_iv(raw.iloc[:, 0])

    # --- normalize / keys ---------------------------------------------------

    @staticmethod
    def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
        """Lowercase columns and sort by date."""
        out = df.copy()
        out.index = to_datetime(out.index)
        out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
        return out.sort_index()

    @staticmethod
    def _normalize_iv(series: pd.Series) -> pd.Series:
        """Sort, coerce numeric, convert percentage-point levels to decimal."""
        out = pd.to_numeric(series.copy(), errors="coerce")
        out.index = to_datetime(out.index)
        out = out.sort_index()
        if out.notna().any() and float(out.dropna().max()) > 3.0:
            out = out / 100.0
        return out

    @staticmethod
    def _spot_key(symbol: str) -> str:
        """Map aliases onto a canonical ``MARKET_SYMBOLS`` key."""
        upper = symbol.upper()
        for key, vendor in MARKET_SYMBOLS.items():
            if upper == key or upper == vendor.upper():
                return key
        if upper.startswith("^") and upper[1:] in MARKET_SYMBOLS:
            return upper[1:]
        raise KeyError(f"Unknown market symbol {symbol!r}. Known: {sorted(MARKET_SYMBOLS)}")

    @staticmethod
    def _iv_key(symbol: str) -> str:
        """Map aliases onto a canonical ``FRED_IV_SERIES`` key."""
        upper = symbol.upper()
        if upper in FRED_IV_SERIES:
            return upper
        for key, fred_id in FRED_IV_SERIES.items():
            if upper == fred_id.upper():
                return key
        if upper in {"EEM", "VXEEM", "VXEEMCLS"}:
            return "EEM_IV"
        if upper in {"FXI", "VXFXI", "VXFXICLS"}:
            return "FXI_IV"
        raise KeyError(f"Unknown IV series {symbol!r}. Known: {sorted(FRED_IV_SERIES)}")
