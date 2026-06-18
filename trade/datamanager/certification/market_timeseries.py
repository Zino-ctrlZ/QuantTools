"""Certification adapters for MarketTimeseries factor getters.

Each ``_get_*_timeseries`` method wraps its payload in a throwaway ``Result`` and
synthetic cache key, then delegates to ``certify_manager_result``. Keys are for
certifier metadata only — not written to ``SPOT_CACHE`` / ``CHAIN_SPOT_CACHE``.

Comment density: orchestration

Core Functions:
    make_market_timeseries_key: Synthetic certifier key for a market factor.
    certify_market_factor_payload: Sanitize, wrap, certify, return payload.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import pandas as pd

from trade.datamanager._enums import ArtifactType, CertificationLevel, Interval, SeriesId
from trade.datamanager.result import MarketTimeseriesFactorResult
from trade.datamanager.utils.data_structure import _data_structure_sanitize
from trade.datamanager.utils.enums_utils import construct_cache_key
from trade.helpers.helper_types import DATE_HINT

from .integration import certify_manager_result

PandasData = Union[pd.Series, pd.DataFrame]

_FACTOR_ARTIFACT: Dict[str, ArtifactType] = {
    "spot": ArtifactType.SPOT,
    "chain_spot": ArtifactType.CHAIN,
    "dividends": ArtifactType.DIVS,
    "dividend_yield": ArtifactType.DIVS,
    "split_factor": ArtifactType.SPOT,
}


def _build_market_factor_result(symbol: str, factor: str, data: PandasData) -> MarketTimeseriesFactorResult:
    """Build a throwaway result that accepts Series or DataFrame payloads."""
    return MarketTimeseriesFactorResult(symbol=symbol, timeseries=data, market_factor=factor)


def make_market_timeseries_key(
    symbol: str,
    factor: str,
    *,
    additional_name: Optional[str] = None,
) -> str:
    """Build a synthetic certifier key for a MarketTimeseries factor.

    Args:
        symbol: Equity ticker.
        factor: Market factor name (spot, chain_spot, dividends, etc.).
        additional_name: Optional label when ``factor`` is ``additional``.

    Returns:
        Pipe-delimited cache key parseable by the certification context.
    """
    artifact_type = _FACTOR_ARTIFACT.get(factor, ArtifactType.SPOT)
    extra: Dict[str, Any] = {
        "cache_namespace": "market_timeseries",
        "market_factor": factor,
    }
    if factor == "dividend_yield":
        extra["current_state"] = "continuous_yield"
    if additional_name is not None:
        extra["additional_name"] = additional_name
    return construct_cache_key(
        symbol=symbol,
        interval=Interval.EOD,
        artifact_type=artifact_type,
        series_id=SeriesId.HIST,
        **extra,
    )


def certify_market_factor_payload(
    symbol: str,
    factor: str,
    data: PandasData,
    start_date: DATE_HINT,
    end_date: DATE_HINT,
    *,
    method: str,
    additional_name: Optional[str] = None,
    certification_level: Optional[CertificationLevel] = None,
) -> PandasData:
    """Sanitize, wrap in a throwaway Result, certify, and return the payload.

    Args:
        symbol: Equity ticker.
        factor: Market factor name.
        data: Raw post-clip timeseries payload.
        start_date: Certification window start.
        end_date: Certification window end.
        method: Getter method name for na_logging labels.
        additional_name: Optional label for custom additional data factors.

    Returns:
        Certified payload; may be mutated at L3.

    Raises:
        DataNotCertifiedException: At L2 when structural checks fail.
        DataCertificationMissingInformationError: When key metadata is invalid.
        ValueError: When ``factor`` is not registered.
    """
    if data is None:
        return data
    if hasattr(data, "empty") and data.empty:
        return data

    sanitized = _data_structure_sanitize(
        data,
        start=start_date,
        end=end_date,
        source_name=f"market_timeseries {factor} for {symbol}",
    )

    cert_factor = "additional" if additional_name is not None else factor
    if cert_factor not in _FACTOR_ARTIFACT and cert_factor != "additional":
        raise ValueError(f"Unsupported market_timeseries factor for certification: {factor}")

    wrapper = _build_market_factor_result(symbol, cert_factor, sanitized)
    key = make_market_timeseries_key(
        symbol,
        cert_factor,
        additional_name=additional_name,
    )

    certified = certify_manager_result(
        wrapper,
        start_date,
        end_date,
        cache_key=key,
        manager="market_timeseries",
        method=method,
        level=certification_level,
    )
    return certified.timeseries  # type: ignore[return-value]
