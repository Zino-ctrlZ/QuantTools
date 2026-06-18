"""Certification context construction for timeseries audit runs.

Builds a ``CertificationContext`` from a manager ``Result``, requested date window,
and optional checked-missing calendar metadata before check leaves execute.

Core Dataclasses:
    CertificationContext: Mutable state shared across check and respond steps.

Core Functions:
    setup_context: Parse key metadata, sync option dates, and populate context.
    prepare_checked_missing_dates: Normalize checked-missing inputs to ``date`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Union

import pandas as pd

from trade.datamanager._enums import ArtifactType, CertificationLevel, OptionSpotEndpointSource
from trade.datamanager.result import Result
from trade.datamanager.utils.date import _sync_date
from trade.datamanager.utils.enums_utils import _parse_cache_key
from trade.helpers.helper import generate_option_tick_new, to_datetime
from trade.helpers.helper_types import DATE_HINT
from trade.helpers.Logging import setup_logger
from trade.datamanager.utils.logging import get_logging_level

from .exceptions import DataCertificationMissingInformationError
from .findings import OPTION_ARTIFACT_TYPES

logger = setup_logger("trade.datamanager.certification.context", stream_log_level=get_logging_level())


def effective_checked_missing_dates(ctx: "CertificationContext") -> List[date]:
    """Return checked-missing dates that apply as exemptions for this artifact.

    Option-related artifacts honor vendor-confirmed missing dates. All other
    artifact types ignore the kwarg for exemption purposes.

    Args:
        ctx: Active certification context.

    Returns:
        Checked-missing dates when ``artifact_type`` is option-related; else ``[]``.
    """
    if ctx.artifact_type.value in OPTION_ARTIFACT_TYPES:
        return list(ctx.checked_missing_dates)
    return []


@dataclass
class CertificationContext:
    """Mutable state shared across certification check leaves and responders.

    Attributes:
        level: Active certification level (L1 log, L2 raise, L3 fix).
        result: Source manager result being certified.
        start_date: Requested window start (may be sync'd for options).
        end_date: Requested window end (may be sync'd for options).
        checked_missing_dates: Vendor-confirmed missing business dates.
        key: Cache key from the result.
        key_dict: Parsed cache key fields.
        artifact_type: Artifact enum parsed from the key.
        opttick: Option tick when key encodes a contract.
        data: Copy of ``result.timeseries`` used for checks and L3 fixes.
        messages: Residual issue messages (L1 log, L3 unfixed).
        issues_fixed: Human-readable log of L3 repairs applied.
    """

    level: CertificationLevel
    result: Result
    start_date: DATE_HINT
    end_date: DATE_HINT
    checked_missing_dates: List[date]
    key: Optional[str]
    key_dict: Dict[str, str]
    artifact_type: ArtifactType
    opttick: Optional[str] = None
    data: Optional[Union[pd.Series, pd.DataFrame]] = None
    messages: List[str] = field(default_factory=list)
    issues_fixed: List[str] = field(default_factory=list)


def prepare_checked_missing_dates(
    checked_missing_dates: Optional[List[DATE_HINT]],
) -> List[date]:
    """Normalize checked-missing inputs to ``datetime.date`` for set comparisons.

    Args:
        checked_missing_dates: Raw date hints from the caller or cache metadata.

    Returns:
        Normalized business dates. Empty when input is ``None``.
    """
    if checked_missing_dates is None:
        return []
    return [to_datetime(d).date() for d in checked_missing_dates]


def _resolve_endpoint_source(result: Result, key_dict: Dict[str, str], opttick: Optional[str]) -> OptionSpotEndpointSource:
    """Resolve option spot endpoint source from the result or cache key.

    Args:
        result: Manager result that may carry ``endpoint_source``.
        key_dict: Parsed cache key fields.
        opttick: Option tick for error messages.

    Returns:
        Resolved ``OptionSpotEndpointSource``.

    Raises:
        DataCertificationMissingInformationError: When neither result nor key provides a source.
    """
    if result.endpoint_source is not None:
        return result.endpoint_source

    logger.warning(
        "Endpoint source missing on result for opttick=%s key=%s; reading from cache key.",
        opttick,
        result.key,
    )
    raw_source = key_dict.get("endpoint_source")
    if raw_source is None:
        raise DataCertificationMissingInformationError(
            f"Endpoint source is missing for opttick: {opttick} and key: {result.key}"
        )
    return OptionSpotEndpointSource(raw_source)


def setup_context(
    result: Result,
    start_date: DATE_HINT,
    end_date: DATE_HINT,
    level: CertificationLevel,
    checked_missing_dates: Optional[List[DATE_HINT]] = None,
) -> CertificationContext:
    """Build certification context from a manager result and requested date window.

    Args:
        result: Manager result containing timeseries and cache key metadata.
        start_date: Requested certification window start.
        end_date: Requested certification window end.
        level: Certification level to apply in responders.
        checked_missing_dates: Optional vendor-confirmed missing business dates.

    Returns:
        Populated ``CertificationContext`` ready for check leaves.

    Raises:
        DataCertificationMissingInformationError: When cache key lacks ``artifact_type``
            or option contract keys lack endpoint source.
    """
    normalized_missing = prepare_checked_missing_dates(checked_missing_dates)
    key = result.key
    key_dict = _parse_cache_key(key) if key is not None else {}

    raw_artifact = key_dict.get("artifact_type")
    if raw_artifact is None:
        raise DataCertificationMissingInformationError(f"Artifact type is missing for key: {key}")
    artifact_type = ArtifactType(raw_artifact)

    data = result.timeseries.copy() if result.timeseries is not None else None
    opttick: Optional[str] = None

    ## Option contract keys: derive opttick and clamp the requested window to listing bounds.
    if all([key_dict.get("strike"), key_dict.get("expiration"), key_dict.get("right")]):
        opttick = generate_option_tick_new(
            key_dict.get("symbol"),
            key_dict.get("right"),
            key_dict.get("expiration"),
            float(key_dict.get("strike")),
        )
        endpoint_source = _resolve_endpoint_source(result, key_dict, opttick)
        start_date, end_date = _sync_date(
            symbol=key_dict.get("symbol"),
            start_date=start_date,
            end_date=end_date,
            expiration=key_dict.get("expiration"),
            right=key_dict.get("right"),
            strike=float(key_dict.get("strike")),
            endpoint_source=endpoint_source,
        )

    return CertificationContext(
        level=level,
        result=result,
        start_date=start_date,
        end_date=end_date,
        checked_missing_dates=normalized_missing,
        key=key,
        key_dict=key_dict,
        artifact_type=artifact_type,
        opttick=opttick,
        data=data,
    )
