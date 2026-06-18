"""Return-boundary certification wiring for datamanager managers.

Managers call these helpers after sanitize and before returning to callers.
Logs NA snapshots on the pre-certify payload, then runs certification.
Sets ``Result.is_certified`` when the certifier completes (distinct from
``CertificationResult.success``).

Comment density: orchestration

Processing Flow:
    1. Attach cache key when missing.
    2. ``log_retrieval_na`` on post-sanitize, pre-certify ``result`` (forensics before L3 fixes).
    3. ``DataCertificationManager.certify_result`` — structural audit / fix / raise.
    4. Mark ``is_certified`` on successful completion.

Core Functions:
    certify_manager_result: Log NAs, certify, and mark the result as seen by certifier.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TypeVar

from trade.datamanager._enums import ArtifactType, CertificationLevel
from trade.datamanager.result import Result
from trade.datamanager.utils.enums_utils import _parse_cache_key
from trade.datamanager.utils.na_logging import log_retrieval_na
from trade.helpers.helper_types import DATE_HINT

from .start import DataCertificationManager

TResult = TypeVar("TResult", bound=Result)

## Cache-key artifact_type → na_logging manager label (matches @log_na_after_retrieval names).
_ARTIFACT_MANAGER_LABELS: Dict[str, str] = {
    ArtifactType.SPOT.value: "spot",
    ArtifactType.RATES.value: "rates",
    ArtifactType.FWD.value: "forward",
    ArtifactType.DIVS.value: "dividend",
    ArtifactType.OPTION_SPOT.value: "option_spot",
    ArtifactType.IV.value: "vol",
    ArtifactType.GREEKS.value: "greeks",
    ArtifactType.CHAIN.value: "market_timeseries",
}


def _resolve_na_log_labels(
    result: Result,
    *,
    manager: Optional[str],
    method: Optional[str],
) -> tuple[str, str]:
    """Derive na_logging manager/method labels from explicit args or cache key metadata."""
    key_dict = _parse_cache_key(result.key) if result.key is not None else {}
    artifact = key_dict.get("artifact_type")
    resolved_manager = manager or _ARTIFACT_MANAGER_LABELS.get(artifact, artifact or "datamanager")
    resolved_method = method or key_dict.get("series_id") or "timeseries"
    return resolved_manager, resolved_method


def certify_manager_result(
    result: TResult,
    start_date: DATE_HINT,
    end_date: DATE_HINT,
    *,
    cache_key: Optional[str] = None,
    checked_missing_dates: Optional[List[DATE_HINT]] = None,
    level: Optional[CertificationLevel] = None,
    manager: Optional[str] = None,
    method: Optional[str] = None,
) -> TResult:
    """Certify a manager result at the return boundary and mark it as certified.

    NA snapshots are logged on the post-sanitize payload **before** certification so
    L3 fixes and L2 raises do not hide the raw retrieval state.

    ``is_certified`` indicates the certifier ran to completion for this return.
    It does not imply ``CertificationResult.success`` (L1 may log issues; L3 may
    leave residuals). L2 raises before ``is_certified`` is set (after NA logging).

    Args:
        result: Assembled manager result with ``timeseries`` payload.
        start_date: Certification window start (sync'd for options).
        end_date: Certification window end (sync'd for options).
        cache_key: Attach to ``result.key`` when not yet set (e.g. cache hit).
        checked_missing_dates: Vendor-confirmed missing business dates for exemptions.
        level: Per-call certification level override.
        manager: Optional na_logging manager label (inferred from key when omitted).
        method: Optional na_logging method label (defaults to key ``series_id`` or ``timeseries``).

    Returns:
        Certified result; ``timeseries`` may be updated at L3.

    Raises:
        DataNotCertifiedException: At L2 when structural checks fail.
        DataCertificationMissingInformationError: When required metadata is missing.
    """
    if cache_key is not None and result.key is None:
        result.key = cache_key

    log_manager, log_method = _resolve_na_log_labels(result, manager=manager, method=method)
    log_context: Dict[str, DATE_HINT] = {}
    if getattr(result, "symbol", None) is not None:
        log_context["symbol"] = result.symbol
    log_retrieval_na(result, manager=log_manager, method=log_method, **log_context)

    cert_return = DataCertificationManager().certify_result(
        result,
        start_date,
        end_date,
        checked_missing_dates=checked_missing_dates,
        level=level,
    )
    outcome = cert_return.result if cert_return.result is not None else result
    outcome.is_certified = True
    return outcome  # type: ignore[return-value]
