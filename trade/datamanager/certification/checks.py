"""Structural certification check leaves (detect-only).

Each function inspects ``CertificationContext`` and returns a ``CertificationFinding``
when an issue exists. Level-specific behavior is applied in ``respond.py``.

Comment density: domain policy

Processing Flow:
    1. ``check_empty`` — no data means later structural checks are meaningless.
    2. ``check_invalid_index`` — calendar and NaN audits require ``DatetimeIndex``.
    3. ``check_duplicate_index`` — dupes are audited on the full frame (not deduped first).
    4. ``check_unexplained_nans`` — NaNs on checked-missing rows are exempt (options only).
    5. ``check_missing_calendar_days`` — B-day gaps in ``[start, end]`` minus exemptions and
       dates not yet available (pre-market today via ``is_available_on_date``).

Core Functions:
    check_empty: Detect missing timeseries payload.
    check_invalid_index: Detect non-DatetimeIndex.
    check_duplicate_index: Detect duplicate index labels.
    check_unexplained_nans: Detect NaNs outside checked-missing dates.
    check_missing_calendar_days: Detect missing business days in the window.
    certification_required_bus_days: B-days cert must cover (shared with L3 grid).
    CERTIFICATION_CHECKS: Ordered list of check callables for the pipeline.
"""

from __future__ import annotations

from datetime import date
from typing import Callable, List, Optional, Union

import pandas as pd

from trade import HOLIDAY_SET
from trade.datamanager.utils.date import is_available_on_date
from trade.helpers.helper import to_datetime
from trade.helpers.helper_types import DATE_HINT

from .context import CertificationContext, effective_checked_missing_dates
from .findings import CertificationFinding, IssueCode

PandasData = Union[pd.Series, pd.DataFrame]
CheckFn = Callable[[CertificationContext], Optional[CertificationFinding]]


def _is_empty(data: Optional[PandasData]) -> bool:
    """Return True when timeseries data is None or has no rows."""
    return data is None or data.empty


def _is_valid_index(data: PandasData) -> bool:
    """Return True when data uses a DatetimeIndex."""
    return isinstance(data.index, pd.DatetimeIndex)


def _filter_checked_missing_dates(
    data: PandasData,
    checked_missing_dates: List[date],
) -> PandasData:
    """Return data with checked-missing index rows removed for value audits."""
    if not checked_missing_dates:
        return data
    ## Vendor-confirmed absent dates may carry NaN; exclude them before NaN audit.
    ## normalize() aligns Timestamp vs date comparisons from cache metadata.
    return data.loc[~data.index.normalize().isin(checked_missing_dates)]


def certification_required_bus_days(
    start_date: DATE_HINT,
    end_date: DATE_HINT,
    checked_missing_dates: Optional[List[date]] = None,
) -> List[date]:
    """Return business days in-window that certification must cover.

    Excludes exchange holidays, vendor checked-missing dates (options), and dates
    not yet available — notably **today before market open** when ``end_date``
    includes today. During market hours, today remains required.

    Args:
        start_date: Certification window start.
        end_date: Certification window end.
        checked_missing_dates: Vendor-confirmed absent dates to omit from the grid.

    Returns:
        Sorted list of dates the certifier treats as must-have observations.
    """
    start_d = pd.Timestamp(to_datetime(start_date)).normalize()
    end_d = pd.Timestamp(to_datetime(end_date)).normalize()
    checked_set = set(checked_missing_dates or [])

    required: List[date] = []
    for ts in pd.date_range(start=start_d, end=end_d, freq="B"):
        day = ts.date()
        if ts.strftime("%Y-%m-%d") in HOLIDAY_SET:
            continue
        if day in checked_set:
            continue
        ## Pre-market today is not required; during market hours today in-range is required.
        if not is_available_on_date(day):
            continue
        required.append(day)
    return required


def _calendar_gaps_excluding_checked_missing(
    data: PandasData,
    start_date: DATE_HINT,
    end_date: DATE_HINT,
    checked_missing_dates: List[date],
) -> List[date]:
    """Return required bus days absent from data index."""
    required = certification_required_bus_days(start_date, end_date, checked_missing_dates)
    if not required:
        return []
    index_dates = {to_datetime(ts).date() for ts in data.index.normalize()}
    return [day for day in required if day not in index_dates]


def _issue_message(ctx: CertificationContext, summary: str) -> str:
    """Format a consistent issue message with opttick and key context."""
    return f"{summary} for opttick: {ctx.opttick} and key: {ctx.key}"


def duplicate_index_labels(data: Optional[PandasData]) -> List[str]:
    """Return sorted YYYY-MM-DD labels that appear more than once on the index.

    Args:
        data: Timeseries payload under audit.

    Returns:
        Unique duplicate index dates as strings; empty when none.
    """
    if data is None or data.empty:
        return []
    dup_mask = data.index.duplicated(keep=False)
    if not dup_mask.any():
        return []
    labels = pd.to_datetime(data.index[dup_mask]).normalize().unique()
    return sorted(ts.strftime("%Y-%m-%d") for ts in labels)


def format_certification_date_range(start_date: DATE_HINT, end_date: DATE_HINT) -> tuple[str, str]:
    """Normalize certification window bounds to ``YYYY-MM-DD`` strings.

    Args:
        start_date: Requested window start.
        end_date: Requested window end.

    Returns:
        ``(start_str, end_str)`` for reports and ``CertificationResult``.
    """
    start_str = pd.Timestamp(to_datetime(start_date)).strftime("%Y-%m-%d")
    end_str = pd.Timestamp(to_datetime(end_date)).strftime("%Y-%m-%d")
    return start_str, end_str


def check_empty(ctx: CertificationContext) -> Optional[CertificationFinding]:
    """Detect empty or missing timeseries data.

    Args:
        ctx: Active certification context.

    Returns:
        Finding when data is empty; otherwise ``None``.
    """
    if not _is_empty(ctx.data):
        return None
    return CertificationFinding(
        code=IssueCode.EMPTY,
        message=_issue_message(ctx, "Data is empty"),
    )


def check_invalid_index(ctx: CertificationContext) -> Optional[CertificationFinding]:
    """Detect timeseries that do not use a DatetimeIndex.

    Args:
        ctx: Active certification context.

    Returns:
        Finding when index type is invalid; otherwise ``None``.
    """
    if _is_empty(ctx.data) or _is_valid_index(ctx.data):
        return None
    ## Downstream calendar/NaN checks require DatetimeIndex; flag before those run.
    return CertificationFinding(
        code=IssueCode.INVALID_INDEX,
        message=_issue_message(ctx, "Data index is not a DatetimeIndex"),
    )


def check_duplicate_index(ctx: CertificationContext) -> Optional[CertificationFinding]:
    """Detect duplicate datetime index labels.

    Args:
        ctx: Active certification context.

    Returns:
        Finding when duplicate indices exist; otherwise ``None``.
    """
    if _is_empty(ctx.data):
        return None
    ## Audit full frame including duplicate rows — do not dedupe before this check.
    duplicate_labels = duplicate_index_labels(ctx.data)
    if not duplicate_labels:
        return None
    return CertificationFinding(
        code=IssueCode.DUPLICATE_INDEX,
        message=_issue_message(ctx, f"Data has duplicate indices: {duplicate_labels}"),
        details={"duplicate_indices": duplicate_labels},
    )


def check_unexplained_nans(ctx: CertificationContext) -> Optional[CertificationFinding]:
    """Detect NaN values on rows that are not vendor checked-missing dates.

    Args:
        ctx: Active certification context.

    Returns:
        Finding when unexplained NaNs exist; otherwise ``None``.
    """
    if _is_empty(ctx.data) or not _is_valid_index(ctx.data):
        return None

    ## Only audit rows that are not vendor-exempt; non-option artifacts get [] exemptions.
    audited = _filter_checked_missing_dates(ctx.data, effective_checked_missing_dates(ctx))
    ## any().any() handles both Series and DataFrame without branching on shape.
    if not audited.isna().any().any():
        return None
    return CertificationFinding(
        code=IssueCode.UNEXPLAINED_NAN,
        message=_issue_message(ctx, "Data has NaN values"),
    )


def check_missing_calendar_days(ctx: CertificationContext) -> Optional[CertificationFinding]:
    """Detect missing business days in the requested certification window.

    Args:
        ctx: Active certification context.

    Returns:
        Finding when unexplained calendar gaps exist; otherwise ``None``.
    """
    if _is_empty(ctx.data) or not _is_valid_index(ctx.data):
        return None

    ## Window may have been sync'd for options in setup_context; exemptions are artifact-gated.
    missing_dates = _calendar_gaps_excluding_checked_missing(
        ctx.data,
        ctx.start_date,
        ctx.end_date,
        effective_checked_missing_dates(ctx),
    )
    if not missing_dates:
        return None
    return CertificationFinding(
        code=IssueCode.MISSING_CALENDAR_DAY,
        message=_issue_message(ctx, f"Data has missing dates: {missing_dates}"),
        details={"missing_dates": missing_dates},
    )


CERTIFICATION_CHECKS: List[CheckFn] = [
    check_empty,
    check_invalid_index,
    check_duplicate_index,
    check_unexplained_nans,
    check_missing_calendar_days,
]
## Order matters: empty/invalid index short-circuit in pipeline via should_short_circuit (L1/L2).
## L3 runs the full list in fix pass, then re-runs all checks for residuals.
