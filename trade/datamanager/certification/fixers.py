"""L3 certification fixers for structural timeseries issues.

Applies in-place repairs on ``CertificationContext.data``. Fixers respect
``checked_missing_dates`` only for option-related artifacts; other artifacts
are fully reindexed and forward-filled within the requested window.

Comment density: domain policy

Processing Flow:
    1. ``apply_l3_fix`` dispatches by ``IssueCode`` (empty is a no-op here).
    2. Index repair runs before dedupe/NaN/calendar fixes (check order in pipeline).
    3. NaN and calendar fixes share ``_ffill_preserving_checked_missing``.
    4. Calendar fix builds B-day grid via ``_business_day_grid`` then reindexes.
    5. Pipeline writes ``ctx.data`` back to ``result.timeseries`` after fix pass.

Core Functions:
    apply_l3_fix: Dispatch a finding to the appropriate L3 fixer.
    effective_checked_missing_dates: Resolve exemption dates for the artifact type.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from trade.helpers.helper import to_datetime
from trade.helpers.Logging import setup_logger
from trade.datamanager.utils.logging import get_logging_level

from .checks import certification_required_bus_days
from .context import CertificationContext, effective_checked_missing_dates
from .exceptions import DataNotCertifiedException
from .findings import CertificationFinding, IssueCode

logger = setup_logger("trade.datamanager.certification.fixers", stream_log_level=get_logging_level())

PandasData = Union[pd.Series, pd.DataFrame]


def _require_data(ctx: CertificationContext) -> PandasData:
    """Return context data or raise when L3 cannot fix an empty payload."""
    if ctx.data is None or ctx.data.empty:
        raise DataNotCertifiedException(
            f"L3 cannot fix empty data for opttick: {ctx.opttick} and key: {ctx.key}"
        )
    return ctx.data


def _business_day_grid(ctx: CertificationContext) -> pd.DatetimeIndex:
    """Build the certification B-day grid for ``[start_date, end_date]``.

    Uses ``certification_required_bus_days`` so the L3 reindex grid matches the
    missing-calendar check — including pre-market today exclusion.
    """
    required = certification_required_bus_days(
        ctx.start_date,
        ctx.end_date,
        effective_checked_missing_dates(ctx),
    )
    if not required:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(pd.to_datetime(required))


def _ffill_preserving_checked_missing(ctx: CertificationContext, data: PandasData) -> PandasData:
    """Forward-fill values while keeping NaN on checked-missing dates (options only)."""
    checked_missing = effective_checked_missing_dates(ctx)
    if not checked_missing:
        return data.ffill()

    ## Fill first, then restore NaN on exempt rows — avoids fabricating vendor-absent values.
    filled = data.ffill()
    preserve_mask = filled.index.normalize().isin(checked_missing)
    if preserve_mask.any():
        filled.loc[preserve_mask] = data.loc[preserve_mask]
    return filled


def _fix_invalid_index(ctx: CertificationContext) -> None:
    """Convert the index to ``DatetimeIndex`` using project date parsing."""
    data = _require_data(ctx)
    try:
        ## to_datetime on index iterable returns DatetimeIndex-compatible values.
        data.index = pd.DatetimeIndex(to_datetime(data.index))
        ctx.data = data.sort_index()
    except Exception as exc:
        raise DataNotCertifiedException(
            f"Failed to convert index to datetime for opttick: {ctx.opttick} and key: {ctx.key}."
        ) from exc


def _fix_duplicate_index(ctx: CertificationContext) -> None:
    """Drop duplicate datetime index labels, keeping the last observation."""
    data = _require_data(ctx)
    ## keep="last" matches typical EOD overwrite semantics when cache merges overlap.
    ctx.data = data[~data.index.duplicated(keep="last")].sort_index()


def _fix_unexplained_nans(ctx: CertificationContext) -> None:
    """Forward-fill unexplained NaNs without filling checked-missing rows."""
    data = _require_data(ctx)
    ctx.data = _ffill_preserving_checked_missing(ctx, data)


def _fix_missing_calendar_days(ctx: CertificationContext) -> None:
    """Reindex to the requested B-day grid and forward-fill vendor gaps."""
    data = _require_data(ctx)
    grid = _business_day_grid(ctx)
    ## New rows start as NaN; ffill back-fills from prior observations in-window.
    reindexed = data.reindex(grid, fill_value=np.nan)
    ctx.data = _ffill_preserving_checked_missing(ctx, reindexed)


def apply_l3_fix(ctx: CertificationContext, finding: CertificationFinding) -> None:
    """Apply the L3 fix that corresponds to a certification finding.

    Args:
        ctx: Mutable certification context; ``ctx.data`` is updated in place.
        finding: Detected issue to repair.

    Raises:
        DataNotCertifiedException: When a fix cannot be applied (e.g. empty data).
        ValueError: When the finding code has no registered fixer.
    """
    ## --- Dispatch mirrors CERTIFICATION_CHECKS order from pipeline fix pass ---
    if finding.code == IssueCode.EMPTY:
        ## Empty series cannot be synthesized at L3; caller records as residual issue.
        logger.warning(
            "L3: data is empty for opttick=%s key=%s; no fix applied.",
            ctx.opttick,
            ctx.key,
        )
        return

    if finding.code == IssueCode.INVALID_INDEX:
        logger.warning(
            "L3: converting index to DatetimeIndex for opttick=%s key=%s.",
            ctx.opttick,
            ctx.key,
        )
        _fix_invalid_index(ctx)
        return

    if finding.code == IssueCode.DUPLICATE_INDEX:
        duplicate_labels = finding.details.get("duplicate_indices", [])
        logger.warning(
            "L3: dropping duplicate index labels %s for opttick=%s key=%s.",
            duplicate_labels,
            ctx.opttick,
            ctx.key,
        )
        _fix_duplicate_index(ctx)
        return

    if finding.code == IssueCode.UNEXPLAINED_NAN:
        logger.warning(
            "L3: forward-filling unexplained NaNs for opttick=%s key=%s.",
            ctx.opttick,
            ctx.key,
        )
        _fix_unexplained_nans(ctx)
        return

    if finding.code == IssueCode.MISSING_CALENDAR_DAY:
        logger.warning(
            "L3: reindexing to business-day grid for opttick=%s key=%s.",
            ctx.opttick,
            ctx.key,
        )
        _fix_missing_calendar_days(ctx)
        return

    raise ValueError(f"No L3 fixer registered for issue: {finding.code.value}")
