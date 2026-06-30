"""Certification pipeline orchestration.

Runs setup → check leaves → level-specific responders, then builds
``CertificationReturn``. ``l1_certification``, ``l2_certification``, and
``l3_certification`` are thin wrappers around ``certify``.

Comment density: orchestration

Processing Flow:
    1. ``setup_context`` parses key metadata and syncs option date windows.
    2. Each check in ``CERTIFICATION_CHECKS`` returns an optional finding.
    3. ``respond_to_finding`` logs (L1), raises (L2), or fixes (L3) per finding.
    4. L3 re-runs checks, records residual issues, writes fixed data to ``result``.
    5. Pipeline returns ``CertificationReturn`` with a ``CertificationResult`` summary.

Core Functions:
    certify: Main entrypoint parameterized by ``CertificationLevel``.
    l1_certification: L1 wrapper (audit and log only).
    l2_certification: L2 wrapper (raise on violations).
    l3_certification: L3 wrapper (fix, verify, write back).
"""

from __future__ import annotations

from typing import List, Optional

from trade.datamanager._enums import CertificationLevel
from trade.datamanager.result import Result
from trade.helpers.helper_types import DATE_HINT
from trade.helpers.Logging import setup_logger
from trade.datamanager.utils.logging import get_logging_level

from .checks import CERTIFICATION_CHECKS, check_missing_calendar_days, format_certification_date_range
from .context import CertificationContext, setup_context
from .respond import (
    collect_residual_findings,
    log_clean_calendar,
    respond_to_finding,
    should_short_circuit,
)
from .types import CertificationResult, CertificationReturn

logger = setup_logger("trade.datamanager.certification.pipeline", stream_log_level=get_logging_level())


def _execute_checks(ctx: CertificationContext, *, respond: bool) -> bool:
    """Run check leaves and optionally dispatch level-specific responses.

    Args:
        ctx: Active certification context.
        respond: When True, call ``respond_to_finding`` for each finding.

    Returns:
        True when the missing-calendar check ran and found no gap.
    """
    calendar_check_ran = False
    for check in CERTIFICATION_CHECKS:
        finding = check(ctx)
        if finding is None:
            if check is check_missing_calendar_days:
                calendar_check_ran = True
            continue

        if respond:
            respond_to_finding(ctx, finding)
            if should_short_circuit(ctx, finding):
                break
        else:
            ctx.messages.append(finding.message)

    return calendar_check_ran


def _write_back_l3_data(ctx: CertificationContext) -> None:
    """Persist L3-repaired timeseries onto the manager result."""
    if ctx.data is not None:
        ctx.result.timeseries = ctx.data


def certify(
    result: Result,
    start_date: DATE_HINT,
    end_date: DATE_HINT,
    *,
    level: CertificationLevel = CertificationLevel.L1,
    checked_missing_dates: Optional[List[DATE_HINT]] = None,
) -> CertificationReturn:
    """Run structural certification at the requested level.

    Args:
        result: Manager result with timeseries and cache key metadata.
        start_date: Requested certification window start.
        end_date: Requested certification window end.
        level: Certification level (L1 log, L2 raise, L3 fix and verify).
        checked_missing_dates: Vendor-confirmed missing business dates for exemptions.

    Returns:
        ``CertificationReturn`` with audit summary; L3 may update ``result.timeseries``.

    Raises:
        DataCertificationMissingInformationError: When required key metadata is absent.
        DataNotCertifiedException: At L2 when any structural issue is detected.
    """
    ## --- Step 1: build shared context (key, artifact, opttick, synced date window) ---
    ctx = setup_context(
        result=result,
        start_date=start_date,
        end_date=end_date,
        level=level,
        checked_missing_dates=checked_missing_dates,
    )
    logger.info(
        "Starting %s certification for result: %s (date range: %s to %s)",
        level.name,
        result,
        ctx.start_date,
        ctx.end_date,
    )

    ## --- Step 2: check + respond ---
    if level == CertificationLevel.L3:
        ## Pass A: apply fixes in issue order; empty data is recorded but not repairable.
        _execute_checks(ctx, respond=True)

        ## Pass B: re-run checks on repaired data; anything left is a residual failure.
        ctx.messages.clear()
        for finding in collect_residual_findings(ctx):
            ctx.messages.append(finding.message)

        _write_back_l3_data(ctx)
        if not ctx.messages:
            log_clean_calendar(ctx)
    else:
        calendar_check_ran = _execute_checks(ctx, respond=True)
        if calendar_check_ran and not ctx.messages:
            log_clean_calendar(ctx)

    ## --- Step 3: package outcome ---
    success = not ctx.messages
    if success:
        logger.info("Certification successful for opttick: %s and key: %s", ctx.opttick, ctx.key)

    start_str, end_str = format_certification_date_range(ctx.start_date, ctx.end_date)
    certification_result = CertificationResult(
        level=level,
        artifact_type=ctx.artifact_type,
        success=success,
        message=list(ctx.messages),
        issues_fixed=list(ctx.issues_fixed),
        opttick=ctx.opttick,
        key=ctx.key,
        start_date=start_str,
        end_date=end_str,
    )
    return CertificationReturn(certification_result=certification_result, result=ctx.result)


def l1_certification(
    result: Result,
    start_date: DATE_HINT,
    end_date: DATE_HINT,
    checked_missing_dates: Optional[List[DATE_HINT]] = None,
) -> CertificationReturn:
    """Run L1 certification: log structural issues and return data as-is."""
    return certify(
        result,
        start_date,
        end_date,
        level=CertificationLevel.L1,
        checked_missing_dates=checked_missing_dates,
    )


def l2_certification(
    result: Result,
    start_date: DATE_HINT,
    end_date: DATE_HINT,
    checked_missing_dates: Optional[List[DATE_HINT]] = None,
) -> CertificationReturn:
    """Run L2 certification: raise on the first structural violation."""
    return certify(
        result,
        start_date,
        end_date,
        level=CertificationLevel.L2,
        checked_missing_dates=checked_missing_dates,
    )


def l3_certification(
    result: Result,
    start_date: DATE_HINT,
    end_date: DATE_HINT,
    checked_missing_dates: Optional[List[DATE_HINT]] = None,
) -> CertificationReturn:
    """Run L3 certification: fix structural issues, verify, and write back data."""
    return certify(
        result,
        start_date,
        end_date,
        level=CertificationLevel.L3,
        checked_missing_dates=checked_missing_dates,
    )
