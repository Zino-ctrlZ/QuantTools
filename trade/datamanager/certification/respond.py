"""Level-specific responses to certification findings.

Check leaves detect issues; this module decides whether to log (L1), raise (L2),
or fix (L3) for each finding.

Comment density: orchestration

Processing Flow:
    1. ``respond_to_finding`` dispatches to L1/L2/L3 handler by ``ctx.level``.
    2. L1 appends to ``ctx.messages``; L2 raises immediately; L3 mutates ``ctx.data``.
    3. L3 fix pass records repairs in ``ctx.issues_fixed`` (not ``messages``).
    4. ``collect_residual_findings`` re-runs checks after L3 fixes for verify pass.
    5. ``should_short_circuit`` stops the check loop on empty data for L1/L2 only.

Core Functions:
    respond_to_finding: Dispatch a finding to the handler for the active level.
    collect_residual_findings: Re-run checks after L3 fixes for unfixed issues.
"""

from __future__ import annotations

from typing import List

from trade.datamanager._enums import CertificationLevel
from trade.helpers.Logging import setup_logger
from trade.datamanager.utils.logging import get_logging_level

from .checks import CERTIFICATION_CHECKS
from .context import CertificationContext
from .exceptions import DataNotCertifiedException
from .findings import CertificationFinding, IssueCode
from .fixers import apply_l3_fix

logger = setup_logger("trade.datamanager.certification.respond", stream_log_level=get_logging_level())


def _respond_l1(ctx: CertificationContext, finding: CertificationFinding) -> None:
    """Log and record a finding without mutating the result (L1)."""
    logger.warning(finding.message)
    ## L1 never touches ctx.data — audit only; caller returns result.timeseries unchanged.
    ctx.messages.append(finding.message)


def _respond_l2(ctx: CertificationContext, finding: CertificationFinding) -> None:
    """Raise when a structural issue is detected (L2)."""
    ## Fail fast on first finding; pipeline does not accumulate messages at L2.
    raise DataNotCertifiedException(finding.message)


def _respond_l3(ctx: CertificationContext, finding: CertificationFinding) -> None:
    """Apply an L3 fix and record what was repaired."""
    if finding.code == IssueCode.EMPTY:
        ## Empty cannot be synthesized; record now — pipeline clears messages before residual pass.
        logger.warning(
            "L3: data is empty for opttick=%s key=%s; cannot fix.",
            ctx.opttick,
            ctx.key,
        )
        ctx.messages.append(finding.message)
        return

    ## Fix pass mutates ctx.data in place; issues_fixed is the fix audit trail.
    apply_l3_fix(ctx, finding)
    fixed_msg = f"Fixed {finding.code.value}: {finding.message}"
    ctx.issues_fixed.append(fixed_msg)
    logger.info("L3: %s", fixed_msg)


def respond_to_finding(ctx: CertificationContext, finding: CertificationFinding) -> None:
    """Apply level-specific behavior for a single certification finding.

    Args:
        ctx: Active certification context carrying the target level.
        finding: Detected structural issue from a check leaf.

    Raises:
        DataNotCertifiedException: At L2 when any finding is present, or when L3
            fix cannot be applied.
    """
    ## --- Level dispatch: detection is shared; behavior is level-specific ---
    if ctx.level == CertificationLevel.L1:
        _respond_l1(ctx, finding)
        return
    if ctx.level == CertificationLevel.L2:
        _respond_l2(ctx, finding)
        return
    if ctx.level == CertificationLevel.L3:
        _respond_l3(ctx, finding)
        return

    raise ValueError(f"Unsupported certification level: {ctx.level}")


def collect_residual_findings(ctx: CertificationContext) -> List[CertificationFinding]:
    """Re-run all check leaves and return findings that remain after L3 fixes.

    Args:
        ctx: Context with mutated ``data`` after L3 fix pass.

    Returns:
        Findings that still fail after fixes were applied.
    """
    ## Detect-only re-run — no respond_to_finding; pipeline maps residuals to messages.
    residual: List[CertificationFinding] = []
    for check in CERTIFICATION_CHECKS:
        finding = check(ctx)
        if finding is not None:
            residual.append(finding)
    return residual


def log_clean_calendar(ctx: CertificationContext) -> None:
    """Log informational message when no calendar gaps remain after checks.

    Args:
        ctx: Context after missing-calendar check passed.
    """
    if ctx.messages:
        return
    ## Informational only — emitted when calendar check passed and no issues were recorded.
    logger.info(
        "Data has no missing dates for opttick: %s and key: %s",
        ctx.opttick,
        ctx.key,
    )


def should_short_circuit(ctx: CertificationContext, finding: CertificationFinding) -> bool:
    """Return True when later checks should be skipped after this finding.

    Args:
        ctx: Active certification context.
        finding: Finding that was just handled.

    Returns:
        True for empty data on L1/L2. L3 continues so residual pass can run.
    """
    if finding.code != IssueCode.EMPTY:
        return False
    ## L3 must not short-circuit on empty — residual pass still needs to record failure.
    ## L1/L2 stop after empty because later checks assume non-empty DatetimeIndex data.
    return ctx.level in (CertificationLevel.L1, CertificationLevel.L2)
