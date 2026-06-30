"""Data certification pipeline for datamanager timeseries results.

Structural audits run through detect-only check leaves and level-specific responders.
L1 logs issues; L2 raises; L3 fixes, re-verifies, and writes data back to the result.

Core Modules:
    pipeline: ``certify``, ``l1_certification``, ``l2_certification``, ``l3_certification``
    checks: Structural check leaves
    respond: L1/L2/L3 response dispatch
    fixers: L3 repair functions
    context: ``setup_context`` and ``CertificationContext``
    types: ``CertificationResult``, ``CertificationReturn``
    start: ``DataCertificationManager`` singleton

Usage:
    >>> from trade.datamanager.certification import l3_certification
    >>> out = l3_certification(result, "2026-01-01", "2026-01-31")
    >>> out.certification_result.issues_fixed
    []
"""

from trade.datamanager.certification.checks import CERTIFICATION_CHECKS
from trade.datamanager.certification.context import (
    CertificationContext,
    effective_checked_missing_dates,
    setup_context,
)
from trade.datamanager.certification.exceptions import (
    DataCertificationError,
    DataCertificationMissingInformationError,
    DataNotCertifiedException,
)
from trade.datamanager.certification.findings import CertificationFinding, IssueCode
from trade.datamanager.certification.fixers import apply_l3_fix
from trade.datamanager.certification.pipeline import (
    certify,
    l1_certification,
    l2_certification,
    l3_certification,
)
from trade.datamanager.certification.report import draft_certificate_report
from trade.datamanager.certification.integration import certify_manager_result
from trade.datamanager.certification.start import DataCertificationManager
from trade.datamanager.certification.types import CertificationResult, CertificationReturn

__all__ = [
    "CERTIFICATION_CHECKS",
    "CertificationContext",
    "CertificationFinding",
    "CertificationResult",
    "CertificationReturn",
    "DataCertificationError",
    "certify_manager_result",
    "DataCertificationManager",
    "DataCertificationMissingInformationError",
    "DataNotCertifiedException",
    "IssueCode",
    "apply_l3_fix",
    "certify",
    "draft_certificate_report",
    "effective_checked_missing_dates",
    "l1_certification",
    "l2_certification",
    "l3_certification",
    "setup_context",
]
