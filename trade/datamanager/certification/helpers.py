"""Backward-compatible re-exports for the certification package.

New code should import from ``trade.datamanager.certification`` or the specific
submodules (``pipeline``, ``checks``, ``types``, etc.).
"""

from __future__ import annotations

from .pipeline import certify, l1_certification, l2_certification, l3_certification
from .report import _draft_certificate_report, draft_certificate_report
from .types import CertificationResult, CertificationReturn, _CertificationReturn

__all__ = [
    "CertificationResult",
    "CertificationReturn",
    "_CertificationReturn",
    "_draft_certificate_report",
    "certify",
    "draft_certificate_report",
    "l1_certification",
    "l2_certification",
    "l3_certification",
]
