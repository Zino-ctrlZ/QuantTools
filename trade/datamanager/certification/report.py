"""Human-readable certification report formatting.

Core Functions:
    draft_certificate_report: Build a text summary from ``CertificationResult``.
"""

from __future__ import annotations

from .types import CertificationResult


def draft_certificate_report(certification_result: CertificationResult) -> str:
    """Build a plain-text certification report for logging or display.

    Args:
        certification_result: Completed certification summary.

    Returns:
        Multi-line report string with outcome, key metadata, and issue messages.
    """
    report = f"Certification Result: {certification_result.success}\n"
    report += f"Level: {certification_result.level.name}\n"
    report += f"Date range: {certification_result.start_date} to {certification_result.end_date}\n"
    report += f"Artifact: {certification_result.artifact_type.value}\n"
    report += f"Opttick: {certification_result.opttick}\n"
    report += f"Key: {certification_result.key}\n"
    if certification_result.issues_fixed:
        report += "Issues Fixed:\n"
        for fixed in certification_result.issues_fixed:
            report += f"  - {fixed}\n"
    report += "Messages:\n"
    for message in certification_result.message:
        report += f"  - {message}\n"
    report += f"Certification Result: {certification_result.success}\n"
    return report


## Backward-compatible alias.
_draft_certificate_report = draft_certificate_report
