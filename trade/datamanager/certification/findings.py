"""Certification issue codes and finding payloads for structural data audits.

Findings are produced by check leaves in ``checks.py`` and consumed by level-specific
responders in ``respond.py``. Detection stays pure; level behavior lives in the responder.

Core Dataclasses:
    CertificationFinding: A single structural issue detected during certification.

Core Enums:
    IssueCode: Stable identifiers for certification check outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class IssueCode(str, Enum):
    """Stable identifiers for certification structural issues."""

    EMPTY = "empty"
    INVALID_INDEX = "invalid_index"
    DUPLICATE_INDEX = "duplicate_index"
    UNEXPLAINED_NAN = "unexplained_nan"
    MISSING_CALENDAR_DAY = "missing_calendar_day"


## Option artifacts that may use checked_missing_dates exemptions (future gating).
OPTION_ARTIFACT_TYPES: frozenset[str] = frozenset({"option_spot", "iv", "greeks"})


@dataclass(frozen=True)
class CertificationFinding:
    """A single structural issue detected during certification.

    Attributes:
        code: Machine-readable issue identifier.
        message: Human-readable description for logs and reports.
        details: Optional structured payload (e.g. missing date list).
    """

    code: IssueCode
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
