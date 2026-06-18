"""Result types returned by the certification pipeline.

Core Dataclasses:
    CertificationResult: Summary of a certification run.
    CertificationReturn: Certified payload plus the original manager result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from trade.datamanager._enums import ArtifactType, CertificationLevel
from trade.datamanager.result import Result


@dataclass(frozen=True)
class CertificationResult:
    """Summary of a certification run at a given level.

    Attributes:
        level: Certification level that was applied.
        artifact_type: Artifact parsed from the cache key.
        success: True when no structural issues were recorded.
        message: Residual issue messages after certification.
        issues_fixed: L3 repairs that were applied (empty for L1/L2).
        opttick: Option tick string when the key describes an option contract.
        key: Cache key associated with the certified timeseries.
        start_date: Certification window start (YYYY-MM-DD).
        end_date: Certification window end (YYYY-MM-DD).
    """

    level: CertificationLevel
    artifact_type: ArtifactType
    success: bool
    message: List[str]
    issues_fixed: List[str] = field(default_factory=list)
    opttick: Optional[str] = None
    key: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@dataclass(frozen=True)
class CertificationReturn:
    """Certification output bundled with the unchanged manager result.

    Attributes:
        certification_result: Audit summary from the certification pipeline.
        result: Manager ``Result``; timeseries may be updated at L3.
    """

    certification_result: CertificationResult
    result: Optional[Result] = None


## Backward-compatible alias used in notebooks and early integrations.
_CertificationReturn = CertificationReturn
