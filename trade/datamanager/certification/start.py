"""Singleton manager entrypoint for datamanager certification.

Dispatches certification runs using the global ``OptionDataConfig`` default level
unless a per-call level override is supplied. Plain-text audit reports log to
``trade.datamanager.certification.report`` only (not ``certification.start``).

Core Classes:
    DataCertificationManager: Singleton facade over ``pipeline.certify``.
"""

from __future__ import annotations

from typing import List, Optional

from trade.datamanager._enums import CertificationLevel
from trade.datamanager.config import OptionDataConfig
from trade.datamanager.result import Result
from trade.datamanager.utils.logging import CERTIFICATION_REPORT_LOGGER_NAME, get_logging_level
from trade.helpers.helper_types import DATE_HINT, SingletonMetaClass
from trade.helpers.Logging import setup_logger

from .pipeline import certify
from .types import CertificationReturn
from .report import draft_certificate_report

report_logger = setup_logger(
    CERTIFICATION_REPORT_LOGGER_NAME,
    stream_log_level=get_logging_level(),
)


class DataCertificationManager(metaclass=SingletonMetaClass):
    """Singleton facade that certifies manager results at a configured level."""

    def __init__(self) -> None:
        """Initialize manager with the current ``OptionDataConfig`` certification level."""

    def certify_result(
        self,
        result: Result,
        start_date: DATE_HINT,
        end_date: DATE_HINT,
        *,
        checked_missing_dates: Optional[List[DATE_HINT]] = None,
        level: Optional[CertificationLevel] = None,
    ) -> CertificationReturn:
        """Certify a manager result using the configured or overridden level.

        Args:
            result: Manager result to certify.
            start_date: Requested certification window start.
            end_date: Requested certification window end.
            checked_missing_dates: Optional vendor-confirmed missing business dates.
            level: Optional per-call level override.

        Returns:
            ``CertificationReturn`` from the certification pipeline.
        """
        active_level = level or OptionDataConfig().certification_level

        cert_return = certify(
            result,
            start_date,
            end_date,
            level=active_level,
            checked_missing_dates=checked_missing_dates,
        )
        report = draft_certificate_report(cert_return.certification_result)
        report_logger.info("Certification Report:\n%s", report)
        return cert_return
