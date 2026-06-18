"""Exceptions raised by the datamanager certification pipeline."""


class DataNotCertifiedException(Exception):
    """Raised at L2 when structural certification checks fail."""


class DataCertificationError(Exception):
    """Base class for certification configuration and preflight failures."""


class DataCertificationMissingInformationError(DataCertificationError):
    """Raised when required cache key or result metadata is missing before checks run."""
