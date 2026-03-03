class DataManagerException(Exception):
    """Base exception for DataManager errors."""
    pass

class EmptyDataException(DataManagerException):
    """Exception raised when data is empty."""
    pass