from enum import Enum

class AttributionModel(Enum):
    """Enumeration for different attribution models."""
    xMULTIPLY = "xmultiply"
    FRV = "frv"
    UNDEFINED = "undefined"