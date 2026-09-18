"""Datamanager configuration enums (interval, artifacts, pricing, ThetaData policy).

Re-exports ThetaData ``ListedSessionNotFoundPolicy`` so callers can import it
next to ``OptionDataConfig`` / other datamanager enums.

Core Enums:
    CertificationLevel: L1 log / L2 raise / L3 fix.
    OptionSpotEndpointSource: EOD vs quote tape.
    ListedSessionNotFoundPolicy: Omit or raise when a listed quote session has no tape.
"""

from enum import Enum, IntEnum
from typing import Literal, get_args

from dbase.DataAPI.ThetaData.v3.vars import ListedSessionNotFoundPolicy, ThetaDataV3Controls
from trade.optionlib.config.types import DivType  # noqa

__all__ = [
    "AVAILABLE_GREEKS",
    "ArtifactType",
    "CertificationLevel",
    "GREEKS",
    "GreekType",
    "Interval",
    "ListedSessionNotFoundPolicy",
    "ModelPrice",
    "OptionPricingModel",
    "OptionSpotEndpointSource",
    "RealTimeFallbackOption",
    "SeriesId",
    "ThetaDataV3Controls",
    "VolatilityModel",
]


class CertificationLevel(IntEnum):
    """
    Guarantee level for data certification.

    Levels:
        L1: Return as-is; log structural issues (dupes, gaps, NaNs).
        L2: Raise on unexplained violations.
        L3: Fix issues (ffill, dedupe, resample), then certify.
    """

    L1 = 1
    L2 = 2
    L3 = 3


class Interval(str, Enum):
    INTRADAY = "intraday"  # historical intraday timestamp
    EOD = "eod"  # end-of-day daily snapshot
    NA = "na"  # not applicable

class RealTimeFallbackOption(str, Enum):
    RAISE_ERROR = "raise_error"
    USE_LAST_AVAILABLE = "use_last_available"
    ZEROED = "zeroed"
    NAN = "nan"
class SeriesId(str, Enum):
    HIST = "hist"
    AT_TIME = "at_time"
    SNAPSHOT = "snapshot"

class ArtifactType(str, Enum):
    # Market / inputs
    SPOT = "spot"
    CHAIN = "chain"
    RATES = "rates"
    DIVS = "divs"
    FWD = "forward"
    OPTION_SPOT = "option_spot"
    DATES = "dates"

    # Volatility
    IV = "iv"
    TVAR = "tvar"

    # Greeks
    GREEKS = "greeks"
    DELTA = "delta"
    GAMMA = "gamma"
    VEGA = "vega"
    THETA = "theta"
    VOLGA = "volga"
    VANNA = "vanna"
    RHO = "rho"

class GreekType(str, Enum):
    GREEKS = "greeks"
    DELTA = "delta"
    GAMMA = "gamma"
    VEGA = "vega"
    THETA = "theta"
    VOLGA = "volga"
    VANNA = "vanna"
    RHO = "rho"

class OptionSpotEndpointSource(Enum):
    """
    Thetadata creates a native EOD report every day by 6pm ET.
    This enum allows choosing between using that EOD report or the intraday quote end point.
    This is essential because during market hours, the EOD report is not yet available.
    """

    EOD = "eod"
    QUOTE = "quote"

class ModelPrice(Enum):
    """Enumeration of model price type."""

    MIDPOINT = "midpoint"
    BID = "bid"
    ASK = "ask"
    OPEN = "open"
    CLOSE = "close"



class OptionPricingModel(Enum):
    """Enumeration of option pricing model."""

    BSM = "Black-Scholes"
    BINOMIAL = "Binomial"
    EURO_EQIV = "European Equivalent"


class VolatilityModel(Enum):
    """Enumeration of volatility model."""

    MARKET = "market"
    MODEL_DYNAMIC = "model_dynamic"


GREEKS = Literal[
    GreekType.DELTA.value,
    GreekType.GAMMA.value,
    GreekType.THETA.value,
    GreekType.VEGA.value,
    GreekType.RHO.value,
    GreekType.VOLGA.value,
]
AVAILABLE_GREEKS = get_args(GREEKS)
