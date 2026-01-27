from enum import Enum


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
    VOMMA = "vomma"
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


class OptionPricingModel(Enum):
    """Enumeration of option pricing model."""

    BSM = "Black-Scholes"
    BINOMIAL = "Binomial"
    EURO_EQIV = "European Equivalent"


class VolatilityModel(Enum):
    """Enumeration of volatility model."""

    MARKET = "market"
    MODEL_DYNAMIC = "model_dynamic"