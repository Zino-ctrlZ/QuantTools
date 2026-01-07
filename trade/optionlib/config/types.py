from enum import Enum

class VolSide(str, Enum):
    """
    Enum for volatility sides.
    """
    CALL = 'call'
    PUT = 'put'
    OTM = 'otm'

class TimeseriesVolType(str, Enum):
    """
    Enum for volatility types in time series context.
    This determines how volatilities are computed for end user consumption.
    """
    BS = 'bs'
    BINOMIAL = 'binomial'
    MODEL_DYNAMICS = 'model_dynamics'

class VolType(str, Enum):
    """
    Enum for volatility types used in calibration.
    Strictly for calibration of vol surface purposes.
    """
    BS = 'bs'
    BINOMIAL = 'binomial'

class DivType(str, Enum):
    """
    Enum for dividend types.
    """
    DISCRETE = 'discrete'
    CONTINUOUS = 'continuous'

class DiscreteDivGrowthModel(str, Enum):
    """
    Enum for discrete dividend models.
    These models define how discrete dividends grow over time.
    """
    CAGR = 'cagr'
    REGRESSION = 'regression'
    AVG = 'avg'
    REGRESSION_CAGR = 'regression_cagr'