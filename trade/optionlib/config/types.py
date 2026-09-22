"""Optionlib configuration enums for vol, dividends, and greek bump factors."""

from enum import Enum


class VolSide(str, Enum):
    """Enum for volatility sides."""

    CALL = "call"
    PUT = "put"
    OTM = "otm"


class TimeseriesVolType(str, Enum):
    """Volatility types in time series context for end-user consumption."""

    BS = "bs"
    BINOMIAL = "binomial"
    MODEL_DYNAMICS = "model_dynamics"


class VolType(str, Enum):
    """Volatility types used in surface calibration."""

    BS = "bs"
    BINOMIAL = "binomial"


class DivType(str, Enum):
    """Dividend types."""

    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class DiscreteDivGrowthModel(str, Enum):
    """How discrete dividends grow over time."""

    CAGR = "cagr"
    REGRESSION = "regression"
    AVG = "avg"
    REGRESSION_CAGR = "regression_cagr"
    CONSTANT = "constant"
    CONSTANT_AVG = "constant+avg"
    CONSTANT_CAGR = "constant+cagr"
    CONSTANT_REGRESSION = "constant+regression"


class GreekFactor(str, Enum):
    """Canonical pricing factors that have a greek finite-difference bump policy."""

    SPOT = "S"
    VOL = "sigma"
    RATE = "r"
    TIME = "T"


class GreekBumpMode(str, Enum):
    """How the absolute FD step is derived from config for a greek factor."""

    MULTIPLICATIVE = "multiplicative"
    ADDITIVE = "additive"
    THETA = "theta"
