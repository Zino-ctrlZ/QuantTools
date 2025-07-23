from . import config, load_config
load_config()
DAILY_BASIS = config['DAILY_BASIS']
DIVIDEND_LOOKBACK_YEARS = config['DIVIDEND_FORECAST_LOOKBACK_YEARS']
DIVIDEND_LOOKFORWARD_YEARS = config['DIVIDEND_FORECAST_LOOKFORWARD_YEARS']
OPTION_TIMESERIES_START_DATE = config['OPTION_TIMESERIES_START_DATE']
DIVIDEND_FORECAST_METHOD = config['DIVIDEND_FORECAST_METHOD']
VOL_EST_UPPER_BOUND= config['VOL_EST_UPPER_BOUND']
VOL_EST_LOWER_BOUND= config['VOL_EST_LOWER_BOUND']

# This class contains the limits on inputs for GBS models
# It is not intended to be part of this module's public interface
class _GBS_Limits:
    # An GBS model will return an error if an out-of-bound input is input
    MAX32 = 2147483248.0

    MIN_T = 1.0 / 1000.0  # requires some time left before expiration
    MIN_X = 0.01
    MIN_FS = 0.01

    # Volatility smaller than 0.5% causes American Options calculations
    # to fail (Number to large errors).
    # GBS() should be OK with any positive number. Since vols less
    # than 0.5% are expected to be extremely rare, and most likely bad inputs,
    # _gbs() is assigned this limit too
    MIN_V = 0.005

    MAX_T = 100
    MAX_X = MAX32
    MAX_FS = MAX32

    # Asian Option limits
    # maximum TA is time to expiration for the option
    MIN_TA = 0

    # This model will work with higher values for b, r, and V. However, such values are extremely uncommon.
    # To catch some common errors, interest rates and volatility is capped to 200%
    # This reason for 2 (200%) is mostly to cause the library to throw an exceptions
    # if a value like 15% is entered as 15 rather than 0.15)
    MIN_b = -1
    MIN_r = -1

    MAX_b = 1
    MAX_r = 2
    MAX_V = 2


# This class defines the Exception that gets thrown when invalid input is placed into the GBS function
class GBS_InputError(Exception):
    def __init__(self, mismatch):
        Exception.__init__(self, mismatch)


# This class defines the Exception that gets thrown when there is a calculation error
class GBS_CalculationError(Exception):
    def __init__(self, mismatch):
        Exception.__init__(self, mismatch)

def _test_option_type(option_type):
    if (option_type != "c") and (option_type != "p"):
        raise GBS_InputError("Invalid Input option_type ({0}). Acceptable value are: c, p".format(option_type))

def _gbs_test_inputs(option_type, fs, x, t, r, b, v):
    """
    Test the inputs to the GBS model for validity.
    This function checks that the inputs are within the acceptable ranges defined in _GBS_Limits.
    If any input is out of range, it raises a GBS_InputError with a descriptive message.
    Parameters:
    ----------
    option_type : str
        The type of option, either 'c' for call or 'p' for put.
    fs : float
        The forward/spot price of the underlying asset.
    x : float
        The strike price of the option.
    t : float
        The time to expiration of the option, in years.
    r : float
        The risk-free interest rate, as a decimal.
    b : float
        The cost of carry, as a decimal.
    v : float
        The implied volatility of the option, as a decimal.
    Raises:
    ------
    GBS_InputError
        If any input is out of the acceptable range defined in _GBS_Limits.
    GBS_CalculationError
        If the inputs are valid but a calculation error occurs.
    
    """

    # -----------
    # Test inputs are reasonable
    _test_option_type(option_type)

    if (x < _GBS_Limits.MIN_X) or (x > _GBS_Limits.MAX_X):
        raise GBS_InputError("Invalid Input Strike Price ({X}). Acceptable range for inputs is {1} to {2}".format(x, _GBS_Limits.MIN_X, _GBS_Limits.MAX_X))

    if (fs < _GBS_Limits.MIN_FS) or (fs > _GBS_Limits.MAX_FS):
        raise GBS_InputError(
            "Invalid Input Forward/Spot Price ({FS}). Acceptable range for inputs is {1} to {2}".format(fs,
                                                                                                      _GBS_Limits.MIN_FS,
                                                                                                      _GBS_Limits.MAX_FS))

    if (t < _GBS_Limits.MIN_T) or (t > _GBS_Limits.MAX_T):
        raise GBS_InputError(
            "Invalid Input Time (T = {0}). Acceptable range for inputs is {1} to {2}".format(t, _GBS_Limits.MIN_T,
                                                                                             _GBS_Limits.MAX_T))

    if (b < _GBS_Limits.MIN_b) or (b > _GBS_Limits.MAX_b):
        raise GBS_InputError(
            "Invalid Input Cost of Carry (b = {0}). Acceptable range for inputs is {1} to {2}".format(b,
                                                                                                      _GBS_Limits.MIN_b,
                                                                                                      _GBS_Limits.MAX_b))

    if (r < _GBS_Limits.MIN_r) or (r > _GBS_Limits.MAX_r):
        raise GBS_InputError(
            "Invalid Input Risk Free Rate (r = {0}). Acceptable range for inputs is {1} to {2}".format(r,
                                                                                                       _GBS_Limits.MIN_r,
                                                                                                       _GBS_Limits.MAX_r))

    if (v < _GBS_Limits.MIN_V) or (v > _GBS_Limits.MAX_V):
        raise GBS_InputError(
            "Invalid Input Implied Volatility (V = {0}). Acceptable range for inputs is {1} to {2}".format(v,
                                                                                                           _GBS_Limits.MIN_V,
                                                                                                           _GBS_Limits.MAX_V))
