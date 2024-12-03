
import inspect
import time
import os
from dotenv import load_dotenv
load_dotenv()
import sys
sys.path.append(
    os.environ.get('WORK_DIR')) # type: ignore
import warnings
from typing import Union
# from trade.helpers.Configuration import Configuration
from trade.helpers.Configuration import ConfigProxy
Configuration = ConfigProxy()
import re
from datetime import datetime
import QuantLib as ql
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from datetime import datetime
from trade.helpers.parse import parse_date, parse_time
import yfinance as yf
# from trade.assets.rates import get_risk_free_rate_helper
from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks.numerical import delta, vega, theta, rho
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
from py_vollib.black_scholes_merton import black_scholes_merton
from scipy.stats import norm
import yfinance as yf
from openbb import obb
import pandas as pd
import numpy as np
import inspect
from trade.helpers.Logging import setup_logger

logger = setup_logger('trade.helpers.helper')

# To-Dos: 
# If still using binomial, change the r to prompt for it rather than it calling a function




def retrieve_timeseries(tick, start, end, interval = '1d', provider = 'yfinance', **kwargs):
    """
    Returns an OHLCV for provided ticker.

    Utilizes OpenBB historical api. Default provider is yfinance.
    """
    data = obb.equity.price.historical(symbol=tick, start_date = start, end_date = end, provider=provider, interval =interval).to_df()
    data.split_ratio.replace(0, 1, inplace = True)
    data['cum_split'] = data.split_ratio.cumprod()
    data['max_cum_split'] = data.cum_split.max()
    data['unadjusted_close'] = data.close * data.max_cum_split
    data['split_factor'] = data.max_cum_split / data.cum_split
    data['chain_price'] = data.close * data.split_factor
    data = data[['open', 'high', 'low', 'close', 'volume','chain_price','unadjusted_close',  'split_ratio', 'cum_split']]
    return data

def identify_interval(timewidth, timeframe, provider = 'default'):
    if provider == 'yfinance':
        TIMEFRAMES = {'day': 'd', 'hour': 'h', 'minute': 'm', 'week': 'W', 'month': 'M', 'quarter': 'Q'}
        assert timeframe.lower() in TIMEFRAMES.keys(), f"For '{provider}' provider timeframes, these are your options, {TIMEFRAMES.keys()}"
        return f"{str(timewidth)}{TIMEFRAMES[timeframe.lower()]}"
    
    elif provider == 'default':
        TIMEFRAMES = {'day': 'd', 'hour': 'h', 'minute': 'm', 'week': 'w', 'month': 'M', 'quarter': 'q', 'year': 'y'}
        assert timeframe.lower() in TIMEFRAMES.keys(), f"For '{provider}' provider timeframes, these are your options, {TIMEFRAMES.keys()}"
        return f"{str(timewidth)}{TIMEFRAMES[timeframe.lower()]}"
    
    
    elif provider == 'fmp':
        TIMEFRAMES = {'day': 'd', 'hour': 'h', 'minute': 'm'}
        assert timeframe.lower() in TIMEFRAMES.keys(), f"For '{provider}' provider timeframes, these are your options, {TIMEFRAMES.keys()}"
        return f"{str(timewidth)}{TIMEFRAMES[timeframe.lower()]}"
    


def identify_length(string, integer):
    TIMEFRAMES_VALUES = {'d': 1, 'w': 5, 'm': 30, 'y': 252, 'q': 91}
    assert string in TIMEFRAMES_VALUES.keys(), f'Available timeframes are {TIMEFRAMES_VALUES.keys()}, recieved "{string}"'
    return integer * TIMEFRAMES_VALUES[string]

def extract_numeric_value(timeframe_str):
    match = re.findall(r'(\d+)([a-zA-Z]+)', timeframe_str)
    integers = [int(num) for num, _ in match][0]
    strings = [str(letter) for _, letter in match][0]
    return strings, integers






def printmd(string):
    from IPython.display import Markdown, display
    display(Markdown(string))

def copy_doc_from(func):
    def wrapper(method):
        method.__doc__ = func.__doc__
        return method
    return wrapper

from datetime import datetime

def contains_time_format(date_str: str) -> bool:
    try:
        datetime.strptime(date_str, '%H:%M:%S')
        return True
    except ValueError:
        return False

def time_distance_helper(exp: str, strt: str = None) -> float:
    if strt is None:
        start_date = datetime.today()
    else:
        strt_2 = parse_date(strt)
        start_date = strt_2
       
    parsed_dte = parse_date(exp + ' 16:00:00') if not contains_time_format(exp) else parse_date(exp)
    parsed_dte = parsed_dte
    days = (parsed_dte - start_date).total_seconds()

    T = days/(365.25*24*3600)
    return T


def binomial(K: Union[int, float], exp_date: str, sigma: float, r: float = None, N: int = 100, S0: Union[int, float, None] = None, y: float = None, tick: str = None,  opttype='P', start: str = None) -> float:
    '''
    Returns the price of an american option

        Parameters:
            K: Strike price
            exp_date: Expiration date
            S0: Spot at current time (Optional)
            r: Risk free rate (Optional)
            N: Number of steps to use in the calculation (Optional)
            y: Dividend yield (Optional)
            Sigma: Implied Volatility of the option
            opttype: Option type ie put or call (Defaults to "P")
            start: Start date of the pricing model. If nothing is passed, defaults to today. If initiated within a context and nothing is passed, defaults to context start date (Optional)
    '''
    if start is None:
        if Configuration.start_date is not None:
            start = Configuration.start_date
        else:
            today = datetime.today()
            start = today.strftime("%Y-%m-%d")
    if tick is not None:
        stock = Stock(tick)
        if y is None:
            y = stock.div_yield()
        if S0 is None:
            S0 = stock.prev_close()
            S0 = S0.close
    else:
        y = 0
    if r is None:
        rates = 0.005
        r = rates.iloc[len(rates)-1, 0]/100

    # Create a formula to get implied vol

    T = time_distance_helper(exp_date, start)
    dt = T/N
    nu = r - 0.5*sigma**2
    u = np.exp(nu*dt + sigma*np.sqrt(dt))
    d = np.exp(nu*dt - sigma*np.sqrt(dt))
    q = (np.exp((r-y)*dt) - d) / (u-d)
    disc = np.exp(-(r-y)*dt)
    opttype = opttype.upper()

    # initialise stock prices at maturity (calculating final stock values at the last nodes)
    S = np.zeros(N+1)
    for j in range(0, N+1):
        S[j] = S0 * u**j * d**(N-j)

    # option payoff, (calculating the payoffs at each final node.)
    C = np.zeros(N+1)
    for j in range(0, N+1):
        if opttype == 'P':
            C[j] = max(0, K - S[j])
        else:
            C[j] = max(0, S[j] - K)

    # backward recursion through the tree
    for i in np.arange(N-1, -1, -1):
        for j in range(0, i+1):
            S = S0 * u**j * d**(i-j)
            C[j] = disc * (q*C[j+1] + (1-q)*C[j])
            if opttype == 'P':
                C[j] = max(C[j], K - S)
            else:
                C[j] = max(C[j], S - K)

    return C[0]


def implied_vol_bs_helper(S0, K, T, r, market_price, flag='c', tol=1e-3, exp_date='2024-03-08'):
    """Compute the implied volatility of a European Option
        S0: initial stock price
        K:  strike price
        T:  maturity
        r:  risk-free rate
        market_price: market observed price
        tol: user choosen tolerance
    """
    max_iter = 200  # max number of iterations
    vol_old = 0.5  # initial guess
    count = 0
    for k in range(max_iter):
        bs_price = bs(flag, S0, K, T, r, vol_old)
        Cprime = vega(flag, S0, K, T, r, vol_old)*100
        C = bs_price - market_price
        vol_new = vol_old - C/Cprime
        bs_new = bs(flag, S0, K, T, r, vol_new)
        # or abs(bs_new - market_price) < tol):
        if (abs((vol_old - vol_new)/vol_old)) < tol:
            break
        vol_old = vol_new
    implied_vol = vol_old

    return implied_vol


def implied_vol_bt(S0, K, r, market_price,exp_date: str, flag='c', tol=0.000000000001,  y=None, start = None):
    """Compute the implied volatility of an American Option
        S0: initial stock price
        K:  strike price
        r:  risk-free rate
        y:  Dividend yield
        market_price: market observed price
        tol: user choosen tolerance
    """
    T = time_distance_helper(exp_date)
    max_iter = 200  # max number of iterations
    vol_old = 0.2  # initial guess
    count = 0
    for k in range(max_iter):
        bs_price = binomial(
            K=K, exp_date=exp_date, S0=S0,  r=r, sigma=vol_old, opttype=flag, y=y, start = start)

        Cprime = vega(flag, S0, K, T, r, vol_old)*100
        C = bs_price - market_price
        vol_new = vol_old - C/Cprime
        bs_new = bs(flag, S0, K, T, r, vol_new)
        # or abs(bs_new - market_price) < tol):
        if (abs((vol_old - vol_new)/vol_old)) < tol:
            break
        vol_old = vol_new
        count += 1
    implied_vol = vol_old

    return implied_vol


def d1_helper(S, K, r, T, sigma, q):
    return (np.log(S / K) + ((r-q) + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2_helper(S, K, r, T, sigma, q):
    return d1_helper(S, K, r, T, sigma, q) - sigma * np.sqrt(T)


def volga(S, K, r, T, sigma, flag, q):
    d1 = d1_helper(S, K, r, T, sigma, q)
    d2 = d2_helper(S, K, r, T, sigma, q)
    flag = flag.upper()
    if sigma <= 0:
        raise ValueError("Volatility must be positive.")
    if flag == 'C' or flag == 'P':
        if flag.upper() == 'C':
            volga = (d1*d2*S*np.exp(-q*T)*norm.cdf(d1)*np.sqrt(T))/sigma
        else:
            volga = (d1*d2*S*np.exp(-q*T)*norm.cdf(-d1)*np.sqrt(T))/sigma
    else:
        raise ValueError("Invalid Option Type. Only 'C' for Call and 'P' for Put are available.")
    return volga


def vanna(S, K, r, T, sigma, flag, q):
    d1 = d1_helper(S, K, r, T, sigma, q)
    d2 = d2_helper(S, K, r, T, sigma, q)
    flag = flag.upper()
    # vg = vega(flag.lower(), S, K, T, r, sigma, q)
    if sigma <= 0:
        raise ValueError("Volatility must be positive.")
    flag = flag.upper()
    if flag == 'C' or flag == 'P':
        if flag.upper() == 'C':
            vanna = -(d2 * np.exp(-q*T)*norm.cdf(d1))/sigma
        else:
            vanna = -(d2 * np.exp(-q*T)*norm.cdf(-d1))/sigma
    else:
        raise ValueError("Invalid Option Type. Only 'C' for Call and 'P' for Put are available.")
    return vanna



import QuantLib as ql
from datetime import datetime

def optionPV_helper(
    spot_price: float,
    strike_price: float | int,
    exp_date: str | datetime, 
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    putcall: str,
    settlement_date_str: str,
    model: str = 'bsm'
):

    """
    Price an American option using QuantLib Engine

    params:
    _________

    spot_price: Underlying Spot
    strike_price: Options Strike price
    exp_date: Options expiration date
    risk_free_rate: Prevailing discount rate, annualized and expressed as 0.01 for 1%
    volatility: Underlying Volatility
    settlement_date_str: pricing date
    model: Preferred pricing method. 
        Available options:
        'bsm': Black Scholes Model
        'bt': Binomial Tree Model
        'mcs': Monte Carlo Simulation

    Returns: 
    ____________

    PV (float): Option present value

    """
    assert model in ['mcs', 'bsm', 'bt'], f"Recieved '{model}' for model. Expected {['mcs', 'bsm', 'bt']}"


    try:
        # Option Parameters
        spot_price = spot_price # Current stock price
        strike_price = strike_price  # Option strike price
        maturity_date_str = exp_date  # Option maturity date as a string
        risk_free_rate = risk_free_rate  # Risk-free interest rate
        volatility = volatility  # Volatility of the underlying asset
        dividend_yield = dividend_yield # Continuous dividend yield

        # Convert string date to QuantLib Date
        maturity_date = ql.Date(pd.to_datetime(maturity_date_str).day,
                                pd.to_datetime(maturity_date_str).month,
                                pd.to_datetime(maturity_date_str).year)

        # QuantLib Settings
        calendar = ql.UnitedStates(ql.UnitedStates.NYSE)  # U.S. market calendar (NYSE)
        day_count = ql.Actual365Fixed()
        settlement_date = ql.Date(pd.to_datetime(settlement_date_str).day,
                                pd.to_datetime(settlement_date_str).month,
                                pd.to_datetime(settlement_date_str).year)

        ql.Settings.instance().evaluationDate = settlement_date

        # Construct the payoff and the exercise objects
        if putcall.upper() == 'P':
            right = ql.Option.Put
        elif putcall.upper() == 'C':
            right = ql.Option.Call
        else:
            raise ValueError(f"Recieved '{putcall}' for putcall. Expected 'P' or 'C'")
        payoff = ql.PlainVanillaPayoff(right, strike_price)
        exercise = ql.AmericanExercise(settlement_date, maturity_date)

        # Market data
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
        dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(settlement_date, dividend_yield, day_count))
        risk_free_ts = ql.YieldTermStructureHandle(ql.FlatForward(settlement_date, risk_free_rate, day_count))
        volatility_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(settlement_date, calendar, volatility, day_count))

        # Black-Scholes-Merton Process (with dividend yield)
        bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividend_ts, risk_free_ts, volatility_ts)

        if model == 'bt':

            # Binomial Pricing (Jarrow-Rudd)
            binomial_engine = ql.BinomialVanillaEngine(bsm_process, "JarrowRudd", 1000)
            american_option = ql.VanillaOption(payoff, exercise)
            american_option.setPricingEngine(binomial_engine)
            binomial_price = american_option.NPV()
            return binomial_price
        
        elif model == 'mcs':
            # Monte Carlo Pricing (Longstaff-Schwartz)
            monte_carlo_engine = ql.MCAmericanEngine(bsm_process, "PseudoRandom", timeSteps=250, requiredSamples=10000)
            american_option = ql.VanillaOption(payoff, exercise)
            american_option.setPricingEngine(monte_carlo_engine)
            monte_carlo_price = american_option.NPV()
            return monte_carlo_price
        
        elif model == 'bsm':
            # Black-Scholes Pricing (Treated as European for comparison)
            european_exercise = ql.EuropeanExercise(maturity_date)
            european_option = ql.VanillaOption(payoff, european_exercise)
            black_scholes_engine = ql.AnalyticEuropeanEngine(bsm_process)
            european_option.setPricingEngine(black_scholes_engine)
            black_scholes_price = european_option.NPV()
            return black_scholes_price
    except Exception as e:
        logger.info('')
        logger.info('"optionPV_helper" raised the below error')
        logger.info(e)
        logger.info(f'Kwargs: {locals()}')
        return 0.0



def pad_string(input_value):
    # Convert float to string and remove the decimal point if needed
    if isinstance(input_value, float):
        input_value = str(input_value).replace('.', '')

    # Convert to string and pad with leading zeros to ensure length of 8
    padded_string = str(input_value).zfill(6)
    
    return padded_string


def IV_handler(**kwargs):
    """
    Calculate the Black-Scholes-Merton implied volatility.

    :param S: underlying asset price
    :type S: float
    :param K: strike price
    :type K: float
    :param sigma: annualized standard deviation, or volatility
    :type sigma: float
    :param t: time to expiration in years
    :type t: float
    :param r: risk-free interest rate
    :type r: float
    :param q: annualized continuous dividend rate
    :type q: float 
    :param flag: 'c' or 'p' for call or put.
    :type flag: str

    >>> S = 100
    >>> K = 100
    >>> sigma = .2
    >>> r = .01
    >>> flag = 'c'
    >>> t = .5
    >>> q = 0

    >>> price = black_scholes_merton(flag, S, K, t, r, sigma, q)
    >>> iv = implied_volatility(price, S, K, t, r, q, flag)

    >>> expected_price = 5.87602423383
    >>> expected_iv = 0.2

    >>> abs(expected_price - price) < 0.00001
    True
    >>> abs(expected_iv - iv) < 0.00001
    
    """


    try:
        iv = implied_volatility(**kwargs)
        if np.isinf(iv):
            iv = 0
        return iv
    except Exception as e:
        logger.warning('')
        logger.warning('"implied_volatility" raised the below error')
        logger.warning(e)
        logger.warning(f'Kwargs: {kwargs}')
        return 0.0



def binomial_implied_vol(price, S, K, r, T, option_type, pricing_date, dividend_yield):
    kwargs = {
        'price': price,
        'S': S,
        'K': K,
        'r': r,
        'T': T,
        'option_type': option_type,
        'pricing_date': pricing_date,
        'dividend_yield': dividend_yield
    }
    try:
        # Ensure option_type is in uppercase
        
        option_type = option_type.upper()
        pricing_date = pd.to_datetime(pricing_date)
        T = pd.to_datetime(T)
        # Convert Python datetime objects to QuantLib Date objects
        pricing_date_ql = ql.Date(pricing_date.day, pricing_date.month, pricing_date.year)
        maturity_date_ql = ql.Date(T.day, T.month, T.year)
        # settlement_date_ql = ql.Date(settlement_date.day, settlement_date.month, settlement_date.year)

        # Option type
        if option_type == 'C':
            option_type = ql.Option.Call
        elif option_type == 'P':
            option_type = ql.Option.Put
        else:
            raise ValueError("Invalid option type. Use 'C' for Call or 'P' for Put.")

        # Set up the QuantLib option
        payoff = ql.PlainVanillaPayoff(option_type, K)
        exercise = ql.EuropeanExercise(maturity_date_ql)
        european_option = ql.VanillaOption(payoff, exercise)

        # Set up the QuantLib Black-Scholes-Merton process
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
        rate_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(pricing_date_ql, r, ql.Actual360())
        )
        dividend_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(pricing_date_ql, dividend_yield, ql.Actual360())
        )
        vol_handle = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(pricing_date_ql, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(0.0)), ql.Actual360())
        )

        bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividend_handle, rate_handle, vol_handle)

        # Set up the binomial engine
        binomial_engine = ql.BinomialVanillaEngine(bsm_process, 'crr', 100)

        # Set the pricing engine to the option
        european_option.setPricingEngine(binomial_engine)

        # Calculate the implied volatility
        implied_volatility = european_option.impliedVolatility(price, bsm_process)
        if np.isinf(implied_volatility):
            implied_volatility = 0
        return implied_volatility

    except Exception as e:
        logger.warning('')
        logger.warning('"binomial_implied_vol" raised the below error')
        logger.warning(e)
        logger.warning(f'Kwargs: {kwargs}')
        return 0.0

def generate_option_tick(symbol, right, exp, strike):
    assert right.upper() in ['P', 'C'], f"Recieved '{right}' for right. Expected 'P' or 'C'"
    assert isinstance(exp, str), f"Recieved '{type(exp)}' for exp. Expected 'str'" 
    assert isinstance(strike, ( float)), f"Recieved '{type(strike)}' for strike. Expected 'float'"
    
    tick_date = pd.to_datetime(exp).strftime('%Y%m%d')
    if str(strike)[-1] == '0':
        strike = int(strike)
    else:
        strike = float(strike)
    return symbol.upper() + tick_date + pad_string(strike) +right.upper()


def wait_for_response(wait_time, condition_func, interval):

    ## Can use time.time to ensure it is not counting (meaning not taking func time into consideration)
    ## This is better to ensure if it at least reaches 15 secs it ends, rather than 15 secs + loop of time to run 
    ## the func call
    elapsed_time = 0
    while elapsed_time < wait_time:
        if condition_func():
            return
        time.sleep(interval)
        elapsed_time += 1

def is_busday(date):
    return bool(len(pd.bdate_range(date, date)))


def is_USholiday(date):
    """
    Returns True if the date is a US holiday, False otherwise
    """

    # import holidays
    import pandas_market_calendars as mcal
    nyse = mcal.get_calendar('NYSE')
    date = pd.to_datetime(date).strftime('%Y-%m-%d')
    return not bool(len(nyse.valid_days(start_date=date, end_date=date)))


def change_to_last_busday(end):
    from pandas.tseries.offsets import BDay
    
    #Enfore time is passed


    if not isinstance(end, str):
        end = end.strftime('%Y-%m-%d %H:%M:%S')

    if pd.to_datetime(end).time() == pd.Timestamp('00:00:00').time():
        end = end + ' 16:00:00' 
    
    ## Make End Comparison Busday
    isBiz = is_busday(end)
    while not isBiz:

        end_dt = pd.to_datetime(end)
        end = (end_dt - BDay( 1)).strftime('%Y-%m-%d %H:%M:%S')
        isBiz = bool(len(pd.bdate_range(end, end)))

    ## Make End Comparison prev day if before 9:30
    if pd.Timestamp(end).time() <pd.Timestamp('9:30').time():
        end = pd.to_datetime(end)-BDay(1)

    # Make End Comparison prev day if holiday
    while is_USholiday(end):
        end_dt = pd.to_datetime(end)
        end = (end_dt - BDay(1)).strftime('%Y-%m-%d %H:%M:%S')

    return pd.to_datetime(end)

def is_class_method(cls, obj):
    """
    Check if an object is a method of a class.

    Params:
    cls: The class to check.
    obj: The object to check.

    Returns:
    bool: True if the object is a method of the class, False otherwise.
    """
    if inspect.isroutine(obj):
        for name, member in inspect.getmembers(cls):
            if member is obj:
                return True
    return False
