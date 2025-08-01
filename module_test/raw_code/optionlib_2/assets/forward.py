from datetime import datetime
import math
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List
import pandas as pd
from trade.helpers.helper import time_distance_helper
from trade.helpers.Logging import setup_logger
from .dividend import (
    Dividend, 
    DividendSchedule, 
    ContinuousDividendYield,
    MarketDividendSchedule,
    MarketContinuousDividends,
    get_vectorized_continuous_dividends,
    get_vectorized_dividend_rate,
    get_vectorized_dividend_scehdule,
    vectorized_discrete_pv
)

from ..config.defaults import (
    DIVIDEND_LOOKBACK_YEARS,
    DIVIDEND_FORECAST_METHOD
)
from ..utils.format import (
    assert_equal_length, 
    convert_to_array
)

logger = setup_logger('trade.optionlib.assets.forward')

dividend_factory = {
    "discrete": DividendSchedule,
    "continuous": ContinuousDividendYield
}

# Base Abstract class for forward models
class ForwardModel(ABC):
    @abstractmethod
    def get_forward_price(self) -> float:
        """
        Calculate the forward price.
        """
        pass

    @abstractmethod
    def summary(self) -> dict:
        """
        Return a summary of the forward model.
        """
        pass

    @abstractmethod
    def get_end_date(self) -> datetime:
        """
        Get the end date of the forward contract.
        """
        pass

    @abstractmethod
    def get_start_date(self) -> datetime:
        """
        Get the start date of the forward contract.
        """
        pass


## Should regular Forward initialize a Market Dividend?: No, seperated to AssetMarketForward
## Technically, regular Forward is a Model. It differs from market forward because it is the actual Forward/Future Price: No reply needed
## Decided to separate the two classes to avoid confusion and maintain clarity in the API.
## Should Dividends see Stock & Stock see dividends?: Stock can't see dividends, but it can have a dividend schedule.
    ## Dividends can see stock.

# Base class for Forward contracts
class Forward(ForwardModel):
    def __init__(self,
                 start_date: datetime|str,
                 end_date: datetime|str,
                 risk_free_rate: float,
                 dividend_type: str,
                 dividend: Union[Dividend, None] = None,
                 valuation_date: datetime = None,
                 spot_price: float=None,
                 freq: str = "quarterly",
                 div_amount: Union[float, List[float]] = 1.0,
                 **kwargs):
        """
        Initialize a Forward object.
        start_date: datetime or str - The start date of the forward contract.
        end_date: datetime or str - The end date of the forward contract.
        spot_price: float - The current spot price of the underlying asset.
        risk_free_rate: float - The risk-free interest rate (annualized).
        dividend_type: str - The type of dividend ('discrete' or 'continuous').
        dividend: Dividend or None - An instance of a Dividend subclass or None.
        valuation_date: datetime - The date for valuation purposes (default is start_date). 
        freq: str - The frequency of dividends ('monthly', 'quarterly', 'semiannual', 'annual').
        div_amount: float or list - The dividend amount (can be a scalar or a list). For discrete dividends, this is the amount per period, while for continuous dividends, it is the yield rate.
        """

    
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self.dividend_type = dividend_type
        self.valuation_date = pd.to_datetime(valuation_date) if valuation_date else pd.to_datetime(start_date)  
        self.start_date = start_date if isinstance(start_date, datetime) else pd.to_datetime(start_date)
        self.end_date = end_date if isinstance(end_date, datetime) else pd.to_datetime(end_date)        
        self._initalize_dividend(dividend_type=dividend_type, 
                                 dividend=dividend, 
                                 freq=freq, 
                                 div_amount=div_amount, **kwargs)
        


    def _initalize_dividend(self, 
                            dividend_type: str, 
                            dividend:Union[None, Dividend], 
                            freq:str, 
                            div_amount:float, 
                            ticker = None,
                            **kwargs):
        """
        Initialize the dividend based on the type.
        """
        ## Ensure dividend is permitted
        if dividend is None:
            if dividend_type not in dividend_factory:
                raise ValueError(f"Unsupported dividend type '{dividend_type}'. Use one of {list(dividend_factory.keys())}.")
        
        ## Create the dividend object based on the type
        if isinstance(dividend, Dividend):
            self.dividend = dividend

        elif dividend_type == 'continuous':

            logger.info("No ticker provided for a continuous dividend yield. Will construct ContinuousDividendYield instead.")
            self.dividend = ContinuousDividendYield(
                yield_rate=div_amount,
                start_date=self.start_date,
                end_date=self.end_date,
                valuation_date=self.valuation_date
            )
        elif dividend_type == 'discrete':
            logger.info("No ticker provided for a discrete dividend schedule. Will construct DividendSchedule instead.")
            self.dividend = DividendSchedule(
                start_date=self.start_date,
                end_date=self.end_date,
                freq=freq,
                amount=div_amount,
                valuation_date=self.valuation_date
            )

        elif not isinstance(dividend, Dividend):
            raise TypeError("Dividend must be an instance of Dividend or its subclasses.")
        
    def summary(self) -> dict:
        return {
        "spot": self.spot_price,
        "forward": self.get_forward_price(),
        "type": self.dividend.get_type(),
        "valuation": self.valuation_date.date(),
        "expiry": self.end_date.date()
        }

    def get_forward_price(self) -> float:
        """
        Calculate the forward price using the formula:
        F = S * e^{(r - q)T}
        where:
        S = spot price
        r = risk-free rate
        q = dividend yield
        T = time to maturity in years
        """
        T = time_distance_helper(self.end_date, self.valuation_date)
        if T <= 0:
            raise ValueError("End date must be after valuation date.")
        
        # Get the present value of the dividend
        if self.dividend.get_type() == "discrete":
            dividend_pv = self.dividend.get_present_value(self.risk_free_rate, sum_up = True)
            return (self.spot_price - dividend_pv) * math.exp(self.risk_free_rate * T)
        elif self.dividend.get_type() == "continuous":
            dividend_factor = self.dividend.get_present_value(self.end_date, **{})
            return self.spot_price * (math.exp(self.risk_free_rate * T) * dividend_factor) ## Already discounted
        else:
            raise ValueError(f"Unsupported dividend type '{self.dividend.get_type()}'. Use 'discrete' or 'continuous'.")
        
    def get_end_date(self) -> datetime:
        return self.end_date
    def get_start_date(self) -> datetime:
        return self.start_date

    def __repr__(self):
        return f"<Forward: ({repr(self.summary())}>"
    

## Would it be better to have Forward access Stock? And enforce singleton?
## For spot bumps, we can have a setter, where additions are moved to bump var, and anytime spot is called, it returns the spot + bump.
## Probably would be a good idea to cache spot timeseries in Stock.


class MarketForward(ForwardModel):
    def __init__(self, forward_price: float, start_date: datetime, end_date: datetime):
        """
        Initialize a MarketForward object.
        forward_price: float - The market forward price.
        start_date: datetime - The start date of the forward contract.
        end_date: datetime - The end date of the forward contract.
        """
        self.forward_price = forward_price
        self.start_date = start_date
        self.end_date = end_date

    def get_forward_price(self) -> float:
        return self.forward_price

    def summary(self) -> dict:
        return {
            "forward": self.forward_price,
            "start_date": self.start_date.date(),
            "end_date": self.end_date.date()
        }
    
    def get_end_date(self) -> datetime:
        return self.end_date
    def get_start_date(self) -> datetime:
        return self.start_date
    def __repr__(self):
        return f"<MarketForward: {self.forward_price} from {self.start_date.date()} to {self.end_date.date()}>"
    


class EquityForward(Forward):
    def __init__(self,
                start_date: datetime|str,
                end_date: datetime|str,
                dividend_type: str,
                dividend: Union[Dividend, None] = None,
                valuation_date: datetime = None,
                risk_free_rate: float = None,
                spot_price: float=None,
                ticker = None,
                **kwargs):
        
        
        self.dividend_type = dividend_type
        self.valuation_date = pd.to_datetime(valuation_date) if valuation_date else pd.to_datetime(start_date)  
        self.start_date = start_date if isinstance(start_date, datetime) else pd.to_datetime(start_date)
        self.end_date = end_date if isinstance(end_date, datetime) else pd.to_datetime(end_date)  
        self.ticker = ticker
        self.forward_spot_price = spot_price
        self._initalize_dividend(dividend_type=dividend_type,
                                    dividend=dividend, 
                                    ticker=ticker, **kwargs)
        self.risk_free_rate = risk_free_rate or self.dividend.asset.rf_rate

    @property
    def spot_price(self):
        logger.info(f"Accessing spot_price of {self.ticker} from {self.dividend.__class__.__name__}")
        if isinstance(self.dividend, (MarketContinuousDividends,
                                        MarketDividendSchedule)):
            return self.dividend.spot_price
        
    @spot_price.setter
    def spot_price(self, v):
        logger.info(f"Setting spot_price of {self.ticker} to {v} in {self.dividend.__class__.__name__}")
        if isinstance(self.dividend, (MarketContinuousDividends,
                                        MarketDividendSchedule)):
            self.dividend.spot_price = v
            self.forward_spot_price = v
        else:
            raise TypeError("Spot price can only be set for MarketContinuousDividends or MarketDividendSchedule.")
            


    def _initalize_dividend(self, 
                            dividend_type: str, 
                            dividend:Union[None, Dividend], 
                            ticker = None,
                            **kwargs):
        """
        Initialize the dividend based on the type.
        """
        ## Ensure dividend is permitted
        if dividend is None:
            if dividend_type not in dividend_factory:
                raise ValueError(f"Unsupported dividend type '{dividend_type}'. Use one of {list(dividend_factory.keys())}.")
        
        ## Create the dividend object based on the type
        if isinstance(dividend, Dividend):
            self.dividend = dividend

        elif dividend_type == 'continuous':
            logger.info("Ticker provided for a continuous dividend yield. Will construct MarketContinuousDividends instead.")
            self.dividend = MarketContinuousDividends(
                            ticker=ticker,
                            start_date=self.start_date,
                            end_date=self.end_date,
                            valuation_date=self.valuation_date,
                            spot_price=self.forward_spot_price
                        )
        elif dividend_type == 'discrete':
            logger.info("Ticker provided for a discrete dividend schedule. Will construct MarketDividendSchedule instead.")
            self.dividend = MarketDividendSchedule(
                ticker=ticker,
                start_date=self.start_date,
                end_date=self.end_date,
                valuation_date=self.valuation_date,
                lookback_years=kwargs.get('lookback_years', DIVIDEND_LOOKBACK_YEARS),
                growth_method=kwargs.get('growth_method', DIVIDEND_FORECAST_METHOD),
                spot_price=self.forward_spot_price
            )

        elif not isinstance(dividend, Dividend):
            raise TypeError("Dividend must be an instance of Dividend or its subclasses.")


# Vectorized version of the Forward class

def vectorized_forward_continuous(S, r, q_factor, T):
    """
    S: spot prices (array)
    r: risk-free rates (array)
    q: Discounted Dividend Factor
    T: time to maturity (array)
    """
    assert_equal_length(S, r, q_factor, T)
    S, r, T, q_factor = convert_to_array(S, r, T, q_factor)
    return S * np.exp(r * T) * q_factor

def vectorized_forward_discrete(S, r, T, pv_divs):
    """
    S: spot prices (array)
    r: risk-free rates (array)
    T: time to maturity (array)
    pv_divs: Summation of present value of all dividends till end date
    """
    assert_equal_length(S, r, pv_divs, T)
    S, r, T, pv_divs = convert_to_array(S, r, T, pv_divs)
    forward = (S - pv_divs) * np.exp(r * T)
    return forward


def vectorized_market_forward_calc(ticks: List[str], 
                                   S: List[float], 
                                   valuation_dates: List[datetime], 
                                   end_dates: List[datetime], 
                                   r: List[float], 
                                   div_type='discrete',
                                   return_div=False) -> np.ndarray:
    """
    Vectorized calculation of forward prices for multiple tickers.
    ticks: List of ticker symbols
    S: List of spot prices (current prices of the underlying assets)
    valuation_dates: List of valuation dates (dates for which the option is priced)
    end_dates: List of end dates (expiration dates of the options)
    r: List of risk-free rates (annualized)
    div_type: Type of dividend ('discrete' or 'continuous')
    Returns: Forward prices (array)
    """
        
    # Get Dividends & Calculate the forward price
    if div_type == 'discrete':

        schedule = get_vectorized_dividend_scehdule(
            tickers=ticks,
            start_dates=valuation_dates,
            end_dates=end_dates,
            valuation_dates=valuation_dates,
        )
        div_amt=vectorized_discrete_pv(schedules=schedule,
                                  r=r,
                                  _valuation_dates=valuation_dates,
                                  _end_dates=end_dates,)
        F = vectorized_forward_discrete(
            S=S,
            r=r,
            T=[time_distance_helper(end_dates[i], valuation_dates[i]) for i in range(len(end_dates))],
            pv_divs=div_amt
        )

    elif div_type == 'continuous':
        div_rate = get_vectorized_dividend_rate(
            tickers=ticks,
            spots=S,
            valuation_dates=valuation_dates,
        )

        div_amt = get_vectorized_continuous_dividends(div_rates=div_rate,
                                                _valuation_dates=valuation_dates,
                                                _end_dates=end_dates,)
        F = vectorized_forward_continuous(
            S=S,
            r=r,
            q_factor=div_amt,
            T=[time_distance_helper(end_dates[i], valuation_dates[i]) for i in range(len(end_dates))]
        )
    else:
        raise ValueError(f"Unsupported dividend type '{div_type}'. Use 'discrete' or 'continuous'.")
    
    return (F, div_amt) if return_div else F