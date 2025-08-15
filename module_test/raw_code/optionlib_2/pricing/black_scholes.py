import math
from datetime import datetime, timedelta
from typing import Union, List
import numpy as np
from scipy.stats import norm
from ..assets.forward import (
    Forward,
    time_distance_helper
)
from ..config.defaults import (
    DAILY_BASIS
)
from ..utils.format import (
    option_inputs_assert,
    convert_to_array_individual
)

from ..greeks.numerical.finite_diff import FiniteGreeksEstimator
from ..core.black_scholes_math import (
    black_scholes_analytic_greeks_vectorized,
    black_scholes_vectorized_base
)
from ..assets.forward import (
    EquityForward,
    vectorized_market_forward_calc
)
from trade.helpers.Logging import setup_logger
logger = setup_logger('trade.optionlib.pricing.black_scholes')


# Vectorized Black-Scholes formula using forward price directly
def black_scholes_vectorized(F: Union[float, np.ndarray],
                             K: Union[float, np.ndarray],
                             T: Union[float, np.ndarray],
                             r: Union[float, np.ndarray],
                             sigma: Union[float, np.ndarray],
                             option_type: Union[str, List[str]] = "c"):
    """
    Vectorized Black-Scholes formula using forward price directly.
    F: Forward price (array)
    K: Strike price (array)
    T: Time to expiration in years (array)
    r: Risk-free interest rate (array)
    sigma: Volatility (array)
    option_type: "c" for call, "p" for put (single string)

    Note: This uses numpy for vectorized operations. And yes, multiprocessing can speed it up
    Returns: Option prices (array)
    """
    F = convert_to_array_individual(F)
    K = convert_to_array_individual(K)
    T = convert_to_array_individual(T)
    r = convert_to_array_individual(r)
    sigma = convert_to_array_individual(sigma)
    option_type = convert_to_array_individual(option_type)


    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    df = np.exp(-r * T)
    if not np.all(np.isin(option_type, ["c", "p"])):
        raise ValueError("option_type must be 'c' or 'p'")
    
    call = option_type == "c"
    put = option_type == "p"
    price = np.zeros_like(F)
    price[call] = df[call] * (F[call] * norm.cdf(d1[call]) - K[call] * norm.cdf(d2[call]))
    price[put] = df[put] * (K[put] * norm.cdf(-d2[put]) - F[put] * norm.cdf(-d1[put]))
    return price


def black_scholes_vectorized_scalar(F, K, T, r, sigma, option_type="c"):
    """
    Vectorized Black-Scholes formula using forward price directly.
    F: Forward price (array)
    K: Strike price (array)
    T: Time to expiration in years (array)
    r: Risk-free interest rate (array)
    sigma: Volatility (array)
    option_type: "c" for call, "p" for put (single string)

    Note: This uses numpy for vectorized operations. And yes, multiprocessing can speed it up
    This allows passing single values for F, K, T, r, sigma and broadcasting them to arrays.
    Returns: Option prices (array)
    """
    F = np.asarray(F)
    K = np.asarray(K)
    T = np.asarray(T)
    r = np.asarray(r)
    sigma = np.asarray(sigma)

    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    df = np.exp(-r * T)
    assert option_type in ["c", "p"], "option_type must be 'c' or 'p'"
    
    if option_type == "c":
        price = df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    elif option_type == "p":
        price = df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

    return price


    

def black_scholes_vectorized_market(ticks: List[str],
                                    S: List[float], 
                                    K: List[float], 
                                    valuation_dates: List[datetime],
                                    end_dates: List[datetime],
                                    r: List[float], 
                                    sigma: List[float], 
                                    option_type: str|List[str] = "c",
                                    div_type='discrete'):
    """
    Vectorized Black-Scholes model for market data.
    ticks: List of ticker symbols
    S: List of spot prices (current prices of the underlying assets)
    K: List of strike prices
    valuation_dates: List of valuation dates (dates for which the option is priced)
    end_dates: List of end dates (expiration dates of the options)
    r: List of risk-free rates (annualized)
    sigma: List of volatilities (annualized)
    option_type: "c" for call, "p" for put (single string or list of strings)
    div_type: Type of dividend ('discrete' or 'continuous')
    
    Returns: Option prices (array)
    """
    F = vectorized_market_forward_calc(
        ticks=ticks,
        S=S,
        valuation_dates=valuation_dates,
        end_dates=end_dates,
        r=r,
        div_type=div_type
    )

    # Ensure option_type is a list
    if isinstance(option_type, str):
        option_type = [option_type] * len(ticks)
    elif len(option_type) != len(ticks):
        raise ValueError("option_type must be a single string or a list of strings with the same length as ticks.")
    
    # Convert valuation_dates and end_dates to Timedelta
    T = [time_distance_helper(end_dates[i], valuation_dates[i]) for i in range(len(end_dates))]

    return black_scholes_vectorized_base(
        F=F, 
        K=K, 
        T=T, 
        r=r, 
        sigma=sigma, 
        option_type=option_type
    )




# Base Black-Scholes class
class BlackScholes:
    __GREEK_CALCULATION_STYLE = 'analytic'  # Default to analytic Greeks, else numerical
    def __init__(self,
                 strike_price: float,
                 expiration: datetime,
                 risk_free_rate: float,
                 volatility: float,
                 start_date: datetime,
                 spot_price: float,
                 dividend_type: str = 'discrete',
                 valuation_date: datetime = None,
                 freq: str = "quarterly",
                 div_amount: Union[float, List[float]] = 1.0,
                 option_type: str = "c"):
        """
        Black-Scholes model using forward price directly.

        Parameters:
        - forward: F (e.g., from a ForwardModel)
        - strike_price: K
        - expiration: T (datetime)
        - risk_free_rate: r
        - volatility: sigma (annualized)
        - option_type: "call" or "put"
        """ 
        self.T = time_distance_helper(expiration, valuation_date)
        risk_free_rate = float(risk_free_rate) if risk_free_rate else 0  # Ensure risk-free rate is a float
        option_inputs_assert(sigma=volatility,
                             K=strike_price,
                             S0=spot_price,
                             T=self.T,
                             r=risk_free_rate,
                             market_price=0.1,
                             q=0.0,
                             flag=option_type.lower(),)
        self.forward= Forward(
            start_date=start_date,
            end_date=expiration,
            spot_price=spot_price,
            risk_free_rate=risk_free_rate,
            dividend_type=dividend_type,
            valuation_date=valuation_date,
            freq=freq,
            div_amount=div_amount
        )
        self.expiration = expiration
        self.F = self.forward.get_forward_price()
        self.K = strike_price
        
        self.r = risk_free_rate
        self.sigma = volatility
        self.option_type = option_type.lower()
        self.finite_estimator = FiniteGreeksEstimator(
            price_func=self.price,
            base_params={
                'F': self.F,
                'K': self.K,
                'T': self.T,
                'r': self.r,
                'sigma': self.sigma,
                'q': 0.0, # Assuming no continuous dividend yield for simplicity
                'S': spot_price,  # Including spot price for delta calculation
                'option_type': self.option_type
            },
            dx_thresh=0.00001,
            method='central'  # Use backward method for finite differences
        )

        # Ensure option_type is either 'call' or 'put'
        if self.option_type not in ["c", "p"]:
            raise ValueError("option_type must be 'c' or 'p'")
        
    @property
    def valuation_date(self):
        return self.forward.valuation_date

    @property
    def spot_price(self):
        return self.forward.spot_price

    
    @classmethod
    def set_greek_calculation_style(cls, style: str):
        """
        Set the style for Greek calculations.
        :param style: 'analytic' or 'numerical'
        """
        if style not in ['analytic', 'numerical']:
            raise ValueError("Style must be either 'analytic' or 'numerical'")
        cls.__GREEK_CALCULATION_STYLE = style

    @classmethod
    def get_greek_calculation_style(cls) -> str:
        """
        Get the current Greek calculation style.
        :return: 'analytic' or 'numerical'
        """
        return cls.__GREEK_CALCULATION_STYLE


    def _d1(self):
        # d1 = [ln(F/K) + 0.5*sigma^2*T] / (sigma * sqrt(T))
        numerator = math.log(self.F / self.K) + 0.5 * self.sigma ** 2 * self.T
        denominator = self.sigma * math.sqrt(self.T)
        return numerator / denominator

    def _d2(self):
        # d2 = d1 - sigma * sqrt(T)
        return self._d1() - self.sigma * math.sqrt(self.T)

    def price(self, F=None, K=None, T=None, r=None, sigma=None, option_type=None, S=None,*args, **kwargs) -> float:
        """
        Calculate the Black-Scholes option price using the forward price directly.
        :param F: Forward price (optional, defaults to self.F)
        :param K: Strike price (optional, defaults to self.K)
        :param T: Time to expiration in years (optional, defaults to self.T)
        :param r: Risk-free interest rate (optional, defaults to self.r)
        :param sigma: Volatility (optional, defaults to self.sigma)
        :param option_type: 'c' for call or 'p' for put (optional, defaults to self.option_type)
        :param S: Spot price (optional, defaults to self.forward.spot_price)
        :return: Option price
        """

        # Compute option price using forward-based Black-Scholes formula
        # Call price: e^(-rT) * (F * N(d1) - K * N(d2))
        # Put price: e^(-rT) * (K * N(-d2) - F * N(-d1))

        ## Handle Forward Price, it has to be repriced
        temp_S = self.forward.spot_price 
        temp_rf = self.forward.risk_free_rate


        ## New inputs into the forward model
        self.forward.spot_price = S if S is not None else temp_S  
        self.forward.risk_free_rate = r if r is not None else temp_rf
        if T is not None:
            # Set valuation date back so that (end_date - valuation_date) = T
            temp_val_date = self.forward.valuation_date
            new_val_date = self.expiration - timedelta(days=T * DAILY_BASIS)
            self.forward.valuation_date = new_val_date
            self.forward.dividend.valuation_date = new_val_date


        else:
            temp_val_date = self.forward.valuation_date


        ## Recalculate the forward price
        F = self.forward.get_forward_price()

        ## Reset the forward inputs
        self.forward.spot_price = temp_S  # Reset to original spot price
        self.forward.risk_free_rate = temp_rf
        self.forward.valuation_date = temp_val_date
        self.forward.dividend.valuation_date = temp_val_date


        # Ensure all parameters are numpy arrays for vectorized operations
        K = np.asarray(K) if K is not None else self.K
        T = np.asarray(T) if T is not None else self.T
        r = np.asarray(r) if r is not None else self.r
        sigma = np.asarray(sigma) if sigma is not None else self.sigma
        option_type = option_type if option_type is not None else self.option_type

        return black_scholes_vectorized(F, K, T, r, sigma, option_type=option_type)
    def summary(self) -> dict:
        # Return a dictionary summarizing the model inputs and price
        return {
            "forward": self.F,
            "strike": self.K,
            "T": self.T,
            "r": self.r,
            "vol": self.sigma,
            "type": self.option_type,
            "price": self.price()
        }
    
    def greeks(self) -> dict:
        """
        Calculate the Greeks using finite differences.
        Returns a dictionary with keys 'delta', 'gamma', 'vega', 'theta', 'rho'.
        """
        if self.__GREEK_CALCULATION_STYLE == 'analytic':
            greek = black_scholes_analytic_greeks_vectorized(
                F=self.F, 
                K=self.K, 
                T=self.T, 
                r=self.r, 
                sigma=self.sigma, 
                option_type=self.option_type
            )
            
        elif self.__GREEK_CALCULATION_STYLE == 'numerical':
            greek = self.finite_estimator.all_first_order()
            greek.update(self.finite_estimator.all_second_order())
            
        else:
            raise ValueError(f"Unknown Greek calculation style '{self.__GREEK_CALCULATION_STYLE}'. Use 'analytic' or 'numerical'.")
        greek = dict(sorted(greek.items(), key=lambda item: item[0]))
        return greek
    
    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.option_type.capitalize()} option forward={self.F:.2f}, strike={self.K:.2f}, T={self.T:.2f}, r={self.r:.4f}, vol={self.sigma:.4f}>"

    def __str__(self):
        return self.__repr__()
    
# Market Black-Scholes model that uses the EquityForward to get the forward price
class MarketBlackScholes(BlackScholes):
    def __init__(self,
                 ticker: str,
                 strike_price: float,
                 expiration: datetime,
                 risk_free_rate: float,
                 volatility: float,
                 start_date: datetime,
                 dividend_type: str = 'discrete',
                 valuation_date: datetime = None,
                 option_type: str = "c"):
        """
        Market Black-Scholes model using forward price directly.
        """
        super().__init__(strike_price=strike_price,
                         expiration=expiration,
                         risk_free_rate=risk_free_rate,
                         volatility=volatility,
                         start_date=start_date,
                         spot_price=1, ## Spot price will be set later from the forward
                         dividend_type=dividend_type,
                         valuation_date=valuation_date,
                         freq='quarterly', ## Default to allow initialization
                         div_amount=0,
                         option_type=option_type)
        
        ## Override super's initialization of forward with EquityForward
        self.forward= EquityForward(
            start_date=start_date,
            end_date=expiration,
            dividend_type=dividend_type,
            dividend=None,  # Market dividend will be set later
            valuation_date=valuation_date,
            risk_free_rate=risk_free_rate,
            spot_price=None, 
            ticker=ticker
        )

        self.expiration = expiration
        self.F = self.forward.get_forward_price()
        self.r = self.forward.risk_free_rate
        self.finite_estimator = FiniteGreeksEstimator(
            price_func=self.price,
            base_params={
                'F': None,
                'K': self.K,
                'T': self.T,
                'r': self.r,
                'sigma': self.sigma,
                'q': 0.0, # Assuming no continuous dividend yield for simplicity
                'S': self.spot_price,  # Including spot price for delta calculation
                'option_type': self.option_type
            },
            dx_thresh=0.00001,
            method='central'  # Use backward method for finite differences
        )


    @property
    def spot_price(self):
        """
        Override spot_price to use the forward's spot price.
        """
        return self.forward.spot_price

    def price(self, F=None, K=None, T=None, r=None, sigma=None, option_type=None, S=None,*args, **kwargs) -> float:
        """
        Override price method to use the forward price from the EquityForward.
        """
        ## S, if user overrides Spot, need to update forward spot price
        if S is not None:
            self.forward.spot_price = S
        else:
            S = self.forward.spot_price


        ## If K is not provided, use the BSM Model's strike price
        if K is None:
            K = self.K

        ## If T is not provided, use the BSM Model's T
        if T is None:
            T = self.T

        else:
            # Set valuation date back so that (end_date - valuation_date) = T
            temp_val_date = self.forward.valuation_date
            new_val_date = self.expiration - timedelta(days=T * DAILY_BASIS)
            self.forward.valuation_date = new_val_date
            self.forward.dividend.valuation_date = new_val_date

        ## If r is not provided, use the BSM Model's risk-free rate
        if r is None:
            r = self.r
            r_old = self.forward.risk_free_rate
        else:
            # Update the forward's risk-free rate if provided
            r_old = self.forward.risk_free_rate
            self.forward.risk_free_rate = r

        ## If sigma is not provided, use the BSM Model's volatility
        if sigma is None:
            sigma = self.sigma

        ## If option_type is not provided, use the BSM Model's option type
        if option_type is None:
            option_type = self.option_type

        ## If F is not provided, use the forward's price
        if F is None:
            F = self.forward.get_forward_price()

        ## Clear all bumps to forward to avoid unintended effects
        self.forward.dividend.asset.clear_bump()
        self.forward.risk_free_rate = r_old  # Reset to original risk-free rate

        price = black_scholes_vectorized(F, K, T, r, sigma, option_type=option_type)

        return price
    