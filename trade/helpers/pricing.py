from assets.rates import get_risk_free_rate_helper
from helpers.parse import parse_date
from bin.asset import Stock
from bin.asset import Stock
import datetime
from datetime import datetime
from datetime import date
import numpy as np
from helpers.Configuration import Configuration
from typing import Union
import warnings
warnings.filterwarnings("ignore")


def time_distance_helper(exp: str, strt: str = None) -> float:
    if strt is None:
        start_date = date.today()
    else:
        strt_2 = parse_date(strt)
        start_date = strt_2.date()
    parsed_dte = parse_date(exp)
    parsed_dte = parsed_dte.date()
    days = (parsed_dte - start_date).days
    T = days/365
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
        rates = get_risk_free_rate_helper()
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
