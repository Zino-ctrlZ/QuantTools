from abc import ABC, abstractmethod
from datetime import datetime
from typing import List
import numpy as np
from typing import Tuple, Iterable
from numba import njit
from numba import types
from numba.typed import List as _List
from trade.helpers.Logging import setup_logger
from trade.helpers.helper import Scalar
from trade.helpers.threads import runThreads
from ..utils.format import assert_equal_length
from ..config.defaults import DAILY_BASIS
from ..assets.forward import time_distance_helper
from ..assets.dividend import (
    DividendSchedule,
    ContinuousDividendYield,
)
from ..assets.forward import EquityForward

logger = setup_logger("trade.optionlib.pricing.binomial")


def convert_schedule_to_numba(schedule: list[tuple[float, float]]) -> List:
    lst = _List.empty_list(types.UniTuple(types.float64, 2))
    for t_frac, amount in schedule:
        lst.append((float(t_frac), float(amount)))
    return lst


def crr_init_parameters(sigma: float, r: float, T: float, N: int, div_yield: float = 0.0, dividend_type: bool = True):
    """
    params:
    sigma: Volatility of the underlying asset
    r: Risk-free interest rate
    dt: Time step size
    div_yield: Dividend yield (if applicable)
    dividend_type: Type of dividend ('discrete' or 'continuous'
    """
    is_continuous = dividend_type
    return _crr_init_parameters(sigma=sigma, r=r, T=T, N=N, div_yield=div_yield, is_continuous=is_continuous)


@njit
def _crr_init_parameters(
    sigma: float, r: float, T: float, N: int, div_yield: float = 0.0, is_continuous: bool = True
) -> Tuple[float, float, float, float]:
    """
    params:
    sigma: Volatility of the underlying asset
    r: Risk-free interest rate
    dt: Time step size
    div_yield: Dividend yield (if applicable)
    dividend_type: Type of dividend ('discrete' or 'continuous'
    """
    dt = T / N
    y = div_yield if is_continuous else 0.0
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - y) * dt) - d) / (u - d)
    return u, d, p, dt


@njit
def build_tree(S0: float, u: float, d: float, N: int) -> np.ndarray:
    """
    params:
    S0: Initial stock price
    u: Up factor (multiplier for upward movement)
    d: Down factor (multiplier for downward movement)
    N: Number of time steps in the binomial tree
    Returns:
    A 2D list representing the binomial tree of stock prices.
    """
    tree = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            tree[i, j] = S0 * (u**j) * (d ** (i - j))
    return tree


def apply_discrete_dividends(discrete_dividends: List[Tuple[float, float]], stock_tree: np.ndarray, N: int):
    """
    Apply discrete dividends to the stock tree.
    discrete_dividends: List of tuples (time_fraction, dividend_amount)
    stock_tree: The binomial tree of stock prices
    N: Number of time steps in the binomial tree
    Returns:
    A modified stock tree with dividends applied.
    """
    dividends = convert_schedule_to_numba(discrete_dividends)  # Convert dividends to a Numba List
    return _apply_discrete_dividends_jit(dividends, stock_tree, N)


@njit
def _apply_discrete_dividends_jit(dividends: List[Tuple[float, float]], tree: np.ndarray, N: int):
    """
    Apply discrete dividends to the stock tree.
    discrete_dividends: List of tuples (time_fraction, dividend_amount)
    stock_tree: The binomial tree of stock prices
    N: Number of time steps in the binomial tree
    Returns:
    A modified stock tree with dividends applied.
    """
    for t_frac, div in dividends:
        div_step = min(int(round(t_frac * N)), N)
        for i in range(div_step, N + 1):
            for j in range(i + 1):
                tree[i, j] = max(tree[i, j] - div, 0)
    return tree


def create_option_tree(stock_tree: np.ndarray, K: float, option_type: str, N: int) -> np.ndarray:
    """
    Create the option value tree based on the stock price tree.
    stock_tree: The binomial tree of stock prices
    K: Strike price of the option
    option_type: 'c' for call, 'p' for put
    N: Number of time steps in the binomial tree
    Returns:
    A 2D list representing the option value tree.
    """
    if stock_tree is None:
        raise ValueError("stock_tree is None before calling _create_option_tree_jit")
    option_type = 0 if option_type.lower() == "c" else 1  # 0 for call, 1 for put
    return _create_option_tree_jit(stock_tree, K, option_type, N)


@njit
def _create_option_tree_jit(tree: np.ndarray, K: float, option_type: int, N: int) -> np.ndarray:
    """
    Create the option value tree based on the stock price tree.
    stock_tree: The binomial tree of stock prices
    K: Strike price of the option
    option_type: 'c' for call, 'p' for put
    N: Number of time steps in the binomial tree
    Returns:
    A 2D list representing the option value tree.
    """

    if tree is None:
        raise ValueError("stock_tree is None before calling _create_option_tree_jit")
    option_tree = np.zeros_like(tree)
    if option_type == 0:
        for j in range(N + 1):
            option_tree[N, j] = max(0.0, tree[N, j] - K)
    else:
        for j in range(N + 1):
            option_tree[N, j] = max(0.0, K - tree[N, j])
    return option_tree


def calculate_option_values(
    stock_tree: np.ndarray,
    option_values: np.ndarray,
    K: float,
    r: float,
    dt: float,
    N: int,
    p: float,
    american: bool = False,
    option_type: int = 0,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calculate the option values at each node in the binomial tree.
    stock_tree: The binomial tree of stock prices
    option_values: The terminal option values
    r: Risk-free interest rate
    dt: Time step size
    N: Number of time steps in the binomial tree
    Returns:
    A 2D list representing the option value tree.
    """
    stock_tree = np.asarray(stock_tree)
    option_values = np.asarray(option_values)
    option_type = 0 if option_type.lower() == "c" else 1  # 0 for call, 1 for put
    return _calculate_option_values_jit(stock_tree, option_values, K, r, dt, N, p, american, option_type)


@njit
def _calculate_option_values_jit(
    tree: np.ndarray,
    option_tree: np.ndarray,
    K: float,
    r: float,
    dt: float,
    N: int,
    p: float,
    american: bool = False,
    option_type: int = 0,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calculate the option values at each node in the binomial tree.
    stock_tree: The binomial tree of stock prices
    option_values: The terminal option values
    r: Risk-free interest rate
    dt: Time step size
    N: Number of time steps in the binomial tree
    Returns:
    A 2D list representing the option value tree.
    """
    if tree is None:
        raise ValueError("stock_tree is None before calling _create_option_tree_jit")
    if option_tree is None:
        raise ValueError("option_tree is None before calling _calculate_option_values_jit")

    V1 = np.zeros(2)
    V2 = np.zeros(3)
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            expected = np.exp(-r * dt) * (p * option_tree[i + 1, j + 1] + (1 - p) * option_tree[i + 1, j])
            if american:
                if option_type == 0:
                    intrinsic = max(tree[i, j] - K, 0)
                else:
                    intrinsic = max(K - tree[i, j], 0)
                option_tree[i, j] = max(expected, intrinsic)
            else:
                option_tree[i, j] = expected
        if i == 2:
            for j in range(3):
                V2[j] = option_tree[i, j]
        elif i == 1:
            for j in range(2):
                V1[j] = option_tree[i, j]
    return option_tree[0, 0], V1, V2


def crr_binomial_pricing(
    K: float,
    T: float,
    sigma: float,
    r: float,
    N: int,
    S0: float,
    option_type: str = "c",
    american: bool = False,
    div_yield: float = 0.0,
    dividends: List[Tuple[float, float]] = [],  # noqa
    dividend_type: str = "discrete",
) -> float:
    """
    Calculate the price of an option using the Cox-Ross-Rubinstein binomial model.

    Parameters:
    - K: Strike price
    - T: Time to expiration (in years)
    - sigma: Volatility of the underlying asset
    - r: Risk-free interest rate (annualized)
    - N: Number of time steps in the binomial tree
    - S0: Current price of the underlying asset
    - option_type: 'c' for call, 'p' for put (default is 'c')
    - american: True for American option, False for European option (default is False)
    - div_yield: Dividend yield (annualized, default is 0.0)
    - dividends: List of tuples (time fraction, amount) for discrete dividends (default is None)
    - dividend_type: 'discrete' for discrete dividends, 'continuous' for continuous dividends (default is 'discrete')

    If 'discrete', dividends should be a list of tuples where each tuple contains the time fraction (as a float) and the amount (as a float).
    If 'continuous', div_yield should be provided as a float representing the annualized dividend yield.
    If no dividends are provided, the function assumes no dividends.
    If 'dividend_type' is 'continuous', the function will treat the dividend yield as a continuous yield.

    Returns:
    The calculated price of the option.
    """
    is_continuous = dividend_type == "continuous"
    option_type = 0 if option_type.lower() == "c" else 1  # 0 for call, 1 for put
    dividends = convert_schedule_to_numba(dividends)  # Convert dividends to a Numba List

    return _crr_binomial_pricing_jit(K, T, sigma, r, N, S0, option_type, american, div_yield, dividends, is_continuous)


@njit
def _crr_binomial_pricing_jit(
    K: float,
    T: float,
    sigma: float,
    r: float,
    N: int,
    S0: float,
    option_type: int = 0,
    american: bool = False,
    div_yield: float = 0.0,
    dividends: List[Tuple[float, float]] = [],  # noqa
    is_continuous: bool = True,
) -> float:
    """
    Calculate the price of an option using the Cox-Ross-Rubinstein binomial model.

    Parameters:
    - K: Strike price
    - T: Time to expiration (in years)
    - sigma: Volatility of the underlying asset
    - r: Risk-free interest rate (annualized)
    - N: Number of time steps in the binomial tree
    - S0: Current price of the underlying asset
    - option_type: 'c' for call, 'p' for put (default is 'c')
    - american: True for American option, False for European option (default is False)
    - div_yield: Dividend yield (annualized, default is 0.0)
    - dividends: List of tuples (time fraction, amount) for discrete dividends (default is None)
    - dividend_type: 'discrete' for discrete dividends, 'continuous' for continuous dividends (default is 'discrete')

    If 'discrete', dividends should be a list of tuples where each tuple contains the time fraction (as a float) and the amount (as a float).
    If 'continuous', div_yield should be provided as a float representing the annualized dividend yield.
    If no dividends are provided, the function assumes no dividends.
    If 'dividend_type' is 'continuous', the function will treat the dividend yield as a continuous yield.

    Returns:
    The calculated price of the option.
    """
    u, d, p, dt = _crr_init_parameters(sigma, r, T, N, div_yield, is_continuous)
    tree = build_tree(S0, u, d, N)
    _apply_discrete_dividends_jit(dividends, tree, N)
    option_tree = _create_option_tree_jit(tree, K, option_type, N)
    price, _, _ = _calculate_option_values_jit(tree, option_tree, K, r, dt, N, p, american, option_type)
    return price


def vector_crr_binomial_pricing(
    K: Iterable,
    T: Iterable,
    sigma: Iterable,
    r: Iterable,
    N: Iterable,
    S0: Iterable,
    right: Iterable,
    american: Iterable,
    dividend_yield: Iterable = None,
    dividends: Iterable = None,
    dividend_type: Iterable = None,
) -> Iterable[float]:
    """
    Vectorized CRR binomial option pricing.
    Parameters:
    - K: Strike prices
    - T: Time to maturities (in years)
    - sigma: Volatilities
    - r: Risk-free interest rates
    - N: Number of binomial steps
    - S0: Current underlying asset prices
    - right: Option types ('c' for call, 'p' for put)
    - american: Flags indicating if options are American (True) or European (False)
    - dividend_yield: Continuous dividend yields (optional)
    - dividends: Discrete dividend schedules (optional)
    - dividend_type: Types of dividends ('continuous' or 'discrete') (optional)
    Returns:
    - List of option prices
    """

    if dividend_yield is None:
        dividend_yield = [0.0] * len(K)

    if dividends is None:
        dividends = [()] * len(K)

    if dividend_type is None:
        dividend_type = ["continuous"] * len(K)

    assert_equal_length(
        K,
        T,
        sigma,
        r,
        N,
        S0,
        right,
        american,
        dividend_yield,
        dividends,
        dividend_type,
        names=["K", "T", "sigma", "r", "N", "S0", "right", "american", "dividend_yield", "dividends", "dividend_type"],
    )

    return runThreads(
        crr_binomial_pricing,
        [
            K,  # K
            T,  # T
            sigma,  # sigma
            r,  # r
            N,  # N
            S0,  # S0
            right,  # option_type
            american,  # american
            dividend_yield,  # dividend_yield
            dividends,  # dividends,
            dividend_type,
        ],
    )


class BinomialBase(ABC):
    def __init__(
        self,
        K: float,
        expiration: datetime | str,
        sigma: float,
        r: float,
        N: int = 100,
        spot_price: float = None,
        dividend_type: str = "discrete",
        div_amount: float = 0.0,
        option_type: str = "c",
        start_date: datetime | str = None,
        valuation_date: datetime | str = None,
        american: bool = False,
    ):
        super().__init__()
        """
        Base class for Binomial Tree models.
        K: Strike price
        expiration: Expiration date of the option
        sigma: Volatility of the underlying asset
        r: Risk-free interest rate
        N: Number of time steps in the binomial tree
        spot_price: Current price of the underlying asset (optional)
        dividend_type: Type of dividend ('discrete' or 'continuous')
        div_amount: Amount of dividend (if applicable)
        option_type: 'c' for call, 'p' for put
        start_date: Start date for the option pricing (optional)
        valuation_date: Date for which the option is priced (optional)
        """
        self._initialized = False
        self.K = K
        self.expiration = expiration
        self.sigma = sigma
        self.r = r
        self.N = N
        self.S0 = spot_price
        self.dividend_type = dividend_type
        self.div_yield = div_amount if dividend_type == "continuous" else 0.0
        self.discrete_dividends = div_amount if dividend_type == "discrete" else []
        self.option_type = option_type
        self.start_date = start_date
        self.valuation_date = valuation_date
        self.T = time_distance_helper(self.expiration, self.valuation_date or datetime.now())
        self.american = american
        self.dt = self.T / self.N
        self.priced = False
        self.tree = []
        self.option_values = []
        self.stock_tree = []
        self.init_parameters()
        self.build_tree()
        self._initialized = True

    @abstractmethod
    def build_tree(self):
        pass

    @abstractmethod
    def init_parameters(self):
        pass

    @abstractmethod
    def _apply_discrete_dividends(self):
        pass

    @abstractmethod
    def delta(self):
        pass

    @abstractmethod
    def gamma(self):
        pass

    def pricing_warning(self):
        """
        Warning message for pricing issues.
        This method can be overridden in subclasses to provide specific warnings.
        """
        if not self.priced:
            # logger.warning("Option has not been priced yet. Please call the price() method first.")
            print("Option has not been priced yet. Please call the price() method first.")

    def reset_pricing_variables(self):
        """
        Reset pricing variables for a new calculation.
        """
        self.tree = []
        self.option_values = []
        self.stock_tree = []
        self.init_parameters()
        self.build_tree()

    def _tree_numerical(self, attr, dx_thresh=0.01):
        """
        Calculate the numerical value of a Greek (delta, gamma, etc.) using the binomial tree.
        This method is used for numerical approximation of Greeks.
        """
        self.pricing_warning()
        actual_value = getattr(self, attr)
        bump = actual_value * dx_thresh
        up_bump = actual_value + bump
        down_bump = actual_value - bump

        setattr(self, attr, up_bump)
        price_up = self.price()

        setattr(self, attr, down_bump)
        price_down = self.price()

        ## Reset
        setattr(self, attr, actual_value)

        return (price_up - price_down) / (2 * bump)

    def _tree_numerical_second_order(self, attr, dx_thresh=0.01):
        """
        Calculate the second-order numerical value of a Greek using the binomial tree.
        This method is used for numerical approximation of second-order Greeks.
        """
        self.pricing_warning()
        actual_value = getattr(self, attr)
        bump = actual_value * dx_thresh
        up_bump = actual_value + bump
        down_bump = actual_value - bump

        setattr(self, attr, up_bump)
        price_up = self.price()

        setattr(self, attr, actual_value)
        price_mid = self.price()

        setattr(self, attr, down_bump)
        price_down = self.price()

        ## Reset
        setattr(self, attr, actual_value)

        return (price_up - 2 * price_mid + price_down) / (bump**2)

    def theta(self, dx_thresh=0.0001):
        """
        Calculate the theta of the option using the binomial tree.
        Theta is the change in option price with respect to a change in time to expiration.
        Returns:
        Theta value as a float.
        """
        return -self._tree_numerical("T", dx_thresh) / DAILY_BASIS

    def vega(self, dx_thresh=0.0001):
        """
        Calculate the vega of the option using the binomial tree.
        Vega is the change in option price with respect to a change in volatility.
        Returns:
        Vega value as a float.
        """
        return self._tree_numerical("sigma", dx_thresh) / 100

    def rho(self, dx_thresh=0.0001):
        """
        Calculate the rho of the option using the binomial tree.
        Rho is the change in option price with respect to a change in risk-free interest rate.
        Returns:
        Rho value as a float.
        """
        return self._tree_numerical("r", dx_thresh) / 100

    def volga(self, dx_thresh=0.0001):
        """
        Calculate the volga of the option using the binomial tree.
        Volga is the change in vega with respect to a change in volatility.
        Returns:
        Volga value as a float.
        """
        return self._tree_numerical_second_order("sigma", dx_thresh) / 100**2

    def __setattr__(self, name, value):
        protected = [
            "K",
            "expiration",
            "sigma",
            "r",
            "N",
            "S0",
            "dividend_type",
            "div_yield",
            "discrete_dividends",
            "option_type",
            "start_date",
            "valuation_date",
            "T",
            "american",
        ]

        if not hasattr(self, "_initialized") or not self._initialized:
            # Allow setting attributes before initialization
            super().__setattr__(name, value)
            return

        if hasattr(self, "_initialized") and self._initialized:
            if name in protected:
                # raise AttributeError(f"'{name}' is read-only after initialization.")
                logger.warning(f"'{name}' is read-only after initialization. Resetting pricing variables.")
        super().__setattr__(name, value)  ## Set
        if name in protected:
            # Reset pricing variables if a protected attribute is set
            logger.info(f"Resetting pricing variables due to change in '{name}'.")
            self.reset_pricing_variables()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(K={self.K}, expiration={self.expiration}, dividend_type={self.dividend_type})"
        )


class VectorBinomialBase(BinomialBase):
    @abstractmethod
    def init_parameters(self):
        """
        Initialize parameters for the binomial tree.
        This method should be called before building the tree.
        """
        pass

    def build_tree(self):
        """
        Build the binomial tree structure.
        This method should be implemented in subclasses.
        """
        self.stock_tree = build_tree(S0=self.S0, u=self.u, d=self.d, N=self.N)
        if self.dividend_type == "discrete":
            self._apply_discrete_dividends()  # Apply discrete dividends at time step 0

    def _apply_discrete_dividends(self) -> float:
        """
        Apply discrete dividend adjustment to the stock price at a given time step.
        """
        if not list(self.discrete_dividends):
            return
        self.stock_tree = apply_discrete_dividends(
            discrete_dividends=self.discrete_dividends, stock_tree=self.stock_tree, N=self.N
        )

    def __create_option_tree(self):
        self.option_values = create_option_tree(
            stock_tree=self.stock_tree, K=self.K, option_type=self.option_type, N=self.N
        )

    def price(self):
        self.__create_option_tree()  # Create the option tree based on terminal stock prices
        option_values = self.option_values
        price, self.V1, self.V2 = calculate_option_values(
            stock_tree=self.stock_tree,
            option_values=option_values,
            K=self.K,
            r=self.r,
            dt=self.dt,
            N=self.N,
            p=self.p,
            american=self.american,
            option_type=self.option_type,
        )
        self.priced = True
        return price

    def delta(self):
        """
        Calculate the delta of the option using the binomial tree.
        Delta is the change in option price with respect to a change in the underlying asset price.
        Returns:
        Delta value as a float.
        """
        self.pricing_warning()
        if self.N < 1:
            raise ValueError("N must be at least 1 to calculate delta.")

        if not hasattr(self, "V1"):
            self.price()

        stock_tree = self.stock_tree
        delta = (self.V1[1] - self.V1[0]) / (stock_tree[1][1] - stock_tree[1][0])
        return delta

    def gamma(self):
        """
        Calculate the gamma of the option using the binomial tree.
        Gamma is the rate of change of delta with respect to a change in the underlying asset price.
        Returns:
        Gamma value as a float.
        """
        self.pricing_warning()
        if self.N < 2:
            raise ValueError("N must be at least 2 to calculate gamma.")

        if not hasattr(self, "V2"):
            self.price()

        V2, S2 = self.V2, self.stock_tree[2]
        delta_up = (V2[2] - V2[1]) / (S2[2] - S2[1])
        delta_down = (V2[1] - V2[0]) / (S2[1] - S2[0])
        gamma = (delta_up - delta_down) / ((S2[2] - S2[0]) / 2)  # Average change in delta over the interval
        return gamma


## Child classes for specific binomial models (Vector Operators)


class VectorBinomialCRR(VectorBinomialBase):
    def init_parameters(self):
        """
        Initialize parameters for the binomial tree.
        This method should be called before building the tree.
        """
        q = self.div_yield if self.dividend_type == "continuous" else 0.0
        self.u, self.d, self.p, _ = crr_init_parameters(
            sigma=self.sigma, r=self.r, T=self.T, N=self.N, div_yield=q, dividend_type=self.dividend_type
        )


class VectorBinomialLR(VectorBinomialBase):  # or NodeBinomialBase
    def init_parameters(self):
        """
        Initialize Leisen-Reimer parameters: u, d, p.
        """
        q = self.div_yield if self.dividend_type == "continuous" else 0.0
        self.dt = self.T / self.N
        v = self.sigma * np.sqrt(self.dt)

        self.u = np.exp(v)
        self.d = np.exp(-v)

        d1 = (np.log(self.S0 / self.K) + (self.r - q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))

        x = d1  # Can also use d2 for puts, but d1 gives better results overall

        # Peizer-Pratt inversion of CDF (used by Leisen-Reimer)
        w = np.sqrt(1 - np.exp(-2 * (x**2) / self.N))
        self.p = 0.5 + np.sign(x) * w / 2


# Child classes for specific binomial models (Node Operators)
class Node(Scalar):
    def __init__(self, stock_price, position, option_value=0.0):
        super().__init__(value=option_value)
        self.stock_price = stock_price
        self.value = option_value
        self.up = None
        self.down = None
        self.position = position  # Position in the binomial tree (e.g., index or identifier)

    @property
    def option_value(self):
        return self.value

    @option_value.setter
    def option_value(self, value):
        self.value = value

    def __eq__(self, value):
        if isinstance(value, Node):
            return self.stock_price == value.stock_price and self.position == value.position
        return False

    def __repr__(self):
        return f"Node(price={self.stock_price}, option_value={self.option_value}, pos={self.position})"


class NodeBinomialBase(BinomialBase):
    @abstractmethod
    def init_parameters(self):
        """
        Initialize parameters for the binomial tree.
        This method should be called before building the tree.
        """
        pass

    def build_tree(self):
        """
        Build the binomial tree structure.
        This method should be implemented in subclasses.
        """
        self.tree = []
        for i in range(self.N + 1):
            level = []
            for j in range(i + 1):
                S = self.S0 * (self.u**j) * (self.d ** (i - j))
                node = Node(stock_price=S, position=(i, j))
                level.append(node)
            self.tree.append(level)

        for i in range(self.N):
            for j in range(i + 1):
                current = self.tree[i][j]
                current.down = self.tree[i + 1][j]  # one down move
                current.up = self.tree[i + 1][j + 1]  # one up move

        if self.dividend_type == "discrete":
            self._apply_discrete_dividends()  # Apply discrete dividends at time step 0

    def _apply_discrete_dividends(self) -> float:
        """
        Apply discrete dividend adjustment to the stock price at a given time step.
        """
        if not list(self.discrete_dividends):
            return

        for t_frac, div in self.discrete_dividends:
            div_step = min(int(round(t_frac * self.N)), self.N)
            for i in range(div_step, self.N + 1):
                for node in self.tree[i]:
                    node.stock_price = max(node.stock_price - div, 0)

    def __create_option_tree(self):
        terminal_nodes = self.tree[-1]
        for node in terminal_nodes:
            node.option_value = (
                max(0, node.stock_price - self.K) if self.option_type == "c" else max(0, self.K - node.stock_price)
            )

    def price(self):
        self.__create_option_tree()
        tree = self.tree

        for i in range(self.N - 1, -1, -1):
            for _, node in enumerate(tree[i]):
                up_val = node.up.option_value
                down_val = node.down.option_value
                expected = np.exp(-self.r * self.dt) * (self.p * up_val + (1 - self.p) * down_val)

                if self.american:
                    intrinsic = (
                        max(node.stock_price - self.K, 0)
                        if self.option_type == "c"
                        else max(self.K - node.stock_price, 0)
                    )
                    node.option_value = max(expected, intrinsic)
                else:
                    node.option_value = expected

        self.priced = True
        return tree[0][0].option_value

    def delta(self):
        """
        Calculate the delta of the option using the binomial tree.
        Delta is the change in option price with respect to a change in the underlying asset price.
        Returns:
        Delta value as a float.
        """
        self.pricing_warning()
        if self.N < 1:
            raise ValueError("N must be at least 1 to calculate delta.")

        stock_tree = self.tree
        V1 = self.tree[1]
        delta = (V1[1] - V1[0]) / (stock_tree[1][1].stock_price - stock_tree[1][0].stock_price)
        return delta

    def gamma(self):
        """
        Calculate the gamma of the option using the binomial tree.
        Gamma is the rate of change of delta with respect to a change in the underlying asset price.
        Returns:
        Gamma value as a float.
        """
        self.pricing_warning()
        if self.N < 2:
            raise ValueError("N must be at least 2 to calculate gamma.")

        if not hasattr(self, "V2"):
            self.price()

        V2, S2 = self.tree[2], self.tree[2]
        delta_up = (V2[2] - V2[1]) / (S2[2].stock_price - S2[1].stock_price)
        delta_down = (V2[1] - V2[0]) / (S2[1].stock_price - S2[0].stock_price)
        gamma = (delta_up - delta_down) / (
            (S2[2].stock_price - S2[0].stock_price) / 2
        )  # Average change in delta over the interval
        return gamma


class NodeBinomialCRR(NodeBinomialBase):
    def init_parameters(self):
        """
        Initialize parameters for the binomial tree.
        This method should be called before building the tree.
        """
        if self.dividend_type == "continuous":
            y = self.div_yield  ## Continuous dividend yield adjustment
        else:
            y = 0.0
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = np.exp(-(self.sigma * np.sqrt(self.dt)))
        self.p = (np.exp((self.r - y) * self.dt) - self.d) / (self.u - self.d)


class NodeBinomialLR(NodeBinomialBase):  # or NodeBinomialBase
    def init_parameters(self):
        """
        Initialize Leisen-Reimer parameters: u, d, p.
        """
        q = self.div_yield if self.dividend_type == "continuous" else 0.0
        self.dt = self.T / self.N
        v = self.sigma * np.sqrt(self.dt)

        self.u = np.exp(v)
        self.d = np.exp(-v)

        d1 = (np.log(self.S0 / self.K) + (self.r - q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))

        x = d1  # Can also use d2 for puts, but d1 gives better results overall

        # Peizer-Pratt inversion of CDF (used by Leisen-Reimer)
        w = np.sqrt(1 - np.exp(-2 * (x**2) / self.N))
        self.p = 0.5 + np.sign(x) * w / 2


# Market Child Classes
class MarketBinomial(VectorBinomialCRR):
    def __init__(
        self,
        tick: str,
        K: float,
        expiration: datetime | str,
        sigma: float,
        N: int = 100,
        dividend_type: str = "discrete",
        option_type: str = "c",
        start_date: datetime | str = None,
        valuation_date: datetime | str = None,
        r: float = None,
        american: bool = False,
    ):
        # super().__init__()
        """
        Base class for Binomial Tree models.
        K: Strike price
        expiration: Expiration date of the option
        sigma: Volatility of the underlying asset
        r: Risk-free interest rate
        N: Number of time steps in the binomial tree
        spot_price: Current price of the underlying asset (optional)
        dividend_type: Type of dividend ('discrete' or 'continuous')
        div_amount: Amount of dividend (if applicable)
        option_type: 'c' for call, 'p' for put
        start_date: Start date for the option pricing (optional)
        valuation_date: Date for which the option is priced (optional)
        """
        self._initialized = False
        self.K = K
        self.expiration = expiration
        self.sigma = sigma
        self.N = N
        self.forward = EquityForward(
            start_date=start_date or datetime.now(),
            end_date=expiration,
            ticker=tick,
            valuation_date=valuation_date or datetime.now(),
            risk_free_rate=r,
            dividend_type=dividend_type,
            dividend=None,  # Market dividend will be set later
        )
        self.r = r or self.forward.risk_free_rate
        self.dividend_type = dividend_type
        self.option_type = option_type
        self.start_date = start_date
        self.valuation_date = valuation_date
        self.T = time_distance_helper(self.expiration, self.valuation_date or datetime.now())
        self.american = american
        self.dt = self.T / self.N
        self.tree = []
        self.option_values = []
        self.stock_tree = []
        self.init_parameters()
        self.build_tree()
        self._initialized = True

    @property
    def asset(self):
        """
        Property to access the underlying asset of the forward contract.
        """
        return self.forward.dividend.asset

    @property
    def S0(self):
        """
        Property to access the current spot price of the underlying asset.
        """
        return self.asset.spot_price

    @property
    def discrete_dividends(self):
        """
        Property to access the discrete dividends of the forward contract.
        """
        if isinstance(self.forward.dividend, DividendSchedule):
            return self.forward.dividend.get_year_fractions()
        else:
            return []

    @property
    def div_yield(self):
        """
        Property to access the continuous dividend yield of the forward contract.
        """
        if isinstance(self.forward.dividend, ContinuousDividendYield):
            return self.forward.dividend.yield_rate
        else:
            return 0.0
