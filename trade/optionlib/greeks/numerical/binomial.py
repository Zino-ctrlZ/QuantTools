import numpy as np
from typing import Union, List
from ...utils.format import convert_to_array, assert_equal_length,equalize_lengths
from ...pricing.binomial import VectorBinomialCRR
from trade.helpers.Logging import setup_logger
logger = setup_logger('trade.optionlib.greeks.numerical.binomial')

def binomial_tree_price_batch(
    K: float|np.ndarray,
    expiration: float|np.ndarray,
    sigma: float|np.ndarray,
    r: float|np.ndarray,
    N: float|np.ndarray,
    S: float|np.ndarray,
    dividend_type: float|np.ndarray,
    div_amount: float|np.ndarray,
    option_type: float|np.ndarray,
    start_date: float|np.ndarray,
    valuation_date: float|np.ndarray,
    american: float|np.ndarray,
):
    """
    Batch pricing of options using a binomial tree model (CRR).
    
    Parameters:
    - K: Strike price
    - expiration: Expiration date of the option
    - sigma: Volatility of the underlying asset
    - r: Risk-free interest rate
    - N: Number of time steps in the binomial tree
    - spot_price: Current price of the underlying asset (optional)
    - dividend_type: Type of dividend ('discrete' or 'continuous')
    - div_amount: Amount of dividend (if applicable)
    - option_type: 'c' for call, 'p' for put
    - start_date: Start date for the option pricing (optional)
    - valuation_date: Date for which the option is priced (optional)
    
    Returns:
    - price: Calculated option prices
    - models: List of binomial tree models used for pricing
    """

    K, expiration, sigma, r, N, S, dividend_type, option_type, start_date, valuation_date, american = equalize_lengths( K, expiration, sigma, r, N, S, dividend_type, option_type, start_date, valuation_date, american)
    K, expiration, sigma, r, N, S, dividend_type, option_type, start_date, valuation_date, american = map(
        convert_to_array, 
        (K, expiration, sigma, r, N, S, dividend_type, option_type, start_date, valuation_date, american)
    )

    equal = assert_equal_length(
        K, expiration, sigma, r, N, S, dividend_type, div_amount, option_type, start_date, valuation_date, american
    )
    if not equal:
        logger.error(f"Lengths: K={len(K)}, expiration={len(expiration)}, sigma={len(sigma)}, r={len(r)}, N={len(N)}, S={len(S)}, "
                 f"dividend_type={len(dividend_type)}, option_type={len(option_type)}, start_date={len(start_date)}, "
                 f"valuation_date={len(valuation_date)}, american={len(american)}, div_amount={len(div_amount)}")
        raise ValueError("All input lists must have the same length.")
    # Ensure all inputs are numpy arrays
    models = [
        VectorBinomialCRR(
            K=k,
            expiration=exp,
            sigma=s,
            r=ri,
            N=int(n),
            spot_price=sp,
            dividend_type=dt,
            div_amount=da,
            option_type=ot,
            start_date=start_date,
            valuation_date=valuation_date,
            american=am
        )
        for k, exp, s, ri, n, sp, dt, da, ot, start_date, valuation_date, am in zip(
            K, expiration, sigma, r, N, S, dividend_type, div_amount, option_type, start_date, valuation_date, american
        )
    ]
    price = np.array([model.price() for model in models])
    # price = []
    # for i, model in enumerate(models):
    #     try:
    #         print(f"Model {i}: K={model.K}, S={model.S0}, N={model.N}, T={model.T}, option_type={model.option_type}")
    #         price.append(model.price())
    #     except Exception as e:
    #         print(f"Error in model {i}: {e}")
    #         print(model.stock_tree)
    #         print(model, )
    #         raise
    price = np.array(price)

    return price, models

def binomial_tree_greeks(
    K: float|np.ndarray,
    expiration: float|np.ndarray,
    sigma: float|np.ndarray,
    r: float|np.ndarray,
    N: float|np.ndarray,
    S: float|np.ndarray,
    dividend_type: float|np.ndarray,
    div_amount: float|np.ndarray,
    option_type: float|np.ndarray,
    start_date: float|np.ndarray,
    valuation_date: float|np.ndarray,
    american: float|np.ndarray,
):
    """
    Calculate Greeks using a binomial tree model.
    
    Parameters:
    - K: Strike price
    - expiration: Expiration date of the option
    - sigma: Volatility of the underlying asset
    - r: Risk-free interest rate
    - N: Number of time steps in the binomial tree
    - spot_price: Current price of the underlying asset (optional)
    - dividend_type: Type of dividend ('discrete' or 'continuous')
    - div_amount: Amount of dividend (if applicable)
    - option_type: 'c' for call, 'p' for put
    - start_date: Start date for the option pricing (optional)
    - valuation_date: Date for which the option is priced (optional)
    
    Returns:
    Dictionary with calculated Greeks.
    """
    price, models = binomial_tree_price_batch(
        K=K,
        expiration=expiration,
        sigma=sigma,
        r=r,
        N=N,
        S=S,
        dividend_type=dividend_type,
        div_amount=div_amount,
        option_type=option_type,
        start_date=start_date,
        valuation_date=valuation_date,
        american=american
    ) 
    

    return {
        'delta': np.array([model.delta() for model in models]),
        'gamma': np.array([model.gamma() for model in models]),
        'vega': np.array([model.vega() for model in models]),
        'theta': np.array([model.theta() for model in models]),
        'rho': np.array([model.rho() for model in models]),
        'volga': np.array([model.volga() for model in models]),
        'model': np.array([model for model in models]),
    }