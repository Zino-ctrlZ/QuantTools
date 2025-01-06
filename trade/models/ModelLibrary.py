import os, sys
import numpy as np
import math
from scipy.stats import norm
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.environ['WORK_DIR'])
sys.path.append(os.environ['DBASE_DIR'])
from trade.helpers.Logging import setup_logger

logger = setup_logger("ModelLibrary.py")

class ModelLibrary:
    @staticmethod
    def MSE_IV_SVIJW(params, T, S0, K, IV_):
        """
        Calculate the mean squared error (MSE) for implied volatility (IV) using SVI-JW.

        Arguments:
        - params: Parameters to be optimized.
        - T: Array of time to maturities.
        - S0: Spot price.
        - K: Array of strike prices.
        - IV_: Observed implied volatilities.
        
        Returns:
        - MSE_IV: Mean squared error between model IV and observed IV.
        """
        MSE_IV = 0
        median_iv = np.mean(IV_*100)
        
        alpha = 10/np.std(np.log(np.array(K)/S0))**2
        for i in range(len(T)):
            
            # Compute model implied volatility using SVI-JW
            svijw_iv = (ModelLibrary.TotalVarSVIJW(S0, K[i], T[i], *params) / T[i])**0.5
            # w = (1/(1+(np.log(K[i]/S0)**2 * alpha)))
            w = abs(1/(median_iv*100 - IV_[i]*100))
            # print(w, K[i])
            # Add the squared error
            MSE_IV += (svijw_iv - IV_[i])**2 #* w
        # Return the mean squared error
        return (MSE_IV / len(T))

    @staticmethod
    def SVIVariables(S0, K, t, v, psi, p, c, v_tilde): 
        """
        Return the SVI-JW parameters. All integers

        Arguments:

        - S0: Spot price.
        - K: Strike price.
        - t: Time to maturity.
        - v: ATM volatility.
        - psi: Skew of the ATM volatility.
        - p: Slope of the left wing.
        - c: Slope of the right wing.
        - v_tilde: Minimum volatility.

        Returns:
        - a: Parameter a.
        - b: Parameter b.
        - rho: Parameter rho.
        - m: Parameter m.
        - sigma: Parameter sigma.
        
        """
        k = np.log(K / S0)
        w= v * t
        b = np.sqrt(w) / 2 * (c + p)
        rho = 1 - p * np.sqrt(w) / b
        #beta has to be in [­1, +1]
        beta = rho - 2 * psi * np.sqrt(w) / b
        alpha = np.sign(beta) * np.sqrt(1 / beta**2 - 1)
        m = (v - v_tilde) * t / (b * (- rho + np.sign(alpha) * np.sqrt(1 + alpha**2) - alpha * np.sqrt(1 - rho**2)))
        sigma = alpha * m
        a = v_tilde * t - b * sigma * np.sqrt(1 - rho**2)
        if (m == 0): sigma = (w - a) / b
        # print(a, b, rho, sigma, m)
        return (a, b, rho, sigma, m )
    
    @staticmethod
    def TotalVarSVIJW(S0, K, t, v, psi, p, c, v_tilde):
        """
        
        Calculate the total variance using the SVI-JW model.
    
        Arguments:

        - S0: Spot price.
        - K: Strike price.
        - t: Time to maturity.
        - v: ATM volatility.
        - psi: Skew of the ATM volatility.
        - p: Slope of the left wing.
        - c: Slope of the right wing.
        - v_tilde: Minimum volatility.

        Returns:
        - a: Parameter a.
        - b: Parameter b.
        - rho: Parameter rho.
        - m: Parameter m.
        - sigma: Parameter sigma.
        
        """
        k = np.log(K / S0)
        w= v * t
        b = np.sqrt(w) / 2 * (c + p)
        rho = 1 - p * np.sqrt(w) / b
        #beta has to be in [­1, +1]
        beta = rho - 2 * psi * np.sqrt(w) / b
        alpha = np.sign(beta) * np.sqrt(1 / beta**2 - 1)
        m = (v - v_tilde) * t / (b * (- rho + np.sign(alpha) * np.sqrt(1 + alpha**2) - alpha * np.sqrt(1 - rho**2)))
        sigma = alpha * m
        a = v_tilde * t - b * sigma * np.sqrt(1 - rho**2)
        if (m == 0): sigma = (w - a) / b
        # print(locals())
        # print(v, psi, p, c, v_tilde)
        return (a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2)))
    
    # Black Scholes Call Price
    @staticmethod
    def CallPrice(S, sigma, K, T, r, q):
        """
        
        Calculate the price of a call option using the Black-Scholes model.

        Arguments:
        - S: Spot price.
        - sigma: Volatility.
        - K: Strike price.
        - T: Time to maturity.
        - r: Risk-free rate.
        - q: Dividend yield.

        """


        d1 = (math.log(S / K) + (r + .5 * sigma**2) * T) / (sigma * T**.5) 
        d2 = d1 - sigma * T**0.5
        n1 = norm.cdf(d1)
        n2 = norm.cdf(d2)
        DF = math.exp(-r * T)
        y = math.exp(-q * T)
        price = S * y * n1 - K * DF * n2
        return price

    @staticmethod
    def PutPrice(S, sigma, K, T, r, q):
        """
        Calculate the price of a put option using the Black-Scholes model.
        
        Arguments:
        - S: Spot price.
        - sigma: Volatility.
        - K: Strike price.
        - T: Time to maturity.
        - r: Risk-free rate.
        - q: Dividend yield.
        
        """
        d1 = (math.log(S / K) + (r + .5 * sigma**2) * T) / (sigma * T**.5) 
        d2 = d1 - sigma * T**0.5
        n1 = norm.cdf(-d1)
        n2 = norm.cdf(-d2)
        DF = math.exp(-r * T)
        y = math.exp(-q * T)
        price = K * DF * n2 - S * y * n1 
        return price
    
    @staticmethod
    def VolSVI(S0, K, T, a, b, rho, sigma, m):
        """
        Calculate the volatility using the SVI-JW model.

        Arguments:
        - S0: Spot price.
        - K: Strike price.
        - T: Time to maturity.
        - a: Parameter a.
        - b: Parameter b.
        - rho: Parameter rho.
        - sigma: Parameter sigma.
        - m: Parameter m.
            
        """
        k = np.log(K / S0)
        v = np.sqrt((a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))) / T)
        # print(locals())
        return v

