"""
Stochastic Volatility Inspired (SSVI) model implementation.
This module provides tools to fit and use the SSVI model for option pricing and volatility surface modeling.
References:
- Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility parameterization with application to the valuation of volatility derivatives. 
  In W. Schoutens (Ed.), Asset and Liability Management Tools (pp. 117-134). Risk Books.
- Gatheral, J., & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. Quantitative Finance, 14(1), 59-71.
- Hendriks, R., & Martini, C. (2019). The SSVI volatility surface: Theory and practice. Wilmott Magazine, 2019(89), 70-81.

This module includes:
- _SSVIModel: A class to represent and fit the SSVI model to market data.
"""