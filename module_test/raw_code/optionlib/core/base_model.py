from abc import ABC, abstractmethod

class OptionModel(ABC):
    """
    Abstract base class for option models.
    This class defines the interface for option pricing models.
    """

    @abstractmethod
    def price(self, *args, **kwargs):
        """
        Calculate the price of an option.
        
        Args:
            *args: Positional arguments for the pricing model.
            **kwargs: Keyword arguments for the pricing model.
        
        Returns:
            float: The calculated option price.
        """
        pass

    @abstractmethod
    def vol(self, *args, **kwargs):
        """
        Calculate the implied volatility of an option.
        
        Args:
            *args: Positional arguments for the volatility model.
            **kwargs: Keyword arguments for the volatility model.
        
        Returns:
            float: The calculated implied volatility.
        """
        pass


def forward_price_continous(spot_price, interest_rate, time_to_maturity):
    """
    Calculate the forward price of an asset.
    
    Args:
        spot_price (float): The current spot price of the asset.
        interest_rate (float): The risk-free interest rate (annualized).
        time_to_maturity (float): Time to maturity in years.
        
    Returns:
        float: The calculated forward price.
    """
    return spot_price * (1 + interest_rate) ** time_to_maturity

def forward_price_discrete(spot_price, interest_rate, dividend_schedule):
    pass