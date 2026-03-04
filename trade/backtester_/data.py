import pandas as pd
from backtesting import Backtest
from typing import Optional
class PTDataset:
    """
    Custom dataset holding ticker name, ticker timeseries & backtest object from backtesting.py

    """

    # name = None
    # data = None
    # backtest = None
    # cash = None
    def __init__(self, name: str, data: pd.DataFrame, param_settings: dict = None):
        self.__param_settings = param_settings  ## Making param_settings private
        self.name = name
        self.data = data
        self.backtest: Optional[Backtest] = None

    def __repr__(self):
        return f"PTDataset({self.name})"

    def __str__(self):
        return f"PTDataset({self.name})"

    @property
    def param_settings(self):
        """Getter for param settings"""
        return self.__param_settings

    @param_settings.setter
    def param_settings(self, value: dict):
        """Setter for param_settings with type checking"""
        if not isinstance(value, dict):
            raise TypeError("param_settings must be a dictionary")
        self.__param_settings = value
