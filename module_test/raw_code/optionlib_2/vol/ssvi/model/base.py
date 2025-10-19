from abc import ABC, abstractmethod
import pandas as pd

class ChainInputModel(ABC):
    """
    Abstract base class for option chain input models.
    """
    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def get_chain(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def build_chain(self) -> pd.DataFrame:
        pass


class BaseSSVIModel(ABC):

    @abstractmethod
    def predict(self, *args, **kwargs):
        """
        Abstract method to predict the implied volatility surface.
        Must be implemented by subclasses.
        """
        pass


    @abstractmethod
    def fit(self, *args, **kwargs):
        """
        Abstract method to fit the SSVI model.
        Must be implemented by subclasses.
        """
        pass