from typing import TypedDict
from enum import Enum
from abc import ABC, abstractmethod
from typing import ClassVar
from weakref import WeakSet
from trade.helpers.exception import SymbolChangeError

class OptionTickMetaData(TypedDict):
    ticker: str
    exp_date: str
    put_call: str
    strike: float
    
class PositionData(TypedDict): 
    long: list[str]
    short: list[str]



class OptionModelAttributes(Enum):
    S0 = 'unadjusted_S0'
    K = 'K'
    exp_date = 'exp'
    sigma = 'sigma'
    y = 'y'
    put_call = 'put_call'
    r = 'rf_rate'
    start = 'end_date'
    spot_type = 'chain_price'



class TickerMap(dict):
    invalid_tickers = {'FB': 'META'}
    def __getitem__(self, key):
        if key in self.invalid_tickers:
            raise SymbolChangeError(f"Tick name changed from {key} to {self.invalid_tickers[key]}, access the new tick instead")
        return super().__getitem__(key)


class SingletonMixin(ABC):
    """
    A mixin class to make a class a singleton by symbol.
    Still a work in progress.
    """

    _registry: ClassVar[WeakSet[type]] = WeakSet()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        SingletonMixin._registry.add(cls)


    @classmethod
    @abstractmethod
    def clear_instances(cls):
        pass

    @classmethod
    @abstractmethod
    def instances(cls):
        pass

    @classmethod
    def clear_all_instances(cls) -> None:
        for sub in list(cls._registry):
            if issubclass(sub, cls):
                try:
                    sub.clear_instances()
                except TypeError:
                    pass


class SingletonMetaClass(type):
    """
    A metaclass for singleton classes.
    It ensures that only one instance of a class is created.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]