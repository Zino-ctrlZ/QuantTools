
import pandas as pd
import numpy as np
from trade.assets.Stock import Stock
from trade.assets.rates import get_risk_free_rate_helper
from trade.helpers.Logging import setup_logger
from dbase.DataAPI.ThetaData import (resample)
from trade.helpers.pools import  parallel_apply
from trade.helpers.decorators import log_error, log_error_with_stack, log_time
from trade.helpers.types import OptionModelAttributes
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import BDay
from trade import PRICING_CONFIG

logger = setup_logger('DataManagers.utils.py')


class _ManagerLazyLoader:
    def __init__(self, symbol):
        self.symbol = symbol
        self.Stock = Stock(self.symbol, run_chain = False)
        self._eod = {}
        self._intra = {}

    def __reduce__(self):
        return (self.__class__, (self.symbol,))


    @property
    def eod(self):
        """
        Returns the end of day data
        """

        return EODData(self)
    
    @property
    def intra(self):
        """
        Returns the end of day data
        """
        return IntraData(self)


    def _lazy_load(self, load_name, **kwargs):
        ## Utilizing the lazy load function to load data on demand, and speed up initialization
        if load_name == 's0_close':

            ## Will use Kwargs to move between intra and EOD.
            kwargs.pop('intra_flag')
            return_item =  (self.Stock.spot(ts = True,
                                          ts_start = pd.to_datetime(self.exp) - relativedelta(years=5),
                                          ts_end =pd.to_datetime(self.exp) + relativedelta(years=5),
                                          **kwargs))
            return return_item
        
        elif load_name == 's0_chain':
            kwargs.pop('intra_flag')
            return_item =  (self.Stock.spot(ts = True,
                                            ts_start = pd.to_datetime(self.exp) - relativedelta(years=5),
                                            ts_end =pd.to_datetime(self.exp) + relativedelta(years=5),
                                            spot_type='chain_price',
                                            **kwargs))
            return return_item
            
        elif load_name == 'r':
            intra_flag = kwargs.get('intra_flag', False)
            r = (get_risk_free_rate_helper()['annualized'])
            if intra_flag:
                return resample(r, PRICING_CONFIG['INTRADAY_AGG'], {'risk_free_rate':'ffill'})
            else:
                return r
    
        elif load_name == 'r_name':
            intra_flag = kwargs.get('intra_flag', False)
            r = (get_risk_free_rate_helper()['name'])

            if intra_flag:
                return resample(r, PRICING_CONFIG['INTRADAY_AGG'], {'risk_free_rate':'ffill'})
            else:
                return r

        elif load_name == 'y':
            ## Get the dividend yield
            intra_flag = kwargs.get('intra_flag', False)
            y = (self.Stock.div_yield_history(start = pd.to_datetime(self.exp) - relativedelta(years=5)))

            if intra_flag:
                return resample(y, PRICING_CONFIG['INTRADAY_AGG'], {'dividend_yield':'ffill'})
            else:
                return y
        


### _ManagerLazyLoader Helpers
class IntraData(dict):
    def __init__(inner, parent):
        inner.parent = parent
        super().__init__()

    def __getitem__(inner, key): ## Custom getter for EOD Dict. To initialize the data, if not already done
        if key not in inner.parent._intra:
            if key not in ['s0_close', 's0_chain', 'r', 'y', 'r_name']:
                raise KeyError(f"{key} not in intra data, expected one of: ['s0_close', 's0_chain', 'r', 'y', 'r_name']")
            inner.parent._intra[key] = inner.parent._lazy_load(key, ts_timewidth = '5', ts_timeframe = 'minute', intra_flag = True)
        return inner.parent._intra[key]
    
    def __contains__(innner, key):
        return key in inner.parent._intra
    
    def __repr__(inner):
        return inner.parent._intra.__repr__()
    
    def __len__(inner):
        return len(inner.parent._intra)
    
    def keys(inner):
        return inner.parent._intra.keys()


class EODData(dict):
    def __init__(inner, parent): ## inner is the instance of the class, parent is the instance of the parent class
        inner.parent = parent
        super().__init__()

    def __getitem__(inner, key): ## Custom getter for EOD Dict. To initialize the data, if not already done
        if key not in inner.parent._eod:
            if key not in ['s0_close', 's0_chain', 'r', 'y', 'r_name']:
                raise KeyError(f"{key} not in eod data, expected one of: ['s0_close', 's0_chain', 'r', 'y', 'r_name]")
            inner.parent._eod[key] = inner.parent._lazy_load(key, intra_flag = False)
        return inner.parent._eod[key]
    
    def __contains__(innner, key):
        return key in inner.parent._eod
    
    def __repr__(inner):
        return inner.parent._eod.__repr__()
    
    def __len__(inner):
        return len(inner.parent._eod)
    
    def keys(inner):
        return inner.parent._eod.keys()

    
