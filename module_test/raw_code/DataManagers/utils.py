
import pandas as pd
from trade.assets.Stock import Stock
from trade.assets.rates import get_risk_free_rate_helper
from trade.helpers.Logging import setup_logger
from dbase.DataAPI.ThetaData import (resample)
from dateutil.relativedelta import relativedelta
from trade import PRICING_CONFIG

logger = setup_logger('DataManagers.utils.py')

# Global MarketTimeseries instance for caching underlier data
# This allows _ManagerLazyLoader to use cached data instead of hitting APIs
_GLOBAL_MARKET_TIMESERIES = None

def set_global_market_timeseries(market_timeseries_instance):
    """
    Set the global MarketTimeseries instance for caching underlier data.
    
    Args:
        market_timeseries_instance: MarketTimeseries instance from EventDriven.riskmanager.market_data
    """
    global _GLOBAL_MARKET_TIMESERIES
    _GLOBAL_MARKET_TIMESERIES = market_timeseries_instance
    logger.info(f"Global MarketTimeseries instance set: {market_timeseries_instance}")

def get_global_market_timeseries():
    """
    Get the global MarketTimeseries instance.
    
    Returns:
        MarketTimeseries instance or None if not set
    """
    return _GLOBAL_MARKET_TIMESERIES

## Temp fix to override going thru my sql.
EMPTY_TIMESEIRES_TABLE = pd.DataFrame({'Open': {},
 'High': {},
 'Low': {},
 'Close': {},
 'Volume': {},
 'Bid_size': {},
 'CloseBid': {},
 'Ask_size': {},
 'CloseAsk': {},
 'Strike': {},
 'Expiration': {},
 'Put/Call': {},
 'Underlier_price': {},
 'RF_rate': {},
 'RF_rate_name': {},
 'dividend': {},
 'OptionTick': {},
 'Underlier': {},
 'Datetime': {},
 'BS_IV': {},
 'Binomial_IV': {},
 'Delta': {},
 'Gamma': {},
 'Vega': {},
 'Theta': {},
 'Rho': {},
 'Vanna': {},
 'Volga': {},
 'Dollar_Delta': {},
 'midpoint': {},
 'midpoint_BS_IV': {},
 'midpoint_Binomial_IV': {},
 'midpoint_Delta': {},
 'midpoint_Gamma': {},
 'midpoint_Vega': {},
 'midpoint_Theta': {},
 'midpoint_Rho': {},
 'midpoint_Vanna': {},
 'midpoint_Volga': {},
 'midpoint_Dollar_Delta': {},
 'weighted_midpoint': {},
 'weighted_midpoint_BS_IV': {},
 'weighted_midpoint_Binomial_IV': {},
 'weighted_midpoint_Delta': {},
 'weighted_midpoint_Gamma': {},
 'weighted_midpoint_Vega': {},
 'weighted_midpoint_Theta': {},
 'weighted_midpoint_Rho': {},
 'weighted_midpoint_Vanna': {},
 'weighted_midpoint_Volga': {},
 'weighted_midpoint_Dollar_Delta': {},
 'OpenInterest': {},
 'bid_bs_iv': {},
 'ask_bs_iv': {},
 'bid_binomial_iv': {},
 'ask_binomial_iv': {},
 'midpoint_binomial_gamma': {},
 'midpoint_binomial_vega': {},
 'midpoint_binomial_delta': {},
 'midpoint_binomial_rho': {},
 'midpoint_binomial_vanna': {},
 'midpoint_binomial_volga': {},
 'midpoint_binomial_theta': {},
 'last_updated': {},
 'binomial_delta': {},
 'binomial_gamma': {},
 'binomial_vega': {},
 'binomial_volga': {},
 'binomial_vanna': {},
 'binomial_rho': {},
 'binomial_theta': {},
 'midpoint_bs_vol_resolve': {},
 'midpoint_binomial_vol_resolve': {},
 'ask_binomial_delta': {},
 'ask_binomial_gamma': {},
 'ask_binomial_vega': {},
 'ask_binomial_rho': {},
 'ask_binomial_theta': {},
 'ask_binomial_vanna': {},
 'ask_binomial_volga': {},
 'bid_binomial_delta': {},
 'bid_binomial_gamma': {},
 'bid_binomial_vega': {},
 'bid_binomial_rho': {},
 'bid_binomial_theta': {},
 'bid_binomial_vanna': {},
 'bid_binomial_volga': {}})
EMPTY_TIMESEIRES_TABLE.columns = [col.lower() for col in EMPTY_TIMESEIRES_TABLE.columns]

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
        market_ts = get_global_market_timeseries()
        
        if load_name == 's0_close':
            ## Will use Kwargs to move between intra and EOD.
            intra_flag = kwargs.pop('intra_flag')
            
            # Try to get from MarketTimeseries cache first
            if market_ts is not None:
                try:
                    interval = '5min' if intra_flag else '1d'
                    logger.debug(f"Attempting to load s0_close for {self.symbol} from MarketTimeseries cache")
                    
                    # Get data from MarketTimeseries (factor='spot')
                    result = market_ts.get_timeseries(
                        sym=self.symbol,
                        factor='spot',
                        interval=interval
                    )
                    if result and result.spot is not None and not result.spot.empty:
                        logger.info(f"✓ Loaded s0_close for {self.symbol} from MarketTimeseries cache ({len(result.spot)} rows)")
                        return result.spot
                except Exception as e:
                    logger.debug(f"MarketTimeseries cache miss for s0_close: {e}, falling back to Stock.spot()")
            
            # Fall back to Stock.spot() if cache miss
            return_item =  (self.Stock.spot(ts = True,
                                          ts_start = pd.to_datetime(self.exp) - relativedelta(years=5),
                                          ts_end =pd.to_datetime(self.exp) + relativedelta(years=5),
                                          **kwargs))
            return return_item
        
        elif load_name == 's0_chain':
            intra_flag = kwargs.pop('intra_flag')
            
            # Try to get from MarketTimeseries cache first (chain_spot type)
            if market_ts is not None:
                try:
                    interval = '5min' if intra_flag else '1d'
                    logger.debug(f"Attempting to load s0_chain for {self.symbol} from MarketTimeseries cache")
                    
                    # Get chain_spot data from MarketTimeseries
                    result = market_ts.get_timeseries(
                        sym=self.symbol,
                        factor='chain_spot',
                        interval=interval
                    )
                    if result and result.chain_spot is not None and not result.chain_spot.empty:
                        logger.info(f"✓ Loaded s0_chain for {self.symbol} from MarketTimeseries cache ({len(result.chain_spot)} rows)")
                        return result.chain_spot
                except Exception as e:
                    logger.debug(f"MarketTimeseries cache miss for s0_chain: {e}, falling back to Stock.spot()")
            
            # Fall back to Stock.spot()
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
            
            # Try to get dividends from MarketTimeseries cache first
            if market_ts is not None:
                try:
                    interval = '5min' if intra_flag else '1d'
                    logger.debug(f"Attempting to load dividends for {self.symbol} from MarketTimeseries cache")
                    
                    # Get dividends data from MarketTimeseries
                    result = market_ts.get_timeseries(
                        sym=self.symbol,
                        factor='dividends',
                        interval=interval
                    )
                    if result and result.dividends is not None and not result.dividends.empty:
                        logger.info(f"✓ Loaded dividends for {self.symbol} from MarketTimeseries cache ({len(result.dividends)} rows)")
                        return result.dividends
                except Exception as e:
                    logger.debug(f"MarketTimeseries cache miss for dividends: {e}, falling back to Stock.div_yield_history()")
            
            # Fall back to Stock.div_yield_history()
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
    
    def __contains__(inner, key):
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
    
    def __contains__(inner, key):
        return key in inner.parent._eod
    
    def __repr__(inner):
        return inner.parent._eod.__repr__()
    
    def __len__(inner):
        return len(inner.parent._eod)
    
    def keys(inner):
        return inner.parent._eod.keys()

    
