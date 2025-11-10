import pandas as pd
import numpy as np
from datetime import datetime
from trade.helpers.Logging import setup_logger
from .utils import _ManagerLazyLoader
from .shared_obj import get_request_list, setup_shared_objects
from trade.helpers.helper import generate_option_tick_new
from typing import List, Union

logger = setup_logger('DataManagers.Request.py')
setup_shared_objects()



def get_bulk_requests():
    """
    Returns the name of requests.
    """
    reqs = list(get_request_list())
    return [x for x in reqs if 'BULK' in x]

def get_chain_requests():
    """
    Returns the name of requests.
    """
    reqs = list(get_request_list())
    return [x for x in reqs if 'CHAIN' in x]


def get_single_requests():
    """
    Returns the name of requests.
    """
    reqs = list(get_request_list())
    return [x for x in reqs if 'SINGLE' in x]




def construct_request_name(
        start: str | datetime,
        end: str | datetime,
        tick: str,
        exp: str,
        type_: str = 'bulk',
        strike: str = None,
        right: str = None,
        **kwargs,
):

    """
    Constructs the request name based on the type of request.
    """
    if type_ == 'bulk':
        exp_str = pd.to_datetime(exp).strftime('%Y-%m-%d')
        start_str, end_str = pd.to_datetime(start).strftime('%Y-%m-%d'), pd.to_datetime(end).strftime('%Y-%m-%d')
        request_name = f"BULK_{tick}_{exp_str}_{start_str}_{end_str}_EOD"
    
    elif type_ == 'single':
        exp_str = pd.to_datetime(exp).strftime('%Y-%m-%d')
        start_str, end_str = pd.to_datetime(start).strftime('%Y-%m-%d'), pd.to_datetime(end).strftime('%Y-%m-%d')
        request_name = f"SINGLE_{tick}_{exp_str}_{start_str}_{end_str}_{strike}_EOD"

    elif type_ == 'chain':
        start_str, end_str = pd.to_datetime(start).strftime('%Y-%m-%d'), pd.to_datetime(end).strftime('%Y-%m-%d')
        request_name = f"CHAIN_{tick}_{start_str}_EOD"
    else:
        raise ValueError(f"Unknown type: {type_}, expected 'bulk' or 'chain'")

    return request_name

### Functions

def create_request_bulk(
        start: str | datetime,
        end: str | datetime,
        tick: str,
        exp: str,
        print_info: bool = False,
        type_: str = 'bulk',
        set_attributes: dict = {},
        _requests: list =None,
        **kwargs,
):

    current_request = datetime.now().strftime("%Y%m%d %H:%M:%S")
    missing_dates = [start, end]
    if type_ == 'bulk':
        dummy_request = BulkOptionQueryRequestParameter(
            table_name='temp_options_eod_new',
            db_name='securities_master',
            start_date=start,
            end_date=end,
            ticker=tick,
            exp=exp,
        )
        request_name = construct_request_name(start=start, end=end, tick=tick, exp=exp, type_=type_)
    
    elif type_ == 'single':
        strike = kwargs['strike']
        right = kwargs['right']
        dummy_request = OptionQueryRequestParameter(
            start_date=start,
            end_date=end,
            ticker=tick,
            exp=exp,
            strike=strike,
            right=right,
            table_name='temp_options_eod_new',
            db_name='securities_master',
        )
        request_name = construct_request_name(start=start, end=end, tick=tick, exp=exp, type_='single', strike=strike)

    elif type_ == 'chain':
        dummy_request = ChainDataRequest(
            symbol=tick,
            date=start,
            db_name='vol_surface',
            table_name='option_chain',
        )
        
        request_name = construct_request_name(start=start, end=end, tick=tick, exp=exp, type_='chain')
        dummy_request.exp = exp
    else:
        raise ValueError(f"Unknown type: {type_}, expected 'bulk' or 'chain'")

    for key, value in set_attributes.items():
        setattr(dummy_request, key, value)
    dummy_request.agg = 'eod'
    dummy_request.start_date = start
    dummy_request.end_date = end
    dummy_request.missing_dates = missing_dates
    lazy_loader = _ManagerLazyLoader(tick)
    lazy_loader.exp = dummy_request.exp
    dummy_request.input_params = lazy_loader.eod
    dummy_request.request_name = request_name
    if _requests:
        if request_name in _requests:
            logger.critical(f"Request {request_name} already exists, skipping save to database")
            return
        
    print("Printing info") if print_info else None
    _requests.append(request_name)
    return dummy_request


#### Request Class
class ChainDataRequest:
    def __init__(self, symbol, date, table_name, db_name, **kwargs):
        self.symbol = symbol
        self.date = date
        self.table_name = table_name
        self.db_name = db_name
        self.post_processed_data = pd.DataFrame()
        self.database_data = pd.DataFrame()
        self.is_empty = False
        self.organized_data = pd.DataFrame()
        self.input_params = None
        self.model = None
        self.ivl_str = None
        self.ivl_int = None
        self.default_fill = None
        self.agg = None
        self.requested_col = None
        self.iv_cols = None
        self.greek_cols = None
        self.col_kwargs = None
        self.pre_process = {}
        

class OptionQueryRequestParameter:
    def __init__(self, table_name, db_name, start_date=None, end_date=None, ticker=None, exp=None, strike=None, right = None, **kwargs):
        self.db_name = db_name
        self.table_name = table_name
        self.start_date = start_date
        self.end_date = end_date
        self.exp = exp
        self.strike  = strike
        self.right = right
        self.symbol = ticker
        self.opttick= generate_option_tick_new(ticker, right, exp, strike)
        self.query = None
        self.y = None
        self.vol = None
        self.spot = None
        self.interval = None
        self.type_ = None
        self.post_processed_data = pd.DataFrame()
        self.database_data = pd.DataFrame()
        self.is_empty = False
        self.organized_data = pd.DataFrame()
        self.missing_dates = []
        self.input_params = None
        self.model = None
        self.ivl_str = None
        self.ivl_int = None
        self.default_fill = None
        self.agg = None
        self.requested_col = None
        self.iv_cols = None
        self.greek_cols = None
        self.col_kwargs = None
        self.pre_process = {}


class BulkOptionQueryRequestParameter:
    def __init__(self, table_name, db_name, start_date=None, end_date=None, ticker=None, exp=None, strikes=None, **kwargs):
        self.db_name = db_name
        self.table_name = table_name
        self.start_date = start_date
        self.end_date = end_date
        self.ticker = ticker
        self.exp = exp
        self.strikes  = strikes
        self.opttick = None
        self.symbol = ticker
        self.query = None
        self.y = None
        self.vol = None
        self.spot = None
        self.interval = None
        self.type_ = None
        self.post_processed_data = pd.DataFrame()
        self.database_data = pd.DataFrame()
        self.is_empty = False
        self.organized_data = pd.DataFrame()
        self.missing_dates = []
        self.input_params = None
        self.model = None
        self.ivl_str = None
        self.ivl_int = None
        self.default_fill = None
        self.agg = None
        self.requested_col = None
        self.iv_cols = None
        self.greek_cols = None
        self.col_kwargs = None
        self.pre_process = {}

def create_kwargs_from_object(obj: Union[OptionQueryRequestParameter, 
                                        BulkOptionQueryRequestParameter, 
                                        ChainDataRequest]) -> dict:
    """
    Creates a dictionary of kwargs from the given object.
    """
    kwargs = {}
    if isinstance(obj, BulkOptionQueryRequestParameter):
        kwargs = dict(exp = obj.exp,
                      start = obj.start_date,
                      end = obj.end_date,
                      tick = obj.symbol,
                      type_ = 'bulk',
                      save_func = 'save_to_database',)
    elif isinstance(obj, OptionQueryRequestParameter):
        kwargs = dict(exp = obj.exp,
                      start = obj.start_date,
                      end = obj.end_date,
                      tick = obj.symbol,
                      type_ = 'single',
                      right = obj.right,
                      strike = obj.strike,
                      save_func = 'save_to_database',)
    elif isinstance(obj, ChainDataRequest):
        kwargs = dict(exp = obj.date,
                      start = obj.date,
                      end = obj.date,
                      tick = obj.symbol,
                      type_ = 'chain',
                      save_func = 'save_chain_data',
                      set_attributes = dict(post_processed_data = obj.post_processed_data.to_dict(orient='records')),)
    else:
        raise ValueError(f"Unknown object type: {type(obj)}")
    return kwargs