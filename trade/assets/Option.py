# This file contains the Option class which is used to handle operations related to Option data querying, manipulation and data provision.

from trade.helpers.helper import  change_to_last_busday
from trade.helpers.Configuration import Configuration
from trade.assets.Stock import Stock
from trade.helpers.helper import generate_option_tick, identify_interval
from threading import Thread
from dateutil.relativedelta import relativedelta
import pandas as pd
from datetime import date
import datetime as dt
from datetime import datetime
import yfinance as yf
from trade.assets.helpers.DataManagers import OptionDataManager


class Option:
    def __repr__(self):
        return f'Option(Strike: {self.K}, Expiration: {self.exp}, Underlier: {self.ticker}, Model: {self.model})'


    def __str__(self):
        return f'Option(Strike: {self.K}, Expiration {self.exp}, Underlier: {self.ticker}, Model: {self.model})'


    def __init__(self, 
    ticker: str,
    strike: float,
    exp_date: str | datetime,
    put_call: str,
    model: str = 'bt',
    default_fill = 'midpoint'):
        """
        The Option Class handles operations related to Option data querying, manipulation and data provision. 
        Type: Class
        Initial Vairables: 
        ticker = tick of the Stock being called
        strike = strike of the option contract
        exp_date = expiration date
        put_call = either a put or call
        model = prefered pricing model. 'bt' = Binomial Tree, 'bsm' = Black Scholes Model, 'mcs' = Monte Carlo Simulation
        """
        assert default_fill is not None, f"default_fill cannot be None"
        assert isinstance(strike, float), f'Strike should be a float, recieved {type(strike)}'
        today = datetime.today()
        start_date_date = today - relativedelta(years = 4)
        start_date = datetime.strftime(start_date_date, format='%Y-%m-%d')
        end_date = datetime.strftime(today, format='%Y-%m-%d')
        
        asset_obj = Stock(ticker)
        self.__asset = asset_obj
        self.__ticker = ticker.upper()
        self.timewidth = Configuration.timewidth or '1'
        self.timeframe = Configuration.timeframe or 'day'
        self.__start_date = Configuration.start_date or start_date
        self.__end_date = Configuration.end_date or end_date
        self.__security = yf.Ticker(ticker.upper())
        self.__S0 = self.__set__S0() # This should be current market spot for the underlier, should always be realtime.
        self.__OptTick = generate_option_tick(ticker, put_call, exp_date, strike)
        self.__prev_close = None # This should be previous DAY close of the OPTION
        self.__y = self.asset.y
        self.rf_rate = self.asset.rf_rate
        self.rf_ts = self.asset.rf_ts
        self.__default_fill = default_fill
        self.__dataManager = OptionDataManager(ticker, exp_date, put_call, strike, self.default_fill)

        if model in ['bt', 'bsm', 'mcs']:
            self.model = model
        else:
            raise ValueError(f"Available models are 'bsm', 'bt', 'mcs'. Recieved {model} ")

    
        self.K = strike
        if isinstance(exp_date, str):
            self.exp = exp_date
        elif isinstance(exp_date, (datetime, date)):
            self.exp = exp_date.strftime("%Y-%m-%d")
        else:
            raise Exception("Invalid type for exp_date. Valid types are 'str' and 'datetime'")

        if put_call.lower() in ['p', 'c']:
            p_c = put_call[0].upper()
            self.put_call = p_c.lower()
        else:
            raise Exception(f"Invalid option type. Please choose either 'p' or 'c'. Recieved {put_call}")
        
        self.__sigma = None
        Thread(target = self.__set_pv).start()
        self.__pv = None
        Thread(target = self.__set_pv).start()

    @property
    def asset(self):
        return self.__asset
    
    @property
    def start_date(self):
        return self.__start_date
    
    @property
    def end_date(self):
        return self.__end_date
    
    @property
    def security(self):
        return self.__security
    
    @property
    def OptTick(self):
        return self.__OptTick

    @property
    def ticker(self):
        return self.__ticker
    
    @property
    def prev_close(self):
        return self.__prev_close
    
    @property
    def y(self):
        return self.__y

    @property
    def S0(self):
        return self.__S0
    
    @property
    def default_fill(self):
        return self.__default_fill
    



    @property
    def sigma(self):
        """
        Retrieve Vol of the option. Using the default fill PV to calculate sigma

        This is intended to retrieve real time vol if end_date is set to today
        """
        self.__set_sigma()
        return self.__sigma

    
    def __set__S0(self):
        s0 = list(self.asset.spot(ts = False).values())[-1]
        return s0

    
    def __set_sigma(self):
        self.__sigma = self.vol()[f'{self.default_fill.capitalize()}_Vol'].values[0]

 


    @property
    def pv(self):
        """
        Retrieve PV of the option. Using the default fill as PV

        This is intended to retrieve real time PV if end_date is set to today

        """
        self.__set_pv()
        return self.__pv

    
    def __set_pv(self):
        self.__pv= self.spot()[f'{self.default_fill.capitalize()}'].values[0]




    def spot(self, 
    ts = False, 
    ts_start = None, 
    ts_end = None, 
    ts_timewidth = None, 
    ts_timeframe = None,
    spot_type = 'quote'):
        """
        The spot method returns a dataframe for latest quote price or a dictionary for last available close. 

        PARAMS
        ______
        ts (Bool): True to return dataframe timeseries, false to return spot in a dict
        ts_start (str|datetime): Start Date
        ts_end (str|datetime): End Date
        ts_timewidth (str|int): Steps in timeframe
        ts_timeframe (str): Target timeframe for series 
        
        RETURNS
        _________
        pd.DataFrame or dict
        """

        if ts_timeframe is None:
            if ts:
                ts_timeframe = 'day'
            else:
                ts_timeframe = 'minute'
        if ts_timewidth is None:
            ts_timewidth = self.timewidth
        if ts_start is None:
            ts_start = change_to_last_busday(self.start_date).strftime('%Y-%m-%d')
        if ts_end is None:
            ts_end = change_to_last_busday(self.end_date).strftime('%Y-%m-%d')
        interval = identify_interval(ts_timewidth, ts_timeframe)
        if self.model == 'mcs':
            model = 'bsm'
        else:
            model = self.model

        if ts:
            data = self.__dataManager.get_timeseries(ts_start, ts_end, interval, type_ = 'spot', model = model)
        else:
            data = self.__dataManager.get_spot(spot_type, query_date = ts_end)

        return data
    
    def vol(self, 
    ts = False, 
    ts_start = None, 
    ts_end = None, 
    ts_timewidth = None, 
    ts_timeframe = None):
        """
        The vol method returns a dataframe for latest quote price vol
        If ts is set to true, It returns the timeseries of vol based on the model.
        No current vol available for mcs model

        PARAMS
        ______
        ts (Bool): True to return dataframe timeseries, false to return spot in a dict
        ts_start (str|datetime): Start Date
        ts_end (str|datetime): End Date
        ts_timewidth (str|int): Steps in timeframe
        ts_timeframe (str): Target timeframe for series 
        

        RETURNS
        _________
        pd.DataFrame
        """

        if ts_timeframe is None:
            if ts:
                ts_timeframe = 'day'
            else:
                ts_timeframe = 'minute'
        if ts_timewidth is None:
            ts_timewidth = self.timewidth
        if ts_start is None:
            ts_start = change_to_last_busday(self.start_date).strftime('%Y-%m-%d')
        if ts_end is None:
            ts_end = change_to_last_busday(self.end_date).strftime('%Y-%m-%d')
        interval = identify_interval(ts_timewidth, ts_timeframe)
        
        if self.model == 'mcs':
            model = 'bsm'
        else:
            model = self.model

        if ts:
            data = self.__dataManager.get_timeseries(ts_start, ts_end, interval, type_ = 'vol', model = model)
        else:
            data = self.__dataManager.get_vol('quote', query_date = ts_end)

        return data


    def greeks(self, 
    greek_type = 'greek', 
    ts_start = None, 
    ts_end = None, 
    ts_timewidth = None, 
    ts_timeframe = None):
        """
        The greeks method returns a timeseries dataframe for greeks based. Only available for BSM model

        PARAMS
        ______
        ts (Bool): True to return dataframe timeseries, false to return spot in a dict
        ts_start (str|datetime): Start Date
        ts_end (str|datetime): End Date
        ts_timewidth (str|int): Steps in timeframe
        ts_timeframe (str): Target timeframe for series 
        greek_type (str): Type of greek to return. Default is 'greek'.
            'greek' returns all greek, while passing 'delta', 'gamma', 'theta', 'vega' returns only the specific greek
        

        RETURNS
        _________
        pd.DataFrame 
        """

        if ts_timeframe is None:
            ts_timeframe = 'day'
        if ts_timewidth is None:
            ts_timewidth = self.timewidth
        if ts_start is None:
            ts_start = self.start_date
        if ts_end is None:
            ts_end = pd.to_datetime(self.end_date ) + relativedelta(days = 1)
        interval = identify_interval(ts_timewidth, ts_timeframe)
        data = self.__dataManager.get_timeseries(ts_start, ts_end, interval, type_ = greek_type)
   

        return data