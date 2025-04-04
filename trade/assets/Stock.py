# This file contains the Stock class which is used to handle operations related to stock data querying, manipulation and data provision. 

## To-Do:
## Find a news source that can provide news as at a date
## Same as for next earnings
## Add unadjusted prices to the stock class - Chain price
## Add timer to each and log it

from openbb import obb
import sys
import os
from dotenv import load_dotenv
load_dotenv()
# sys.path.extend(
#     [os.environ.get('WORK_DIR', ''), os.environ.get('DBASE_DIR', '')])
from dbase.DataAPI.ThetaData import (list_contracts)

# from trade.helpers.Configuration import Configuration
from trade.helpers.Configuration import ConfigProxy
Configuration = ConfigProxy()
from trade.helpers.Context import Context
import re
from dateutil.relativedelta import relativedelta
from trade.helpers.exception import OpenBBEmptyData, raise_tick_name_change
import numpy as np
import requests
import pandas as pd
from datetime import datetime
import robin_stocks as robin
from trade.helpers.parse import *
from trade.helpers.helper import *
from trade.helpers.openbb_helper import *
from trade.models.VolSurface import SurfaceLab
load_openBB()
import yfinance as yf
from trade.assets.rates import get_risk_free_rate_helper
from dbase.DataAPI.ThetaData import resample, retrieve_chain_bulk
from pandas.tseries.offsets import BDay
from dbase.database.SQLHelpers import DatabaseAdapter
from pathos.multiprocessing import ProcessingPool as Pool
from trade.helpers.helper import change_to_last_busday
from trade.assets.OptionChain import OptionChain
from threading import Thread, Lock
from trade.assets.helpers.utils import TICK_CHANGE_ALIAS, INVALID_TICKERS, verify_ticker
from trade.helpers.types import OptionModelAttributes
logger = setup_logger('trade.asset.Stock')



class Stock:
    rf_rate = None  # Initializing risk free rate
    rf_ts = None
    init_date = None
    dumas_width = 0.75
    _instances = {}


    def __new__(cls, ticker: str, **kwargs):
        # Use (ticker, end_date) as the unique key for singleton behavior
        today = datetime.today()
        _end_date = datetime.strftime(today, format='%Y-%m-%d')
        ## I don't need an instance per minute. I need an instance per day.
        ## I can update the end_date in __init__ if I need to

        end_date = Configuration.end_date or _end_date
        key = (verify_ticker(ticker).upper(), end_date)
        if key not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[key] = instance
        return cls._instances[key]



    def __repr__(self):
        return f'Stock(Ticker: {self.ticker}, Build Date: {self.end_date})'

    def __str__(self):
        return f'Stock(Ticker: {self.ticker}, Build Date: {self.end_date})'


    def __init__(self, ticker: str, **kwargs):
        """
        The Stock Class handles operations related to stock data querying, manipulation and data provision. 
        Type: Class
        Initial Vairables: 
        
        ticker: Stock Tick
        """

        ## Attr to be initalized every time
        today = datetime.today()
        ## Chaning to last business day, so far, there's no need to create Stock for. 
        ## Temp: Changing to last business day ensures that the data is available, +  avoids the missing error
        ## Note: Monitor if this is the best approach
        end_date = change_to_last_busday(today).strftime('%Y-%m-%d %H:%M:%S')
        self.__end_date = Configuration.end_date or end_date

        if not hasattr(self, '_intialized'):
            
            ## Attr to be initalized once
            self._intialized = True
            start_date_date = today - relativedelta(months=12)
            start_date = datetime.strftime(start_date_date, format='%Y-%m-%d %H:%M:%S')
            self.__security_obj = yf.Ticker(ticker.upper())
            self.__ticker = ticker.upper()
            self.timewidth = Configuration.timewidth or '1'
            self.timeframe = Configuration.timeframe or 'day'
            self.__start_date = Configuration.start_date or start_date
            self.__close = None
            self.prev_close()
            self.__y = None
            self.div_yield()
            self.__OptChain = None
            self.__chain = None
            self.asset_type = self.__security_obj.info['quoteType']
            self.kwargs = kwargs
            
            """ Opting to make stock class Risk Rate personal to the instance, because the end_date can change"""
            self.init_rfrate_ts()
            self.init_risk_free_rate()
            # self.__init_option_chain() 

        ## Logic to run chain, compatible with singleton behavior
        
        run_chain = kwargs.get('run_chain', False) ## Leave default as False for now)
        self.run_chain = run_chain
        if run_chain:
            if self.__OptChain is None:
                self.chain_init_thread = Thread(target = self.__init_option_chain, name = self.__repr__() + '_ChainInit')
                self.chain_init_thread.start()


    @property
    def is_chain_ready(self):
        return  not self.chain_init_thread.is_alive()
    @property
    def ticker(self):
        return self.__ticker
    
    @property
    def security_obj(self):
        return self.__security_obj
    
    @property
    def start_date(self):
        return self.__start_date
    
    @property
    def end_date(self):
        return self.__end_date
    
    @property
    def close(self):
        return self.__close

    @property
    def y(self):
        return self.__y

    @property
    def OptChain(self):
        if self.__OptChain is not None:
            return self.__OptChain
        else:
            logger.info(f'Chain not ready for {self.ticker} on {self.end_date}')
            print('Chain is not ready')
            return None
    
    @property
    def chain(self):
        return self.__chain
    
    @classmethod
    def clear_instances(cls, pop_all = True):
        if pop_all:
            cls._instances = {}
        else:
            if cls._instances:
                return cls._instances.popitem()
            
            return None
    
    @classmethod
    def list_instances(cls):
        return cls._instances
    
    def __set_close(self, value):
        self.__close = value


    def __set_y(self, value):
        self.__y = value


    def __init_option_chain(self):
        try:
            force_build = self.kwargs.get('force_build', False)
            logger.info(f'Initializing Option Chain for {self.ticker} on {self.end_date}')
            end_date = pd.to_datetime(self.end_date).strftime('%Y-%m-%d')
            chain = OptionChain(self.ticker, end_date, self, self.dumas_width, force_build = force_build)
            self.__OptChain = chain
            self.__chain = self.__OptChain.get_chain(True)
            self.OptChain.initiate_lab()
        except Exception as e:
            logger.error(f"Error initializing Option Chain for {self.ticker} on {self.end_date}: {e}")
            self.__OptChain = None
            self.__chain = None
            return None



    def init_rfrate_ts(self):
        """
        Class method that initiates Risk Free rate timeseries across all classes
        """
        ts = get_risk_free_rate_helper()
        self.rf_ts = ts


    def init_risk_free_rate(self):
        """
        Class method that initiates today's risk free rate across all classes
        """

        ts = self.rf_ts
        last_bus = change_to_last_busday(self.end_date)
        self.rf_rate = ts[ts.index == pd.to_datetime(last_bus).strftime('%Y-%m-%d')]['annualized'].values[0]

    
    def rebuild_chain(self):
        """
        Method to rebuild the Option Chain
        """
        self.__init_option_chain()
        self.run_chain = True


    def vol(self, exp, strike_type, strike, right):
        """
        Calculates implied volatility based on VolSurface modelling

        params:
        exp: Expressed in terms of intervals. Eg: 1D, 1W, 1M, 1Y
        strike_type: 'p' for percent of spot, 'k' for absolute strike
        strike: Corresponding strike value. Can be either a list, ndarray or float
        right: 'C' for call, 'P' for put

        returns:
        dict: {'k': strike, 'dumas': dumas vol, 'svi': svi vol}
        """
        exp = identify_length(*extract_numeric_value(exp))
        spot = list(self.spot(spot_type=OptionModelAttributes.spot_type.value).values())[0]

        ## Check if strike is a list, ndarray or float
        if isinstance(strike, list):
            strike = np.array(strike)
        elif isinstance(strike, float):
            strike = np.array([strike])

        elif isinstance(strike, np.ndarray):
            pass
        else:
            raise ValueError('Strike must be either a list, ndarray or float')
        

        ## Check if strike_type is either 'p' or 'k'. P is percent of spot, K is absolute strike
        if strike_type == 'p':
            strike_k = strike * spot
        else:
            strike_k = strike

        ## Check if Option Chain is ready, this is because the OptionChain initiation is done in a thread
        if not self.run_chain:
            logger.error('Option Chain not initialied. Use self.rebuild_chain() to initialize')
            return None

        if self.OptChain is None or self.OptChain.lab is None:
            raise ValueError('Option Chain not ready, please wait for it to be ready')
            return None

        vol = self.OptChain.lab.predict(exp, strike_k, right)
        
        try:
            if strike_type == 'p':
                vol['k'] = strike
            return vol
        except Exception as e:
            logger.error(f"Error calculating vol for exp={exp}, strike={strike}: {e}")
            return None


    def prev_close(self):
        """
        The prev_close returns the Previous days close, regardless of start time, end time, or configuration time
        """

        ## To-do: Ensure this is previous day close RELATIVE to the end date
        try:
           close = float(obb.equity.price.quote(symbol=self.ticker, provider='yfinance').to_dataframe()['prev_close'].values[0])
        except Exception as e:
            if 'Results not found'.lower() in e.__str__().lower():
                close = float(obb.equity.price.quote(symbol=self.ticker, provider='fmp').to_dataframe()['prev_close'].values[0])
                self.__set_close(close)
        return close


    def div_yield(self, div_type = 'yield'):
        div_history = self.div_yield_history(div_type = div_type)
        if not isinstance(div_history, pd.Series):
            if div_type == 'yield':
                self.__set_y(0)
            return 0
        else:
            end = change_to_last_busday(self.end_date).strftime('%Y-%m-%d')
            y = div_history[div_history.index == end][0]
            if div_type == 'yield':
                self.__set_y(y)
            return y
        


    def div_yield_history(self, start = None, ts_timeframe = 'day', ts_timewidth = '1', div_type = 'yield'):
        """
        Return the yearly dividend yield
        """
        if not start:
            start_date = self.start_date

        else:
            start_date = start


        try:
            div_history = obb.equity.fundamental.dividends(symbol=self.ticker, provider='yfinance').to_df()
        except:
            return 0
        

        
        div_history.rename(columns = {'ex_dividend_date':'date'}, inplace = True)
        div_history.date = pd.to_datetime(div_history.date)
        div_history.set_index('date', inplace = True)
        interval = identify_interval(ts_timewidth,ts_timeframe)

        ## Convert to daily, filling missing with 0
        div_history = div_history.asfreq('1D').fillna(0)

        ## Calculate yearly dividend using 365 rolling sum
        div_history["yearly_dividend"] = div_history.rolling(365).sum()
        div_history.dropna(inplace = True)

        if div_type == 'value':
            dates = pd.date_range(start = div_history.index.min(), end = datetime.today() ,freq = 'B')
            div_history = div_history.reindex(dates, method = 'ffill')
            return resample(div_history['yearly_dividend'], interval, {'yearly_dividend':'last'})['yearly_dividend']
        elif div_type == 'yield':
            pass
        else:
            raise ValueError("div_type must be either 'value' or 'yield'")

        ## Get Spot Price timeseries
        spot = self.spot(ts = True, ts_timeframe = ts_timeframe, ts_timewidth = ts_timewidth, ts_start = pd.to_datetime(start_date), ts_end = self.end_date)
        spot["Date"] = pd.to_datetime(spot.index.date)
        spot.reset_index(inplace = True)
        spot.set_index('Date', inplace = True)
        spot['yearly_dividend'] = div_history['yearly_dividend']
        spot.fillna(method = 'ffill', inplace = True)
        spot.set_index('date', inplace = True)
        ## Calculate Dividend Yield
        spot['dividend_yield'] = (spot['yearly_dividend']/spot['close'])
        div_yield = spot['dividend_yield']
        div_yield = resample(div_yield, interval, {'dividend_yield':'last'})
        return div_yield['dividend_yield'] if isinstance(div_yield, pd.DataFrame) else div_yield


    def spot(self,
        provider = 'yfinance', 
        spot = None, 
        ts: bool = False , 
        ts_start = None, 
        ts_end = None, 
        ts_timewidth = None, 
        ts_timeframe = None, 
        model_override = None,
        spot_type ='close'):
        """
        The spot method returns a dictionary holding the latest available close price for the ticker. It begins with the given or default end and start dates

        PARAMS
        ______
        ts (Bool): True to return dataframe timeseries, false to return spot in a dict
        provide (str): OpenBB Param. Available so far: 'yfinance', 'fmp'
        ts_start (str|datetime): Start Date
        ts_end (str|datetime): End Date
        ts_timewidth (str|int): Steps in timeframe
        ts_timeframe (str): Target timeframe for series 
        spot_type (str): Type of spot to return. Default is 'close'. Others are 'chain_price', 'unadjusted_close'

        RETURNS
        _________
        pd.DataFrame or dict
        """

        if spot_type not in ['close', 'chain_price', 'unadjusted_close']:
            raise ValueError("spot_type must be either 'close', 'chain_price', 'unadjusted_close'")
        
        if ts_timeframe is None:
            ts_timeframe = 'day'
        
        if ts_timewidth is None:
            ts_timewidth = self.timewidth
        
        if ts_start is None:
            ts_start = self.start_date
            ts_start = change_to_last_busday(ts_start)
        else:
            ts_start = pd.to_datetime(ts_start)
        
        if ts_end is None:
            ts_end = self.end_date
            ts_end = change_to_last_busday(ts_end)
        else:
            ts_end = pd.to_datetime(ts_end) + BDay(1)
        
        interval = identify_interval(ts_timewidth, ts_timeframe, provider)
        ts_start, ts_end = pd.to_datetime(ts_start).strftime('%Y-%m-%d'), pd.to_datetime(ts_end).strftime('%Y-%m-%d')
        
        if spot_type == 'chain_price':
                df = retrieve_timeseries(self.ticker, end =change_to_last_busday(datetime.today()).strftime('%Y-%m-%d'), 
                                         start = '1960-01-01', interval= '1d', provider = provider)
                df.index = pd.to_datetime(df.index)
                df = df[(df.index >= pd.Timestamp(ts_start)) & (df.index <= pd.Timestamp(ts_end))]
                df['close'] = df['chain_price']
        else:
            # print(ts_start, ts_end, interval, provider)
            df = retrieve_timeseries(self.ticker, end =ts_end, start = ts_start, interval= interval, provider = provider)

        if ts:
            df.index = pd.to_datetime(df.index)
            return df
        else:
            if pd.to_datetime(self.end_date).date() >= datetime.today().date() and self.asset_type not in ['ETF', 'MUTUALFUND', 'INDEX']:

                spot = {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):float(obb.equity.price.quote(symbol=self.ticker, provider='yfinance').to_dataframe()['last_price'].values[0])}
            else:
                end  = change_to_last_busday(ts_end)
                df.index = pd.to_datetime(df.index)
                df = df[df.index.date == pd.to_datetime(end).date()]
                spot = {end: df[spot_type].values[-1]}
            return spot
    
    # def option_chain(self, date = None, return_price = 'Midpoint'):
        # if not date:
        #     date = self.end_date
        # contracts = retrieve_chain_bulk(self.ticker, 0, date, date, '16:00')
        # contracts['DTE'] = (contracts['Expiration'] - pd.to_datetime(date)).dt.days
        # return contracts.pivot_table(index=['Expiration', 'DTE','Strike'], columns='Right',values = return_price, aggfunc=sum)

    def option_chain(self, date = None):
        if not date:
            date = self.end_date
        contracts = list_contracts(self.ticker, date)
        contracts.expiration = pd.to_datetime(contracts.expiration, format='%Y%m%d')
        contracts['DTE'] = (contracts['expiration'] - pd.to_datetime(date)).dt.days
        contracts_v2 = contracts.pivot_table(index=['expiration', 'DTE','strike'], columns='right',values = 'root', aggfunc='count')
        contracts_v2.fillna(0, inplace = True)
        contracts_v2.where(contracts_v2 == 1, False, inplace = True)
        contracts_v2.where(contracts_v2 == 0, True, inplace = True)
        return contracts_v2





    def returns(self, freq = '1B', ts_start = None, ts_end = None,  periods = None, log = False):
        """
        Calculates the daily, weekly, monthly and cumulative returns at any given timeframe. 
        Returns a dataframe with spot and return
        """
        ts_timeframe = 'day'
        ts_timewidth = '1'
        df = self.spot(ts = True, ts_end= ts_end, ts_start=ts_start, ts_timeframe=ts_timeframe, ts_timewidth=ts_timewidth)
        if periods:
            if not log:
                df['returns'] = df.close.pct_change(freq = freq, periods = periods)
            else:
                df['returns'] = np.log(df.close.pct_change(freq = freq, periods = periods) + 1)

        else:
            if log:
                df['returns'] = np.log(df.close.pct_change(freq = freq) + 1)
            else:
                df['returns'] = df.close.pct_change(freq = freq)
        df = df[['close', 'returns']]
        df = df[~df['returns'].isna()]

        # CUMULATIVE RETURNS FROM START DATE
        df['cumulative_returns'] = (df.returns + 1).cumprod()
        return df

    def rvol(self, window = 30, **kwargs):
        """
        Calculates the realized volatility of a stock based on the given time window

        Params
        --------
        window: int
            The time window to calculate the realized volatility. Only accepts integer values
            And must be in days
        
        **kwargs: Corresponds to Stock.returns() method args


        Returns a dataframe with specified realized vol
        """

        df = self.returns('1B', **kwargs)
        rolling_std = df.returns.rolling(window).std()
        annualized_std = rolling_std * np.sqrt(252)
        df[f'rvol_{window}D'] = annualized_std
        return df.dropna()

    def pct_spot_slides(self, pct_spot=[0.8, 0.9, 0.95, 1.05, 1.1, 1.2]):
        """
        Calculates Slide scenario based on provided lists of slides

        Returns a dictionary containing both PnL and price post shock
        """

        spot = list(self.spot().values())[0]
        PnL_dict = {}
        Spot = {}

        for slides in pct_spot:
            spot_slide_str = str(round((slides - 1)*100, 0))
            PnL = round(((spot*slides) - spot), 2)
            Spot_slide = round(spot*slides, 2)
            PnL_dict[spot_slide_str] = PnL
            Spot[spot_slide_str] = Spot_slide
        slides_dict = {'PnL': PnL_dict, 'Spot': Spot}
        return slides_dict

    def details(self):
        """
        The prev_close returns the latest close parameters, regardless of the start/end time.
        """
        info = self.security_obj.info

        print(f"Company Name: {info.get('longName', 'N/A')}")
        if info.get('marketCap', 'N/A') == 'N/A':
            print(f"Market Cap: N/A")
        else:
            cap = int(info.get('marketCap', 'N/A'))
            print(f"Market Cap: {cap/1_000_000_000:.2f}bn")
        print(f"PE Ratio: {info.get('trailingPE', 'N/A')}")
        print(f"Dividend Yield: {info.get('dividendYield', 'N/A')}")
        print(f"52 Week High: {info.get('fiftyTwoWeekHigh', 'N/A')}")
        print(f"52 Week Low: {info.get('fiftyTwoWeekLow', 'N/A')}")
        print(f"Industry: {info.get('industry', 'N/A')}")
        print(f"Sector: {info.get('sector', 'N/A')}")
        return(info)

    def news(self, breaker=5):
        """
       Prints top latest news based on the number designated to breaker variable
        """
        news = [{pd.to_datetime(x['providerPublishTime'], unit = 's') : x['title'] } for x in self.security_obj.news]
        links = [{pd.to_datetime(x['providerPublishTime'], unit = 's') : x['link'] } for x in self.security_obj.news]
        for nws in news:
            for date, title in nws.items():
                print(date.strftime('%Y-%m-%d'), ':', title)
        return links
        

    def get_next_earnings(self, robin_stocks_credentials=None):
        try:
            if robin_stocks_credentials is not None:
                login = robin.login(
                    robin_stocks_credentials['name'], robin_stocks_credentials['password'])
                totp = pyotp.TOTP(robin_stocks_credentials['auth']).now()

                # run login function
            quarters = [{"start": "01-01", "end": "03-31"}, {"start": "04-01", "end": "06-30"},
                        {"start": "07-01", "end": "09-30"}, {"start": "10-01", "end": "12-31"}]
            current_year = datetime.now().year
            current_month = datetime.now().month
            current_day = datetime.now().day
            quarter = {}
            if (current_month <= 3):
                quarter["period"] = 1
                quarter["date"] = quarters[0]
            elif (current_month <= 6 and current_month >= 4):
                quarter["period"] = 2
                quarter["date"] = quarters[1]
            elif (current_month <= 9 and current_month >= 7):
                quarter["period"] = 3
                quarter["date"] = quarters[2]
            else:
                quarter["period"] = 4
                quarter["date"] = quarters[3]

            earnings = robin.stocks.get_earnings(self.ticker)
            if earnings is None:
                raise ValueError(
                    "robin stocks returned no earnings, resolving to alpha vantage")
            for earning in earnings:
                # make sure earnings report date is within range of the current quarter
                if earning["year"] == current_year and earning["quarter"] == quarter["period"]:
                    if int(earning["report"]["date"].split('-')[1]) < int(quarter["date"]["start"].split('-')[0]) or int(earning["report"]["date"].split('-')[1]) > int(quarter["date"]["end"].split('-')[0]):
                        raise ValueError(
                            f"Earnings date is out of quarter range: {earning['report']} of quarter: {quarter}")
                    earning_report = earning["report"]
                    break

            if earning_report:
                date_split = earning_report["date"].split('-')
                days_to_earning = datetime(int(date_split[0]), int(date_split[1]), int(
                    date_split[2])) - datetime(int(current_year), int(current_month), int(current_day))
                return [earning_report["date"], f"{days_to_earning.days}"]
            else:
                return [364, ""]

        except Exception as e:
            # exception occured, resolved to alpha vantage api
            try:

                CSV_URL = f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&apikey={alpha_vantage_key}&symbol={self.ticker}&horizon=3month"
                decoded_data = requests.get(CSV_URL).content.decode('utf-8')
                pattern = r',(\d{4}-\d{2}-\d{2}),'
                matches = re.findall(pattern, decoded_data)
                current_date = datetime.now()
                start = f"{datetime.now().year}-01-01"
                end = f"{datetime.now().year}-12-31"
                difference = datetime.strptime(
                    end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")
                closest_earning_date = ""
                for match in matches:
                    matchDate = datetime.strptime(match, "%Y-%m-%d")
                    date_diff = matchDate - current_date
                    if (date_diff < difference and matchDate > current_date):
                        difference = date_diff
                        closest_earning_date = match

            except Exception as e:
                print('exception occured: ', e)
                return [364, ""]
            return [closest_earning_date, f"{difference.days}"]

    def get_earnings_ts(self):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.5414.120 Safari/537.36 CCleaner/109.0.19987.122'}
            url = f"https://finance.yahoo.com/calendar/earnings?symbol={self.ticker}"
            r = requests.get(url, headers=headers)
            soup = BeautifulSoup(r.text, 'html.parser')
            earning_dates_tds = soup.find_all(
                'td', {'aria-label': 'Earnings Date'})
            date_strings = [' '.join(span.stripped_strings)
                            for span in earning_dates_tds]
            date_strings = [date_str.replace(
                'EDT', 'EST') for date_str in date_strings]
            dates = pd.to_datetime(
                date_strings, format='%b %d, %Y, %I %p %Z', errors='coerce')

            # Define the market hours
            market_open = dates + pd.Timedelta('09:30:00')
            market_close = dates + pd.Timedelta('16:00:00')

            # Create DataFrame
            df = pd.DataFrame({'Date': dates, 'After Hours': dates > market_close, 'Before Hours': dates <
                              market_open, 'During': (dates >= market_open) & (dates <= market_close)})

            return df
        except Exception as e:
            print(e)
            return []

    def get_typical_price(self, start, end):
        data = self.spot(ts= True, ts_start=start, ts_end=end)
        data.columns = [x.capitalize() for x in data.columns]
        data['typical_money_flow'] = (
            data['High'] + data['Low'] + data['Close'])/3
        return data

    def get_raw_money_flow(self, start, end):
        data = self.get_typical_price(start, end)
        data['raw_money_flow'] = data['typical_money_flow'] * data['Volume']
        return data

    def get_typical_money_flow(self, start, end):
        data = self.get_raw_money_flow(start, end)
        money_flow = data['raw_money_flow']
        # create an array with positive or negative
        signal = np.where(money_flow > money_flow.shift(
            1), 1, np.where(money_flow < money_flow.shift(1), -1, 0))
        money_flow_s = money_flow * signal  # map money flow to sign
        data['typical_money_flow'] = money_flow_s
        return data

    def get_money_flow_ratio(self, start, end, window=2):
        data = self.get_typical_money_flow(start=start, end=end)
        money_flow_s = data['money_flow']
        money_flow_positive = money_flow_s.rolling(window).apply(
            lambda x: np.sum(np.where(x >= 0.0, x, 0.0)), raw=True)
        money_flow_negative = money_flow_s.rolling(window).apply(
            lambda x: np.sum(np.where(x < 0.0, x, 0.0)), raw=True)

        # Money flow index
        money_flow_ratio = money_flow_positive / money_flow_negative
        data['money_flow_ratio'] = money_flow_ratio
        return data

    def get_money_flow_index(self, start, end):
        data = self.get_money_flow_ratio(start, end, window=14)
        data['money_flow_index'] = 100. - (100./(1 + data['money_flow_ratio']))
        return data



