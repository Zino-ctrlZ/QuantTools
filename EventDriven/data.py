#This handles how market data is sourced and drip feeds the event loop
from typing import Optional
from dotenv import load_dotenv
load_dotenv()
import datetime
import os, os.path
import pandas as pd
import sys
sys.path.append(
    os.environ.get('WORK_DIR')) #type: ignore
sys.path.append(
    os.environ.get('DBASE_DIR')) #type: ignore
from dbase.database.SQLHelpers import query_database # type: ignore
from dbase.DataAPI.ThetaData import retrieve_option_ohlc # type: ignore
from abc import ABCMeta, abstractmethod

from EventDriven.event import MarketEvent





class DataHandler(object):
    """
    DataHandler is an abstract base class providing an interface for
    all subsequent (inherited) data handlers (both live and historic).

    The goal of a (derived) DataHandler object is to output a generated
    set of bars (OLHCVI) for each symbol requested. 

    This will replicate how a live strategy would function as current
    market data would be sent "down the pipe". Thus a historic and live
    system will be treated identically by the rest of the backtesting suite.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars from the latest_symbol list,
        or fewer if less bars are available.
        """
        raise NotImplementedError("Should implement get_latest_bars()")

    @abstractmethod
    def update_bars(self):
        """
        Pushes the latest bar to the latest symbol structure
        for all symbols in the symbol list.
        """
        raise NotImplementedError("Should implement update_bars()")
    
    
class HistoricDataFrameDataHandler(DataHandler): 
    """
    HistoricDataFrameDataHandler is designed to read data from a mysql database 
    for each requested symbol and return the latest bar of each symbol.
    """
    
    def __init__(self, events, db_name, db_table, symbol_list):
        """
        Initialises the historic data handler by requesting the latest bars from the database 
        and returns a pandas dataframe with the lates bars for each symbol.
        """
        
        self.events = events
        self.db_name = db_name
        self.symbol_list = symbol_list
        self.db_table = db_table
        self.latest_symbol_data = {}
        self.continue_backtest = True
        self.symbol_data = {}
        self.symbol_list = symbol_list
        self.open_symbol_data()
        
    
    def open_symbol_data(self):
        """
        Queries the database for all entries based on symbol list and returns a panda dataframe 
        """
        symbol_list = self.symbol_list.copy()
        symbol_list_str = f"'{symbol_list.pop(0)}'"
        for symbol in symbol_list:
            symbol_list_str += f",'{symbol}'"
        query = f"SELECT * FROM {self.db_table} WHERE Underlier IN ({symbol_list_str})"
        db_table = query_database(self.db_name,query)
        self.data_raw = db_table
        db_table['Datetime'] = pd.to_datetime(db_table['Datetime'])
        for symbol in self.symbol_list:
            self.symbol_data[symbol] = db_table[db_table['Underlier'] == symbol]
            self.latest_symbol_data[symbol] = pd.DataFrame(columns=db_table.columns)
        
    def _get_new_bar(self, symbol):
        """
        Returns the latest bar from the data feed
        """
            
        while not self.symbol_data[symbol].empty: 
            bar = self.symbol_data[symbol].iloc[0]
            self.symbol_data[symbol] = self.symbol_data[symbol].iloc[1:]
            yield bar

    def get_latest_bars(self, symbol, N=1):
        """
         Returns the last N bars from the latest_symbol list,
        """
        try: 
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
        else:
            return bars_list.tail(N)
        
    def update_bars(self):
        """
        Pushes the latest bar to the latest_symbol_data structure
        for all symbols in the symbol list.
        """
        for s in self.symbol_list:
            try: 
                bar_generator = self._get_new_bar(s)
                #Get next bar from the generator

                ## Question: This is solely curiousity, but why is the next function called on the generator object?
                ## Like why not just return a row. It doesnt seem like there's another next call and generator looks to always return one row at a time? I may be wrong
                bar = next(bar_generator)
            except StopIteration:
                print(f"No more data available for symbol: {s}")
                self.continue_backtest = False
            else: 
                if bar is not None:
                    bar_df = pd.DataFrame([bar])
                    self.latest_symbol_data[s] = pd.concat([self.latest_symbol_data[s], bar_df], axis=0)
        self.events.put(MarketEvent())
        
        
        
    

class HistoricCSVDataHandler(DataHandler):
    """
    HistoricCSVDataHandler is designed to read CSV files for
    each requested symbol from disk and provide an interface
    to obtain the "latest" bar in a manner identical to a live
    trading interface. 
    """

    def __init__(self, events, csv_dir, symbol_list):
        """
        Initialises the historic data handler by requesting
        the location of the CSV files and a list of symbols.

        It will be assumed that all files are of the form
        'symbol.csv', where symbol is a string in the list.

        Parameters:
        events - The Event Queue.
        csv_dir - Absolute directory path to the CSV files.
        symbol_list - A list of symbol strings.
        """
        self.events = events
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list

        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True       

        self._open_convert_csv_files()
        
    
    def _open_convert_csv_files(self):
        """
        Opens the CSV files from the data directory, converting
        them into pandas DataFrames within a symbol dictionary.

        For this handler it will be assumed that the data is
        taken from Yahoo. Thus its format will be respected.
        """
        comb_index = None
        for s in self.symbol_list:
            # Load the CSV file with no header information, indexed on date
            self.symbol_data[s] = pd.read_csv(
                os.path.join(self.csv_dir, '%s.csv' % s),
                header=0, index_col=0, parse_dates=True,
                names=[
                    'datetime', 'open', 'high', 
                    'low', 'close', 'adj_close', 'volume'
                ]
            )
            self.symbol_data[s].sort_index(inplace=True)

            # Combine the index to pad forward values
            if comb_index is None:
                comb_index = self.symbol_data[s].index
            else:
                comb_index.union(self.symbol_data[s].index)

            # Set the latest symbol_data to None
            self.latest_symbol_data[s] = []

        for s in self.symbol_list:
            self.symbol_data[s] = self.symbol_data[s].reindex(
                index=comb_index, method='pad'
            )
            self.symbol_data[s]["returns"] = self.symbol_data[s]["adj_close"].pct_change().dropna()
            self.symbol_data[s] = self.symbol_data[s].iterrows()

        # Reindex the dataframes
        for s in self.symbol_list:
            self.symbol_data[s] = self.symbol_data[s].reindex(index=comb_index, method='pad').iterrows()
            
    
    def _get_new_bar(self, symbol):
        """
        Returns the latest bar from the data feed as a tuple of 
        (sybmbol, datetime, open, low, high, close, volume).
        """
        for b in self.symbol_data[symbol]:
            yield tuple([symbol, datetime.datetime.strptime(b[0], '%Y-%m-%d %H:%M:%S'), 
                        b[1][0], b[1][1], b[1][2], b[1][3], b[1][4]])

    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars from the latest_symbol list,
        or N-k if less available.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
        else:
            return bars_list[-N:]        
        
    def update_bars(self):
        """
        Pushes the latest bar to the latest_symbol_data structure
        for all symbols in the symbol list.
        """
        for s in self.symbol_list:
            try:
                bar = self._get_new_bar(s).next() # type: ignore
            except StopIteration:
                self.continue_backtest = False
            else:
                if bar is not None:
                    self.latest_symbol_data[s].append(bar)
        self.events.put(MarketEvent())
        

class HistoricTradeDataHandler(DataHandler): 
    """
    HistoricTradeDataHandler is designed to read from a pandas dataframe with trades data, 
    convert that to signals of 1 (for buy), -1 (for sell), 0 (for do nothing)
    """
    
    def __init__(self, events, trades_df): 
        self.trades_df = trades_df
        self.continue_backtest = True 
        self.events = events
        self._open_trade_data()
        self.options_data = {}
        
    def _open_trade_data(self): 
        unique_tickers = self.trades_df['Ticker'].unique()
        self.symbol_list = unique_tickers
        self.trades_df['EntryTime'] = pd.to_datetime(self.trades_df['EntryTime'])
        self.trades_df['ExitTime'] = pd.to_datetime(self.trades_df['ExitTime'])
        
        self.start_date = self.trades_df['EntryTime'].min()
        self.end_date = self.trades_df['ExitTime'].max()
        date_range = pd.date_range(start=self.start_date, end=self.end_date)
        #initialize signal dataframe
        self.signal_df = pd.DataFrame({'Date': date_range})
        
        for ticker in unique_tickers: 
            self.signal_df[ticker] = 0   
            
        #populate signal dataframe
        for _, row in self.trades_df.iterrows():
            entry_time = row['EntryTime']
            exit_time = row['ExitTime']
            ticker = row['Ticker']
            size = row['Size']
            #size in positive is for long positions whilenegative size is for short positions
            self.signal_df.loc[(self.signal_df['Date'] == entry_time) & (size > 0), ticker] = 1 
            self.signal_df.loc[(self.signal_df['Date'] == entry_time) & (size < 0), ticker] = 2
            self.signal_df.loc[self.signal_df['Date'] == exit_time, ticker] = -1 
        
        signal_columns = ['Date'].append(unique_tickers)
        self.latest_signal_df = pd.DataFrame(columns=signal_columns)
            
    def _get_new_bar(self): 
        """
        Return latest bar from data feed
        """
        while not self.signal_df.empty: 
            bar = self.signal_df.iloc[0]
            self.signal_df = self.signal_df.iloc[1:]
            yield bar
            
    def get_latest_bars(self, symbol ='', N=1) -> pd.DataFrame:
        return self.latest_signal_df.tail(N)
    
    def update_bars(self) -> Optional[bool]:
        try: 
            bar_generator = self._get_new_bar()
            #Get next bar from the generator
            bar = next(bar_generator)
        except StopIteration:
            self.continue_backtest = False
        else: 
            if bar is not None:
                bar_df = pd.DataFrame([bar])
            self.latest_signal_df = pd.concat([self.latest_signal_df, bar_df], axis=0)
            self.events.put(MarketEvent())
        
        return self.continue_backtest
        
    def update_options_data_on_order(self, contract): 
        """
        Updates the option data based on the fill contract
        """
        if contract is not None:
            option_id = self.get_option_id(contract)
            if option_id not in self.options_data: 
                start_date = self.start_date.strftime('%Y%m%d')
                end_date = self.end_date.strftime('%Y%m%d')
                exp = f'{contract["expiration"]}'
                strike = contract['strike']
                options = retrieve_option_ohlc(symbol = contract['root'], exp = exp, strike= strike, right=contract["right"], start_date=start_date, end_date=end_date)
                if options is not None: 
                    self.options_data[option_id] = options # a dataframe with columns: ms_of_day,open,high,low,close,volume,count,date
                else: 
                    print(f"Option data not available for {option_id}") #TODO: good place to use logger
                #Request ohlc data for option 
           
            
            
    def get_options_data(self, option_id: str) -> pd.DataFrame:
        """
         returns a dataframe with columns: ms_of_day,open,high,low,close,volume,count,date
        """
        return self.options_data[option_id]
    
    
    def get_option_id(self, contract: pd.DataFrame) -> str: 
        """
            returns a string format of underlier-expiration-strike-type from the dataframe of the columns root, expiration, strike, right
        """
        return f"{contract['root']}-{contract['expiration']}-{contract['strike']}-{contract['right']}"