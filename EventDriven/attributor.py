from trade.assets.OptionStructure import OptionStructure
from trade.assets.helpers.loaders import create_object_from_id
from trade.helpers.Context import Context
from trade.assets.Calculate import Calculate
import yfinance as yf   
import pandas as pd
import numpy as np
from trade.assets.Stock import Stock
from pandas.tseries.offsets import BDay
from trade.assets.Option import Option
import threading
import time
from trade.helpers.Logging import setup_logger
from IPython.display import clear_output
logger = setup_logger('EventDriven.attributor')


class EVBAttributor:
    """
    Class to load data from Event Driven Backtester and calculate PnL and Greeks
    """
    def __init__(self, 
                 trades: pd.DataFrame, 
                 attribution_fill: str = 'default_fill',
                 option_fill: str = 'midpoint',
                 retries: int = 3):
        """
        trades: DataFrame containing trades data from Event Driven Backtester
            Eexpected columns: EntryTime, ExitTime, Quantity, Positions
        retries: Number of retries for each trade
        attribution_fill: Method to fill missing data in attribution calculation
        option_fill: Method to fill missing data in option class. Default is midpoint
        """
        assert 'EntryTime' in trades.columns, 'EntryTime column not found'
        assert 'ExitTime' in trades.columns, 'ExitTime column not found'
        assert 'Quantity' in trades.columns, 'Quantity column not found'
        assert 'Positions' in trades.columns, 'Positions column not found'

        self.trades = trades
        self.trades['Structure'] = None
        self.retries = retries
        self.stored_data = {
            'attribution': {},
            'greeks': {},
            'vol_slides': {},
            'pct_spot_slides': {}
        }
        self.attribution_fill = attribution_fill
        self.option_fill = option_fill

    def _create_object_from_id(self, *args ,**kwargs):
        try:
            return create_object_from_id(*args, **kwargs)
        except Exception as e:
            print(e)
            print(kwargs, args)
            return None
    
    def load_data(self, 
                  attribution: bool = True, 
                  greeks: bool = True, 
                  attribution_method: str = 'RV', 
                  print_output: bool = True) -> None:
        
        """
        Load data from trades DataFrame and calculate PnL and Greeks

        params:
        attribution: bool - Calculate attribution data if True
        greeks: bool - Calculate greeks data if True
        attribution_method: str - Method to calculate attribution data. available methods are 'RV', 'GB'
        print_output: bool - Print output if True

        return: None
        use self.attribution and self.greeks to access the calculated data
        """
        trades = self.trades
        
        date_range = pd.date_range(trades['EntryTime'].min(), trades['ExitTime'].max(), freq = 'B')
        tries = 0
        failed = ['start'] ## Making sure the loop runs at least once
        ## Load Data
        while len(failed) > 0 :
            if tries >= self.retries:
                print(f"Retries exceeded for {failed}")
                break
            tries += 1
            for index in trades[trades.Structure.isna()].index:
                clear_output(wait=True)
                print(f"Starting {index}") if print_output else None
                start = (pd.to_datetime(trades.loc[index]['EntryTime']) + BDay(1)).strftime('%Y-%m-%d') ## Calculate PnL from the day after trade entry. This replicates buy at close and sell at open
                end = (pd.to_datetime(trades.loc[index]['ExitTime']) - BDay(1)).strftime('%Y-%m-%d') ## Temp fix for trades enddate issue
                quantity = trades.loc[index]['Quantity']
                id = trades.loc[index]['Positions']
                structure = self._create_object_from_id(id, start, default_fill = self.option_fill)
                
                ## Create PnL and Greeks Data from structure object
                if structure is not None:
                    if attribution:
                        pnl = Calculate.attribution(structure, start, end, method = attribution_method, replace = self.attribution_fill) * quantity
                        self.stored_data['attribution'][index] = pnl
                    if greeks:
                        greeks_data = structure.greeks('greek', ts_start = start, ts_end = end) * quantity * 100
                        self.stored_data['greeks'][index] = greeks_data
                else:
                    trades.loc[index, 'Structure'] = None
                    print(f"Failed to load {index}") if print_output else None
                    self.stored_data['attribution'][index] = 0
                    self.stored_data['greeks'][index] = 0
                    continue



                trades.loc[index, 'Structure'] = structure
                print(f"Completed {index}") if print_output else None
                

            ## Produce Empty DataFrames based on Date Range from Trades Entry to Exit
            ## Columns off the DataFrames are based on the first successful attribution and greeks data
            for ind in self.stored_data['attribution'].keys():
                if isinstance(self.stored_data['attribution'][ind], pd.DataFrame):
                    pnl_sample = self.stored_data['attribution'][ind].copy()
                    greeks_sample = self.stored_data['greeks'][ind].copy()
            
            attribution = pd.DataFrame(index = date_range,

                        data = {x: [0] * len(date_range) for  x in pnl_sample.columns})
            
            pt_greeks = pd.DataFrame(index = date_range,
                                data = {x: [0] * len(date_range) for  x in greeks_sample.columns})
            
            pct_spot_slides = pd.DataFrame(index = date_range,
                                data = {x: [0] * len(date_range) for  x in greeks_sample.columns})
            
            vol_slides = pd.DataFrame(index = date_range,
                                data = {x: [0] * len(date_range) for  x in greeks_sample.columns})
                

            ## Fill DataFrames with PnL and Greeks Data
            ## This is done by adding the pnl and greeks data to the corresponding columns in the dataframes
            failed = []
            for index, pnl in self.stored_data['attribution'].items():
                if not isinstance(pnl, pd.DataFrame):
                    print(f"{index} failed") if print_output else None
                    failed.append(index)
                    continue
                days_mask = attribution.index.isin(pnl.index)
                attribution.loc[days_mask, :] += pnl
                attribution.fillna(0, inplace=True)
                
            for index, greeks in self.stored_data['greeks'].items():
                if not isinstance(greeks, pd.DataFrame):
                    print(f"{index} failed") if print_output else None
                    failed.append(index)
                    continue
                days_mask = pt_greeks.index.isin(greeks.index)
                pt_greeks.loc[days_mask, :] += greeks
                pt_greeks.fillna(0, inplace=True)

            self.attribution = attribution
            self.greeks = pt_greeks
            self.pct_spot_slides = pct_spot_slides
            self.vol_slides = vol_slides
                        

            
