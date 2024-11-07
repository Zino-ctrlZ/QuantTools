import os
import sys
import pickle
sys.path.append(
    os.environ.get('WORK_DIR')) # type: ignore
from backtesting.backtesting import Strategy
import pandas as pd
from pandas.tseries.offsets import BDay
import numpy as np
from datetime import datetime
from trade.backtester_.utils.WalkForwardUtils import *
from trade.backtester_.Universe import universe
from trade.helpers.helper import printmd
UNIVERSE = universe



class WFO:
    is_train_backtest = True # TO ONLY
    is_test_backtest = True
    is_reopened = False
    prev_open_positions_dict = None
    is_position_open = False
    open_positions = []
    counter = 50
    official_start = None
    i = 0
    size = None
    exit_price = None
    exit_date = None
    init_buy = None
    log = pd.DataFrame()
    should_log = False



    def init(self):
        super().init()
        self.last_date = self.data.index[-1]
        if self.prev_open_positions_dict and len(self.prev_open_positions_dict) != 0:
            if self._name in self.prev_open_positions_dict.keys():
                position_details = self.prev_open_positions_dict[self._name]
                self.size, self.exit_price, self.exit_date =  position_details['Size'], ((position_details['ExitPrice'])), pd.to_datetime(position_details['ExitTime'] - BDay(0))
                # Using prev day because backtesting.py buys on next day
        assert (self.is_train_backtest or self.is_test_backtest) and not (self.is_train_backtest and self.is_test_backtest), "Exactly one of self.is_train_backtest or self.is_test_backtest must be True."
        assert  self.official_start, "Test/Validation backtest needs self.official_start to run."
    
    def next(self):
        # ## REOPEN PREV OPEN POSITIONS

        date = self.data.index[-1] ## SET CURRENT DATE ON EACH NEXT LOOP
        if self.is_train_backtest:
            self.train_backtest(date) ## Begin train_backtest if user wants to train (setting is_train_backtest to true)
        elif self.is_test_backtest:

                ## Begin test_backtest if user wants to test (setting is_test_backtest to true)
            self.test_backtest(date)

    def check_open_positions(self, date): # This creates a dictionary holding positions that weren't closed. This is necessary for positions to be carried over to next validation run
        # This is only created in test backtest because we are only accumulating returns of validation runs

        if date == self.last_date :
            # print('HI', self.last_date, 'Test run', self.is_test_backtest, 'Train run', self.is_train_backtest)
            pass
            # if self.position:
            #     self.open_positions.append({'Name': self._name,'Size' : self.position.size, 'ExitDate': self.last_date, 'Close': self.data.Close[-1], 'Prev_Close': self.data.Close[-2]})
            # else:
            #     self.open_positions.append({'Name': self._name, 'Size' : 'NO OPEN POSITION', 'ExitDate': self.last_date, 'Close': self.data.Close[-1], 'Prev_Close': self.data.Close[-2]})
        
        
        # self.logger({'Date': [date],
        #         'Close': [self.data.Close[-1]],
        #         'Upper Band': [self.upper_inner_band[-1]],
        #         'Name': [self._name],
        #         'Crossover Flag': [crossover(self.data.Close, self.upper_inner_band)]
        #         }) if self.should_log else None


    def train_backtest(self, date):

        self.prev_open_positions_dict = None # Handle reset of this variable incase it hasn't already been handled
        if date >= self.official_start:
            super().next()
        self.check_open_positions(date) #Comment this out when officially carrying out WFO

    def test_backtest(self, date):
        assert self.prev_open_positions_dict is not None, f"self.prev_open_positions_dict cannot be none in test_backtest. If there are no previous positions, pass an empty dictionary"
       
        # if self._name in self.prev_open_positions_dict and not self.is_reopened:
        #     self.reopen_previous_position(date)
        # else:
        if date >= self.official_start:
            super().next()
            self.check_open_positions(date)

    def reopen_previous_position(self, date):
        if date >= self.exit_date and not self.position and not self.is_reopened:
            self.init_buy = date
            self.buy()
            self.is_reopened = True
            print(f'Reopened {self._name} on {date}, for size {self.size}')


    @classmethod
    def logger(cls, dictionary):
        item = pd.DataFrame(dictionary)
        cls.log = pd.concat([cls.log, item]).reset_index(drop =True)


    @classmethod           
    def reset_variables(cls): 
        # PTBacktester is basically looping individual backtest.run(). Backtest object USES a deep copy of strategy object, which means every instance of Backtest will have the same
        # strategy object & variables. Therefore to ensure that during the PTBacktester looping has fresh values, we have to reset position related variables.

        cls.exit_price = None
        cls.exit_date = None
        cls.size = None
        cls.is_reopened = False
        cls.is_train_backtest = True
        cls.is_test_backtest = True






from typing import Type, Union, Tuple
from backtesting.backtesting import Strategy
from datetime import datetime
import asyncio
class WalkForwardAnalysis:
    def __init__(self, names: list, strategy: Union[Type, Type[Strategy]], optimize_var: dict, engine: str):
        assert engine in ['position', 'cross'], f'Available engines are "position" or "cross". Recieved "{engine}"'
        self.names = names
        self.strategy = strategy
        self.split_datasets = None # Initiated by method split_dataset
        self.settings = None # Settings holding other important attributes. This just helps us ensure we can expand the code w/o over populating the __init__.
                             # This is set by the set_settings method, will be called at initialization since there are defaults

        self.strategy_settings_lib = None
        self.optimize_var = optimize_var
        self.windows = None # Dict holding the windows for everthing which includes: 
                            # Train: Start, end
                            # Test: Official Start, Start, end.
                            # This is set by a method split_windows. Can't split at initiation cause we need to pass the settings first
        self.trainOpt_data = None # Variable holding the data obtained from running an optimization on train dataset
        self.tested_data = None # Variable holding the data obtained from OOS testing
        self.engine = engine #String with name
        self.set_settings({})

    # async def run(self):
    #     variable = await self.run_process()
    #     if variable:
    #         self.save_class()

    #     return variable

    # async def run_process(self):
    def run(self):
        train_windows = self.windows['train_window']
        test_windows = self.windows['test_window']
        train_data = self.split_datasets['train_window']
        test_data = self.split_datasets['test_window']
        train_packaged_data = {}
        test_packaged_data = {}
        to_dataframe_dict = {}
        wfe_data = {}
        mega_data = {}
        saved_strat_settings = {}
        val_, train_start, train_end, test_start, test_end, train_CAGR, test_CAGR, wfe_, train_drawdown = [], [], [], [],[], [], [], [],[]
        for val_run, data in test_data.items():
            printmd(f"### **Validation Run: {val_run}**") if self.printHeaders else None
            ## RETRIEVE RUN DATA & SPLIT INTO VARIABLES
            test_run_data = data
            train_run_data = train_data[val_run]
            off_start = self.windows['test_window'][val_run]['off_start']
            test_end_date = self.windows['test_window'][val_run]['End']
            train_end_date = self.windows['train_window'][val_run]['End']
            off_start_train = self.windows['train_window'][val_run]['off_start']
            no_days_traded_in_test = np.busday_count(off_start.strftime('%Y-%m-%d'),test_end_date)

            ## TRAIN DATA
            trained_data = self.train(train_run_data, off_start_train)
            trained_target_metric = trained_data['agg']['CAGR [%]']

            ## TEST WITH STRATEGY SETTING
            strategy_setting = trained_data['strategy_settings']
            tested_data = self.test(test_run_data, strategy_setting, off_start)
            tested_target_metric = tested_data['agg']['CAGR [%]']
            tested_drawdown = tested_data['agg']['Max. Drawdown [%]']
            annualized_drawdown = tested_drawdown * 1/(no_days_traded_in_test/260)
            
            ## SAVE AGG FROM BOTH
            train_packaged_data[val_run] = trained_data
            test_packaged_data[val_run] = tested_data
            wfe_data[val_run] = tested_target_metric/trained_target_metric
            saved_strat_settings[val_run] = strategy_setting
            ## PRINT WFE FOR THE RUN
            print(f"Validation Run: {val_run}, WFE: {tested_target_metric/trained_target_metric}, Test CAGR [%]: {tested_target_metric}, Train CAGR [%]: {trained_target_metric}, Annualized Drawdowon [%]: {annualized_drawdown}") if self.printHeaders else None

            ## Appending data to list which goes into dataframe
            for lst, val in zip([val_, train_start, train_end, test_start, test_end, train_CAGR, test_CAGR, wfe_, train_drawdown], 
                                [val_run, off_start_train.strftime('%Y-%m-%d'), train_end_date, off_start.strftime('%Y-%m-%d'), test_end_date, trained_target_metric, tested_target_metric,tested_target_metric/trained_target_metric ,tested_drawdown]):
                lst.append(val)
        ## Creating a data dictionary

        for lst, col in zip([train_start, train_end, test_start, test_end, train_CAGR, test_CAGR, wfe_, train_drawdown], 
                            ['Train_Start_Date', 'Train_end_Date', 'Test_Start_Date', 'Test_End_Date', 'Train_CAGR', 'Test_CAGR', 'WFE', 'TEST_ANNUALIZED_DRAWDOWN']):
            to_dataframe_dict[col.upper()] = lst
        
        ## Saving data to class attributes
        self.trained_data = train_packaged_data
        self.tested_data = test_packaged_data
        self.strategy_settings_lib = saved_strat_settings
        self.WFE_data = wfe_data
        stats = pd.DataFrame(index = pd.Index(val_, name = 'VALIDATION_RUN', dtype = 'int64'), data = to_dataframe_dict )
        stats['WFE_ADJUSTED'] = stats.WFE * np.sign(stats.TRAIN_CAGR)
        self.stats = stats
        self.save_class()

        return stats

    def save_class(self):
        className = self.strategy.__bases__[1].__name__
        print(className)
        anc = 'ANCHORED' if self.anchored else 'UNANCHORED'
        name = f'{className}_{"_".join(self.names)}_lookback_{self.lookback_bars}_val_{self.validation_bars}_warmup_{self.warmup_bars}'
        today = datetime.today().strftime('%Y-%d-%m')
        save_location = f'WFA/{today}/{anc}/{name}.pkl'
        dir = os.path.dirname(save_location)
        os.makedirs(dir, exist_ok = True)
        with open(save_location, 'wb') as file:
            pickle.dump(self, file)

        
    def split_(self) -> Tuple[dict, dict]:
        """
        Returns a dictionary holding both the split up window & corresponding dataset objects to carry out backtesting.
        
        """
        return split_window(stocks = self.names,
                        strategy= self.strategy,
                        data_end= self.data_end,
                        data_length_str= self.data_length_str,
                        warmup_bars = self.warmup_bars,
                        lookback_bars= self.lookback_bars,
                        validation_bars = self.validation_bars,
                        anchored = self.anchored,
                        interval = self.interval
                        )
     

    def set_settings(self, settings) -> None:
        """
        Attribute responsible for initiating the necessary settings to assist with the WFA. Pass a dict with setting name as key and corresponding setting as values

        dict params:
        ____________

        BaseRun (bool): designates whether this WFA is a base Run (no optimization, just runs of split up data with constant parameters)
        anchored (bool): True to run an anchored WFA or False to not
        printHeaders (bool): Bool deciding whether to print headers or not
        data_end (datetime): Datetime object for when the WFA data should end
        warmup_bars (int): Number of bars to be used as warmup bars
        lookback_bars (int): Number of bars to be used as lookback/train bars
        validation_bars (int): Number of bars to be used in valudation
        cash (int, dict): Cash value. int defaults to setting all names with the cash value supplied. Dict must be {ticker: cash} with corresponding names in names
        commission (float): Commission
        data_length_str (string): Length string to evaluate. Eg: years = 6. refer to dateutils.relativedelta.relativedelta for available options
        interval (string): Timeseries interval
        optimize_str (str): Applicable to 'position' engine in WF. The associated string to be optimized from backtesting.py optimizer
        optimize_list (list): Applicable to 'cross' engine in WF. Associated list of items to be optimized in PTBacktester optimizer
        target_metric (str): This is the index name as seen in the aggregate function. PLEASE PASS EXACTLY
        
        Defaults:
        __________

        {'BaseRun': False, 'anchored': False, 'printHeaders': False, 'data_end': datetime.datetime(2024, 8, 16, 16, 58, 5, 875120), 'warmup_bars': 300, 'lookback_bars': 1308, 'cash': 1000, 
            'commission': 0.002, 'interval': '1d', 'validation_bars': 126, 'data_length_str': 'years = 15', 'optimize_str': 'Return [%]', 'optimize_list': ['rtrn']}
        """
        
        settings_list = ['BaseRun', 'anchored', 'printHeaders', 'data_end', 'warmup_bars', 'lookback_bars', 'cash', 'commission', 'interval', 'validation_bars', 'data_length_str', 'optimize_str', 'optimize_list', 'target_metric'] # List of available settings
        settings_default = [False, False, False, datetime.today(), 300, 252*4+300, 1000, 0.002, '1d', 126, 'years = 15', 'Return [%]', ['rtrn'], 'Return [%]'] # Available settings corresponding default args
        settings_type = [bool, bool, bool, datetime, int, int, [int, dict, float], float, str, int, str, str, list, str] # Available settings corresponding datatype
        settings_default_dict = dict(zip(settings_list, settings_default)) #Creating settings default 
        settings_criteria = dict(zip(settings_list, settings_type)) # Creating dict with settings type to assert types allowed

        for i, (key, value) in enumerate(settings.items()):
            assert key in settings_list, f"Setting '{key}' not a valid settings. Valid settings: {settings_list}"
            if key == 'cash':
                assert isinstance(value, settings_criteria[key][0]) or isinstance(value, settings_criteria[key][1]) or isinstance(value, settings_criteria[key][2]), f"Type {type(value)} not a valid type for '{key}', expecting {settings_criteria[key]}"
            else:
                assert isinstance(value, settings_criteria[key]), f"Type {type(value)} not a valid value for {key}, expecting {settings_criteria[key]}"
            settings_default_dict[key] = value

        for key, value in settings_default_dict.items():
            setattr(self, key, value)
        self.settings = settings_default_dict
        self.windows, self.split_datasets = self.split_()
    
    def train(self, data: list, off_start: pd.Timestamp):
        
        if self.engine == 'position':
            agg_train = position_train(self.strategy, data, self.optimize_var, off_start =off_start, optimize_str= self.optimize_str, cash = self.cash, baseRun = self.BaseRun, printHeaders = self.printHeaders )
        else:
            agg_train = cross_train(self.strategy, data, self.optimize_var,off_start =off_start, optimize_list= self.optimize_list, cash = self.cash,baseRun = False, printHeaders = self.printHeaders)
        return agg_train

    def test(self, data: list, strategy_setting: dict, off_start: pd.Timestamp):
        if self.engine == 'position':
            agg_test = position_test(strategy  = self.strategy, off_start= off_start, strategy_settings= strategy_setting,validation_datas= data,cash = self.cash, commission= self.commission, plot_positions= False )
        else:
            agg_test = cross_test(self.strategy, off_start, strategy_setting, data,self.cash, self.commission)
        return agg_test
    




    def produce_summary(self, data_choice: str = 'test') -> pd.Series:
        """
        Params:
        ________

        data_choice: 'test' to recieve summary for OOS data and 'train' for IS data

        
        Returns:
        _________
        pd.Series
        
        """
        assert self.tested_data is not None, f"Please run Walk Forward Analysis to produce necessary datapoints"
        assert data_choice in ['test', 'train'], f"Only options for summary production is 'test' and 'train'. Recieved '{data_choice}'"
        def compute_summary(data_choice):
            dataChoice = self.tested_data if data_choice == 'test' else self.trained_data
            val_run = list(dataChoice.keys())
            metrics = ['# Trades', 'Best Trade [%]', 'Worst Trade [%]', 'Avg. Winning Trade [%]', 'Avg. Losing Trade [%]', 'Avg. Trade [%]', 'Winning Streak', 'Losing Streak']
            windows = self.windows['train_window'] if data_choice == 'train' else self.windows['test_window']
            summary = pd.DataFrame(index = val_run)
            for col in metrics:
                for run in val_run: 
                    no_bus_days = np.busday_count(windows[run]['off_start'].strftime('%Y-%m-%d'), windows[run]['End'])
                    summary.at[run, col] = dataChoice[run]['agg'][col]
                    summary.at[run, 'Net PnL']  = dataChoice[run]['agg']['Equity Final [$]'] - dataChoice[run]['agg']['equity_curve']['Total'][0]
                    summary.at[run, 'Annualized Net PnL']  =  annualize_net_profit(dataChoice[run]['agg']['Equity Final [$]'] - dataChoice[run]['agg']['equity_curve']['Total'][0],dataChoice[run]['agg']['equity_curve']['Total'][0], no_bus_days)
                    summary.at[run, 'Annualized Net PnL [%]']  =  annualize_net_profit(dataChoice[run]['agg']['Equity Final [$]'] - dataChoice[run]['agg']['equity_curve']['Total'][0],dataChoice[run]['agg']['equity_curve']['Total'][0], no_bus_days,False)
                    summary.at[run, 'Net PnL [%]']  = (dataChoice[run]['agg']['Equity Final [$]'] - dataChoice[run]['agg']['equity_curve']['Total'][0])/dataChoice[run]['agg']['equity_curve']['Total'][0]
                    summary.at[run, 'Losing Trades'] = (dataChoice[run]['agg']['# Trades'] * (dataChoice[run]['agg']['Lose Rate [%]']/100)).round(0)
                    summary.at[run, 'Winning Trades'] = (dataChoice[run]['agg']['# Trades'] * (1-(dataChoice[run]['agg']['Lose Rate [%]']/100))).round(0)
            summarized_summary = pd.Series()
            summarized_summary['Avg Net Profit'] = summary['Net PnL'].mean()
            summarized_summary['Avg Annualized Net Profit'] = summary['Annualized Net PnL'].mean()
            summarized_summary['Annualized Net PnL [%]'] = summary['Annualized Net PnL [%]' ].mean()
            summarized_summary['Avg Net Profit'] = summary['Net PnL'].mean()
            summarized_summary['Avg Net Profit [%]'] = summary['Net PnL [%]'].mean()
            summarized_summary['Total # of Trades'] = summary['# Trades'].sum()
            summarized_summary['Total # of Winning Trades'] = summary['Winning Trades'].sum()
            summarized_summary['Total # of Losing Trades'] = summary['Losing Trades'].sum()
            summarized_summary['Largest Losing Trades [%]'] = summary['Worst Trade [%]'].min()
            summarized_summary['Largest Winning Trades [%]'] = summary['Best Trade [%]'].max()
            summarized_summary['Avg. Winning Trade [%]'] = summary['Avg. Winning Trade [%]'].mean()
            summarized_summary['Avg. Losing Trade [%]'] = summary['Avg. Losing Trade [%]'].mean()
            summarized_summary['Avg. Trade [%]'] = summary['Avg. Trade [%]'].mean()
            summarized_summary['Max Winning Streak'] = summary['Winning Streak'].max()
            summarized_summary['Max Losing Streak'] = summary['Losing Streak'].max()
            return summarized_summary

        test_summary = compute_summary('test')
        train_summary = compute_summary('train')
        test_summary['WFE'] = test_summary['Avg Annualized Net Profit']/train_summary['Avg Annualized Net Profit']
        train_summary['WFE'] = test_summary['Avg Annualized Net Profit']/train_summary['Avg Annualized Net Profit']

        return test_summary  if data_choice == 'test' else train_summary
