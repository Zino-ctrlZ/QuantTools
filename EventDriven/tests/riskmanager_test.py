from EventDriven.riskmanager import RiskManager
from dbase.DataAPI.ThetaData import list_contracts, retrieve_option_ohlc, is_theta_data_retrieval_successful #type: ignore
import datetime
import pandas as pd
import pandas_market_calendars as mcal
import unittest
import numpy as np

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD']
date_range = pd.date_range()

#generate date range 
nyse = mcal.get_calendar('NYSE')
year_ago_date = datetime.datetime.now() - datetime.timedelta(days=365)
schedule = nyse.schedule(start_date=year_ago_date, end_date=datetime.datetime.now())
date_range = mcal.date_range(schedule, frequency='1D')
dates = [date.strftime('%Y-%m-%d') for date in date_range]




class RiskManagerOperations(unittest.TestCase):
    def set_up(self):
        self.risk_manager = RiskManager()
        
    def test_order_picker(self):
        ticker = np.random.choice(tickers)
        contract_date = np.random.choice(dates)
        contracts = list_contracts(ticker, pd.to_datetime(contract_date).strftime('%Y%m%d'))
        self.assertTrue(is_theta_data_retrieval_successful(contracts))
        
        contract = contracts.sample()
        contract_right = contract['right']
        contract_expiration = pd.to_datetime(contract['expiration']).strftime('%Y%m%d')
        contract_strike = float(contract['strike'])
        max_close = np.random.randint(1, 10)
        
        #order settings 
        moneyness_width = np.random.uniform(0.01, 0.05)
        rel_strike_long = np.random.uniform(1.05, 1.3) 
        rel_strike_short = np.random.uniform(0.7, 0.95)
        dte = np.random.randint(30, 365)
        
        order_settings = {
            'type': 'spread',
            'specifics': [
                {'direction': 'long', 'rel_strike': rel_strike_long, 'dte': dte, 'moneyness_width': moneyness_width},
                {'direction': 'short', 'rel_strike': rel_strike_short, 'dte': dte, 'moneyness_width': moneyness_width} 
            ],
            'name': 'vertical_spread'
        }
       
        try:
            self.order = self.risk_manager.OrderPicker.get_order(ticker, contract_expiration, contract_right, max_close, order_settings)
            self.assertIsInstance(self.order, dict)
            self.assertIsInstance(self.order['long'], list)
            self.assertIsInstance(self.order['short'], list)
            self.assertGreater(len(self.order['long']), 0)
            self.assertGreater(len(self.order['short']), 0)
            self.assertIsInstance(self.order['close'], float)
        except AssertionError as e:
            print(f"AssertionError: {e}")
            print(f"Ticker: {ticker}")
            print(f"Contract Date: {contract_date}")
            print(f"Contracts: {contracts}")
            print(f"Contract: {contract}")
            print(f"Contract Right: {contract_right}")
            print(f"Contract Expiration: {contract_expiration}")
            print(f"Contract Strike: {contract_strike}")
            print(f"Max Close: {max_close}")
            print(f"Order Settings: {order_settings}")
            raise
        except Exception as e:
            print(f"Exception: {e}")
            print(f"Ticker: {ticker}")
            print(f"Contract Date: {contract_date}")
            print(f"Contracts: {contracts}")
            print(f"Contract: {contract}")
            print(f"Contract Right: {contract_right}")
            print(f"Contract Expiration: {contract_expiration}")
            print(f"Contract Strike: {contract_strike}")
            print(f"Max Close: {max_close}")
            print(f"Order Settings: {order_settings}")
            raise

if __name__ == "__main__":
    unittest.main()