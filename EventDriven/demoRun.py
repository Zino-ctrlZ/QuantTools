
from dotenv import load_dotenv
load_dotenv()
import os
import sys
import cProfile
import pstats
import io
sys.path.append(
    os.environ.get('WORK_DIR')) #type: ignore
sys.path.append(
    os.environ.get('DBASE_DIR')) #type: ignore
from dbase.DataAPI.ThetaData import * #type: ignore
from dbase.database.SQLHelpers import * #type: ignore
import pandas as pd
from data import HistoricTradeDataHandler
from event import *
from queue import Queue
from trade.backtester_.backtester_ import PTDataset, PTBacktester
import pandas_ta as ta
from trade.assets.Stock import Stock
from trade.backtester_.utils.WalkForwardUtils import prev_monday 
from trade.backtester_.strats import MAStrat
import yfinance as yf
from datetime import datetime
from backtest import OptionSignalBacktest
import asyncio


def create_datasate(stocks: list, start: str,interval: str, engine: str = 'yf', timewidth = None, timeframe = None, end: str = datetime.today(), return_object = False ):
    dataset = []
    if engine.lower() == 'yf':
        for stock in stocks:
            start = prev_monday(start)
            data2 = yf.download(stock, start = start, end = end, interval=interval, progress = False)

            dataset.append(PTDataset(stock, data2))
    else:
        for stk in stocks:
            stock = Stock(stk)
            data = stock.spot(ts = True, ts_start = '2018-01-01')
            data.rename(columns = {x:x.capitalize() for x in data.columns}, inplace= True)
            data['Timestamp'] = pd.to_datetime(data['Timestamp'], format = '%Y-%m-%d')
            data2 = data.set_index('Timestamp')
            data2 = data2.asfreq('W', method = 'ffill')
            data2 = data2.fillna(0)
            data2['Next_Day_Open'] = data2.Open.shift(-1)
            data2['EMA'] = ta.ma('ema', data2.Close, length = 21).fillna(0)
            dataset.append(PTDataset(stk, data2))
    return dataset if return_object else data2
  
  


async def main():
  start, end, interval = '2023-05-29', '2024-05-28','1d'
  STOCKS = ['AAPL', 'MSFT','GOOGL', 'AMD', 'AMZN']
  dataset = create_datasate(STOCKS, start, interval,end = end , return_object=True)
  MAStrat.start_date = pd.to_datetime('1994-03-22')
  tt = PTBacktester(dataset, MAStrat, cash =1000, commission = 0.0035)
  stats = tt.run()
  trades = tt.__trades()
  shorts = tt.__trades()[tt.__trades()['Size'] < 0]
  trades = trades[:10]
  
  # Backtest class 
  evb_backtest = OptionSignalBacktest(trades) 
  profiler = cProfile.Profile()
  profiler.enable()
  # Run backtest
  await evb_backtest.run()
  profiler.disable()
  stream = io.StringIO()
  stats = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
  stats.print_stats(15)
  stream.seek(0)
  
  with open('backtest_stats.txt', 'w') as f:
    f.write(stream.read())
    f.flush()

if __name__ == '__main__':
  asyncio.run(main())