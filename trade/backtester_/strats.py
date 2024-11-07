from backtesting.lib import crossover
import pandas_ta as ta
import os
import sys
sys.path.append(
    os.environ.get('WORK_DIR')) # type: ignore
from backtesting.backtesting import Strategy
import pandas as pd
    

def shift(series,n):
    pd.Series(series)
    return pd.Series(series).shift(n)


class BBandsTrend2(Strategy):
    start_date = None
    outter_band = 2
    inner_band = 0.5
    length = 190
    exit_ma = 35
    stop_loss = 0.1
    long_close = False
    long_close_counter = 0
    long_open = False
    long_open_counter = 0
    short_open_ = False
    short_open_counter = 0
    short_close_ = False
    short_close_counter = False
    open_wait_days = 0
    close_wait_days =0
    gapper_limit = 1000
    start, end, interval = '2000-06-08', '2024-06-29', '1d'

        
    def init(self):
        #COMPUTE MOVING AVERAGES FOR STRATEGY
        bbands_outter = ta.bbands(pd.Series(self.data.Close), length=self.length, std=self.outter_band, mamode='sma')
        bbands_inner = ta.bbands(pd.Series(self.data.Close), length=self.length, std=self.inner_band, mamode='sma')
        macd = ta.macd(pd.Series(self.data.Close), 12,26,9)
        adx = ta.adx(pd.Series(self.data.High), pd.Series(self.data.Low), pd.Series(self.data.Close), 21, mamode='wilders')
        # Assign bands to instance variables
        self.lower_inner_band = self.I(lambda: bbands_inner[f'BBL_{int(self.length)}_{float(self.inner_band)}'], name = 'lower_inner_band', color = 'blue', overlay = True)
        self.middle_inner_band = self.I(lambda: bbands_inner[f'BBM_{int(self.length)}_{float(self.inner_band)}'], name = 'middle_inner_band')
        self.upper_inner_band = self.I(lambda: bbands_inner[f'BBU_{int(self.length)}_{float(self.inner_band)}'], name = 'upper_inner_band', color = 'blue', overlay= True)
        # self.sp500 = self.I(create_datasate,['SPY'], self.start, '1d',end = self.end , return_object=False)
        self.MACD_Hist = self.I(lambda: macd['MACDh_12_26_9'], name = 'MACD HIS')
        self.ADX = self.I(lambda: adx['ADX_21'])
        
        self.exit_ma = self.I(ta.ema, pd.Series(self.data.Close), length=self.exit_ma)
    
    def next(self):
        price = self.data.Close[-1]
        date = self.data.index[-1]
        macd = self.MACD_Hist[-1]
        adx = self.ADX[-1]
        upper = self.upper_inner_band[-1]
        lower = self.lower_inner_band[-1]
        date = self.data.index[-1]
        middle = self.middle_inner_band[-1]
        exit_ma = self.exit_ma[-1]
        up_gap_ = abs((upper / exit_ma) - 1)*100
    




        print(date, ':','Open', self.long_open, self.long_open_counter) if self.long_open_counter > 51 else None
        print(date, ':','Close', self.long_close, self.long_close_counter) if self.long_close_counter > 51 else None
        # Check for entry crossover from below to above
        if price > upper and not self.long_open:
            # print('Set Long Flag to True', date)
            self.long_open = True
            self.long_open_counter += 1

        # Check for exit crossover from below
        if price < upper and not self.long_close:
            self.long_close = True
            self.long_close_counter = True

        # If Price goes below entry crossover, reset entry flags
        if price < upper and self.long_open:
            # print('Set Long Flag to False', date)
            self.long_open = False
            self.long_open_counter = 0


        #If price goes back above exit crossover, reset exit flags
        if price > upper and self.long_close:
            # print('Set Close Long Flag to True', date)
            self.long_close = False
            self.long_close_counter = 0

        # Increment open counter if entry is still valid
        if self.long_open:
            # print('Hi')
            self.long_open_counter += 1
        

        #Increment close counter if exit is still valid
        if self.long_close:
            self.long_close_counter += 1

        # Enter a trade after waiting period if no position is open
        if self.long_open and self.long_open_counter >= self.open_wait_days:
            if not self.position:
                # print('Opening Long', date)
                self.buy(sl=self.data.Close[-1] * (1 - self.stop_loss))
            self.long_open = False
            self.long_open_counter = 0

        #Exit a trade after waiting period if position is still open
        if self.long_close and self.long_close_counter >= self.close_wait_days:
            if self.position:
                # print('Closing Long', date)
                self.position.close()
            self.long_close = False
            self.long_close_counter = 0


class MAStrat(Strategy):
    trend_ma = 71
    entry_ma = 36
    exit_ma_v = 20
    stop_loss = 0.30  # 2% stop loss
    take_profit = 0.15 
    shift = 4
        
    def init(self):
        #COMPUTE MOVING AVERAGES FOR STRATEGY
        # self.ma_trend = self.I(ta.ma, "ema", pd.Series(self.data.Close), length = self.trend_ma )
        self.ma_entry = self.I(ta.ma, "ema", pd.Series(self.data.Close), length = self.entry_ma )
        self.exit_ma = self.I(ta.ma, "ema", pd.Series(self.data.Close), length = self.exit_ma_v )
        self.ma_shifter = self.I(shift, self.exit_ma, self.shift)
        self.close_shifter = self.I(shift, self.data.Close, self.shift)
        # self.trend_ma = self.I(ta.ma, "ema", pd.Series(self.data.Close), length = self.trend_ma )
        
    def next(self):
        shifted = self.ma_shifter[-1]
        price = self.data.Close[-1]
        date = self.data.index[-1]
        entry = self.ma_entry[-1]
        close_shifted = self.close_shifter[-1]
        # trend = self.trend_ma[-1]
        # print(self.data.Next_Day_Open[-1]) 
        # print(entry, shifted)

        #IF WE DON'T ALREADY HAVE A POSITION
        if  crossover(self.data.Close,self.ma_entry ) and not self.position and entry >= shifted and price >= close_shifted:
            self.buy(sl=self.data.Close[-1] * (1 - self.stop_loss))

        
        elif (self.position and price < self.exit_ma)  :
            self.position.close()
        
        # elif price < trend:

        #     if crossover(self.ma_entry, self.data.Close ) and not self.position and price < entry:
        #         self.sell()

        #     elif (self.position and price > self.exit_ma):
        #         self.position.close()


        # FIGURE OUT HOW TO USE CROSS
        elif not self.position and crossover(self.data.Close,self.exit_ma ) :
            # RE-ENTER TRADE AS LONG AS ABOVE 21 SMA & CROSS OVER 13
            self.buy(sl=self.data.Close[-1] * (1 - self.stop_loss))
                
