import os
from dotenv import load_dotenv
load_dotenv()
import sys
from helpers.helper import binomial, implied_vol_bt
import warnings
from helpers.Context import Context
from helpers.Configuration import Configuration

from helpers.helper import time_distance_helper
import numpy as np
import pandas as pd
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
import math


warnings.filterwarnings("ignore")


def pnl_data_organizer_helper(strike, exp, flag, op_ts = None, stock_ts = None, rates_ts = None):
    op_ts = op_ts
    stock_ts = stock_ts
    rf_times = rates_ts


    op_ts = op_ts[['timestamp', 'close']]
    op_ts = op_ts.rename(columns = {'close': "option_close",})
    op_ts.timestamp = pd.to_datetime(op_ts.timestamp, format = "%Y-%m-%d")
    op_ts.timestamp = op_ts.timestamp.apply(lambda x:x.strftime("%Y-%m-%d"))

    stock_ts = stock_ts[['timestamp', 'close']]
    stock_ts = stock_ts.rename(columns = {'close': "stock_close"})


    rf_times = rf_times.reset_index()
    rf_times.annualized = rf_times.annualized/100
    rf_times = rf_times[['timestamp', 'annualized']]
    rf_times.rename(columns = {"annualized": 'rates'}, inplace = True)

    merged = op_ts.merge(stock_ts, on = 'timestamp', how = 'right')
    merged = merged.merge(rf_times, on = 'timestamp', how = 'left')
    merged = merged.dropna()

    merged.rename(columns = {'timestamp': 'Date'}, inplace = True)

    for index, row in merged.iterrows():
        S0 = row['stock_close']
        r = row['rates']
        exp = exp
        market_price = row['option_close']
        flag = flag.lower()
        start = row['Date']
        K = strike
        vol = implied_vol_bt(S0= S0, K = K,r=r, market_price=market_price,exp_date = exp,flag = flag,start=start)
        if math.isnan(vol):
            try:
                vol = implied_volatility(price = market_price, S = S0, K = K, t = time_distance_helper(exp, start), r = r, q = 0, flag = flag)
            except:
                vol = np.nan
        merged.at[index, 'vol'] = vol
        merged.at[index, 'pv'] = binomial(S0= S0, K = K,r=r, sigma=vol,exp_date = exp,opttype = flag,start=start)


    merged.Date = pd.to_datetime(merged.Date, format = "%Y-%m-%d")
    merged = merged.set_index('Date')
    merged = merged.asfreq('B')
    merged.reset_index(inplace = True)
    merged.vol = merged.vol.bfill()
    merged.Date = merged.Date.apply(lambda x:x.strftime("%Y-%m-%d"))
    merged['timestamp'] = merged.Date
    merged = merged[['timestamp', 'option_close', 'vol']]
    merged = merged.merge(rf_times, on = 'timestamp', how = 'left')
    merged = merged.merge(stock_ts, on = 'timestamp', how = 'right')
    merged.rename(columns = {'timestamp': 'Date'}, inplace = True)

    
    merged['prev_stock_close'] = merged.stock_close.shift(periods = -1)
    merged['prev_rates_close'] = merged.rates.shift(periods = -1)
    merged['rates_change'] = merged['rates'] - merged['prev_rates_close']
    merged['prev_date'] = merged.Date.shift(periods = -1)
    merged['stock_change'] = merged['stock_close'] - merged['prev_stock_close']
    merged['prev_day_vol'] =  merged.vol.shift(periods = -1)
    merged['vol_change'] = merged['vol'] - merged['prev_day_vol']


    for index, row in merged.iterrows():
        S0, r, exp, market_price, flag, start, K, sig  = row['stock_close'], row['rates'], exp, row['option_close'], flag.lower(), row['Date'], strike, row['vol']
        pv = binomial(K = K, exp_date = exp, sigma= sig, r = r, S0 = S0, start=start,opttype=  flag )
        merged.at[index,'pv'] = pv
    
    merged['prev_option_pv'] = merged.pv.shift(periods = -1)
    merged['option_close'] = merged.pv
    merged['option_change'] = merged['option_close'] - merged['prev_option_pv']

    return merged.dropna()



def rv_attribution_helper(strike, exp, flag, merged_df) :
    merged = merged_df

    for index, row in merged.iterrows():
        
        # INITIATE VARIABLE
        prev_S0, S0 = row['prev_stock_close'], row['stock_close']
        prev_r, r = row['prev_rates_close'], row['rates']
        exp = exp
        market_price, prev_market_price = row['option_close'], row['prev_option_pv']
        flag = flag.lower()
        today_start,prev_start = row['Date'], row['prev_date']
        K = strike
        prev_pv, pv = row['prev_option_pv'], row['option_close']
        prev_vol, vol = row['prev_day_vol'], row['vol']

        #THETA PNL
        theta_bump_pv = binomial(S0= prev_S0, K = K,r=prev_r, sigma=prev_vol,exp_date = exp,opttype = flag,start=today_start)
        theta_pnl = -prev_pv + theta_bump_pv
        merged.at[index, 'theta_pnl'] = round(theta_pnl,6)*100


        change = S0 - prev_S0
        merged.at[index, 'stock_change'] = change
        option_change = market_price -prev_market_price
        
        
        #SPOT PNL (GAMMA + DELTA)
        S0_bump = prev_S0 +0.0001
        bump = binomial(S0 = S0_bump, r =prev_r, sigma= prev_vol, K = K, exp_date = exp, opttype= flag, start = prev_start )
        delta_rv_pnl = binomial(S0 = S0, r =prev_r, sigma= prev_vol, K = K, exp_date = exp, opttype= flag, start = prev_start )
        bump_pnl = bump - prev_pv
        rv_pnl = delta_rv_pnl-prev_pv
        delta_pnl = change * bump_pnl* 10000
        g_pnl = rv_pnl - (change * bump_pnl* 10000)
        merged.at[index,'delta_pnl'] =round(change * bump_pnl* 10000, 6)*100
        merged.at[index, 'rv_pnl'] = delta_rv_pnl-prev_pv
        merged.at[index, 'gamma_pnl'] = round(g_pnl,6)*100



        # VEGA PNL (VEGA + VOLGA)
        vol_bump = prev_vol+ 0.0000001
        vol_change = vol-prev_vol
        bump = binomial(S0 = prev_S0, r =prev_r, sigma= vol_bump, K = K, exp_date = exp, opttype= flag, start = prev_start )
        vega_rv_pv = binomial(S0 = prev_S0, r =prev_r, sigma= vol, K = K, exp_date = exp, opttype= flag, start = prev_start )
        bump_pnl = bump - prev_pv
        v_rv_pnl = vega_rv_pv-prev_pv #PNL from Vol change at T-0
        vega_pnl = vol_change * bump_pnl* 10000000
        volga_pnl = v_rv_pnl - vega_pnl #PNL from Vol Change (thru reval) - Vol Spot change
        
        merged.at[index,'vega_pnl'] = round(vega_pnl, 6)*100
        merged.at[index, 'vega_rv_pnl'] = vega_rv_pv-prev_pv
        merged.at[index, 'volga_pnl'] = round(volga_pnl,6)*100



        # RHO PNL

        rho_prev_rv = binomial(S0 = prev_S0, r =prev_r, sigma= prev_vol, K = K, exp_date = exp, opttype= flag, start = prev_start )
        rho_shock_rv = binomial(S0 = prev_S0, r =r, sigma= prev_vol, K = K, exp_date = exp, opttype= flag, start = prev_start )

        rho_pnl = rho_shock_rv - rho_prev_rv
        merged.at[index, 'rho_pnl'] = round(rho_pnl,6)*100

        # TOTAL ATTRIBUTION
        total_att = rho_pnl + volga_pnl + vega_pnl + g_pnl + delta_pnl + theta_pnl
        merged.at[index, 'total_att'] = round(total_att,6)*100

        #UNEXPLAIND PNL
        unexplained_pnl = -option_change + total_att
        merged.at[index, 'unexplained_pnl'] = round(unexplained_pnl, 6)*100
        merged.at[index, 'option_change'] = option_change*100
        merged.at[index, 'option_close'] = market_price*100
        merged.sort_values(by = 'Date', ascending = True, inplace = True)
    return merged