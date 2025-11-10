#Load Backtest class 
from EventDriven.backtest import OptionSignalBacktest
from EventDriven.riskmanager.sizer import DefaultSizer, ZscoreRVolSizer, BaseSizer
import pandas as pd
pd.options.display.max_rows = 50
pd.options.display.max_columns = 50

def create_backtest_object(cash, weights, trades_, max_cash):
    evb_backtest = OptionSignalBacktest(trades_, initial_capital=cash, t_plus_n=1, symbol_list = list(weights.keys()) )
    evb_backtest.portfolio.initial_capital
    w_map = {x: w  * 0.95 for x, w in weights.items()}
    evb_backtest.portfolio.weight_map = w_map
    evb_backtest.portfolio.weight_map
    evb_backtest.portfolio.risk_manager.OrderPicker.liquidity_threshold = 50
    evb_backtest.portfolio.risk_manager.OrderPicker.lookback = 10
    evb_backtest.portfolio.risk_manager.sizing_lev = 4.5
    evb_backtest.portfolio.max_contract_price_factor = 2
    evb_backtest.portfolio.min_moneyness_threshold = 3
    evb_backtest.portfolio.risk_manager.OrderPicker.data_availability_threshold = 0.5
    order_settings =  {
                'type': 'spread',
                'specifics': [
                    {'direction': 'long', 'rel_strike': 1.0, 'dte': 365, 'moneyness_width': 0.1},
                    {'direction': 'short', 'rel_strike': 0.85, 'dte': 365, 'moneyness_width': 0.1} 
                ],
                'name': 'vertical_spread',
                'strategy': 'vertical',
                'target_dte': 360,
                'structure_direction': 'long',
                'spread_ticks': 1,
                'dte_tolerance': 60,
                'min_moneyness': 0.65,
                'max_moneyness': 1.,
                'min_total_price': 0.95
            }
    evb_backtest.portfolio.order_settings = order_settings
    evb_backtest.portfolio.risk_manager.max_dte_tolerance = order_settings['target_dte'] - 240
    evb_backtest.portfolio.risk_manager.max_tries = 15
    evb_backtest.portfolio.max_contract_price = max_cash
    evb_backtest.executor.commission_rate = 0.65/100
    evb_backtest.portfolio.min_moneyness_threshold = 5
    evb_backtest.executor.max_slippage_pct = 0.075
    evb_backtest.portfolio.roll_map = 180
    evb_backtest.portfolio.moneyness_width_factor = .025
    evb_backtest.portfolio.dte_reduction_factor = 30
    evb_backtest.portfolio.min_acceptable_dte_threshold = 95
    evb_backtest.portfolio.risk_manager.limits['dte'] = True
    evb_backtest.portfolio.risk_manager.limits['delta'] = True
    evb_backtest.portfolio.risk_manager.limits['moneyness'] = True
    evb_backtest.portfolio.risk_manager.max_moneyness = 1.15
    evb_backtest.portfolio.risk_manager.max_slippage = 0.075
    evb_backtest.portfolio.risk_manager.otm_moneyness_width = 0.45
    evb_backtest.portfolio.risk_manager.itm_moneyness_width = 0.10
    evb_backtest.portfolio.risk_manager.re_update_on_roll = False
    evb_backtest.portfolio.risk_manager.t_plus_n = 1
    for key  in max_cash:
        if max_cash[key]*100 > evb_backtest.portfolio.allocated_cash_map[key]:
            # print(key, max_cash[key]*100, evb_backtest.portfolio.allocated_cash_map[key])
            pass

    return evb_backtest
