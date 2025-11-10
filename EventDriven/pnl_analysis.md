## Problem of Growing Delta:
```python
key = 10
Ticker = AAPL
Trade Details = &L:AAPL20240119C210&S:AAPL20240119C245	2023-02-03	2023-02-27	AAPL

Backtester Attributes:
evb_backtest.portfolio.risk_manager.OrderPicker.liquidity_threshold = 50
evb_backtest.portfolio.risk_manager.OrderPicker.lookback = 10
evb_backtest.portfolio.risk_manager.OrderPicker.data_availability_threshold = 0.5
evb_backtest.portfolio.order_settings = {'type': 'naked',
 'specifics': [{'direction': 'long',
   'rel_strike': .70,
   'dte': 365,
   'moneyness_width': 0.10},
   {'direction': 'short',
  'rel_strike': .65,
  'dte': 365,
  'moneyness_width': 0.10}
],
 'name': 'vertical_spread'}


evb_backtest.portfolio.max_contract_price = max_cash (2)
evb_backtest.executor.commission_rate = 0.65/100
evb_backtest.portfolio.min_moneyness_threshold = 5
evb_backtest.executor.max_slippage_pct = 0.075
evb_backtest.portfolio.roll_map = 30
evb_backtest.portfolio.moneyness_width_factor = .025
evb_backtest.portfolio.dte_reduction_factor = 30
evb_backtest.portfolio.min_acceptable_dte_threshold = 180
```

-  Potential Solution:
  - On Vector test (refer to `EventDriven/demos/pnl_analysis/1_aapl_analysis_down_pnl.ipynb`), PnL improves after reducing quantity when delta breaches a limit.
  - Would need to test include commission & slippage to be certain of change