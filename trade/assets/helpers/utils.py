from trade.helpers.types import TickerMap
INVALID_TICKERS = {'FB': 'META'}
TickerMap.invalid_tickers = INVALID_TICKERS

## Key is (OLD, NEW, DATE)
TICK_CHANGE_ALIAS = TickerMap({
    'META': ('FB', 'META', '2022-06-09'),
})

def verify_ticker(ticker):
    if ticker in INVALID_TICKERS:
        raise_tick_name_change(ticker, INVALID_TICKERS[ticker])
    return ticker

