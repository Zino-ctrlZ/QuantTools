from trade.helpers.types import TickerMap
INVALID_TICKERS = {'FB': 'META'}
TickerMap.invalid_tickers = INVALID_TICKERS

## Key is (OLD, NEW, DATE)
TICK_CHANGE_ALIAS = TickerMap({
    'META': ('FB', 'META', '2022-06-09'),
})

def raise_tick_name_change(ticker, new_ticker):
    raise ValueError(
        f'Ticker {ticker} has changed to {new_ticker}. Please use the new ticker.'
    )

def verify_ticker(ticker):
    if ticker in INVALID_TICKERS:
        raise_tick_name_change(ticker, INVALID_TICKERS[ticker])
    return ticker

def swap_ticker(ticker):
    """
    Swap ticker if it is in the TICK_CHANGE_ALIAS map.
    """
    if ticker in TickerMap.invalid_tickers:
        return TickerMap.invalid_tickers[ticker]
    return ticker