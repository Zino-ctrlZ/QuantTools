universe = {'top_10_snP': ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'TSLA', 'BRK.B', 'UNH', 'XOM', 'LLY'],
'blue_chip_largest_vol': ['AAPL', 'MSFT', 'SOFI', 'GOOGL', 'F', 'AMD','BAC', 'PFE', 'INTC', 'PLTR'],
'most_traded_etf': ['SPY', 'XLF', 'TLT', 'QQQ', 'FXI', 'HYG', 'SLV', 'IWM', 'EEM', 'GDX'] ,# NO INVERSES, NO CRYPTO, NO DAILY DIRECTION,
'top_10_snP_excl_wildMovers': ['AAPL', 'MSFT', 'AMZN', 'JPM', 'GOOGL', 'JNJ', 'BRK.B', 'UNH', 'XOM', 'LLY'],
'small_volatile': [ 'LRHC','KAVL', 'CNSP', 'SVMH', 'GOEV', 'SDOT', 'NXL', 'QXO','VLCN', 'KYTX'],
'top_as_at_2019': ['AAPL', 'MSFT', 'AMZN', 'BRK.B', 'FB', 'META', 'GOOGL', 'JNJ', 'XOM','JPM','V', 'PG', 'DIS', 'BAC' ]
}

sp500 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "BRK.B", "META", "TSLA", "XOM",
    "UNH", "JNJ", "JPM", "V", "PG", "MA", "CVX", "HD", "LLY", "PFE",
    "KO", "BAC", "PEP", "AVGO", "COST", "ABBV", "CSCO", "MRK", "DIS", "TMO",
    "GS", "ABT", "CMCSA", "BA", "SCHW", "C", "MCD", "VZ", "TXN", "AMD",
    "CRM", "QCOM", "HON", "UNP", "INTC", "NFLX", "AXP", "WMT", "NKE", "TRV",
    "MDT", "NEE", "LOW", "AMGN", "UPS", "MS", "PM", "ORCL", "RTX", "SPGI",
    "CVS", "BLK", "LIN", "T", "USB", "GS", "AMT", "BKNG", "CAT", "DE",
    "ADBE", "PYPL", "SBUX", "INTU", "ISRG", "MDLZ", "GE", "MMM", "SYK", "LMT",
    "CI", "MO", "DUK", "SO", "PLD", "BDX", "TJX", "ADP", "CB", "MMC",
    "CCI", "APD", "DHR", "ITW", "PNC", "NSC", "WM", "SHW", "EL", "FIS"
]


qqq  =[
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOG", "META", "TSLA", "GOOGL", "AVGO", "PEP",
    "COST", "ADBE", "CSCO", "TXN", "QCOM", "AMD", "CMCSA", "NFLX", "INTC", "HON",
    "AMGN", "INTU", "SBUX", "ISRG", "MDLZ", "GE", "MMM", "SYK", "LMT", "CI",
    "MO", "DUK", "SO", "PLD", "BDX", "TJX", "ADP", "CB", "MMC", "CCI",
    "APD", "DHR", "ITW", "PNC", "NSC", "WM", "SHW", "EL", "FIS", "PSA",
    "AON", "GD", "BK", "USB", "MET", "ICE", "ECL", "TFC", "COF", "AIG",
    "ALL", "MCO", "MSCI", "TRV", "SPGI", "CME", "AFL", "PGR", "PRU", "STT",
    "BKNG", "CAT", "DE", "ADBE", "PYPL", "SBUX", "INTU", "ISRG", "MDLZ", "GE",
    "MMM", "SYK", "LMT", "CI", "MO", "DUK", "SO", "PLD", "BDX", "TJX"
]

xle = [
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PXD", "VLO", "PSX", "OXY",
    "HAL", "KMI", "WMB", "BKR", "DVN", "HES", "OKE", "LNG", "APA", "CTRA",
    "MRO", "FANG", "TRGP", "AR", "SWN"
]

xop = [
    "MRO", "DVN", "EOG", "PXD", "COP", "OXY", "APA", "FANG", "HES", "CTRA",
    "AR", "SWN", "CLR", "MUR", "SM", "RRC", "OVV", "CPE", "MTDR", "PDCE",
    "LPI", "CDEV", "NOG", "CRK", "SBOW"
]

xlf = [
    "BRK.B", "JPM", "BAC", "WFC", "C", "MS", "GS", "AXP", "BLK", "SPGI",
    "CB", "MMC", "CCI", "APD", "DHR", "ITW", "PNC", "NSC", "WM", "SHW",
    "EL", "FIS", "PSA", "AON", "GD"
]


xlk = [
    "AAPL", "MSFT", "NVDA", "AVGO", "CSCO", "ADBE", "TXN", "QCOM", "AMD", "INTC",
    "CRM", "ORCL", "ACN", "IBM", "NOW", "AMAT", "ADI", "LRCX", "MU", "INTU",
    "MSI", "HPQ", "TEL", "GLW", "CDNS"
]


iwm = [
    "FTAI", "SFM", "PCVX", "INSM", "MLI", "AIT", "FLR", "FN", 
    "ENSG", "RVMD", "EXLS", "BECN", "CMC", "LSCC", "SAIA", "CELH", 
    "ALGM", "AXON", "SWAV", "GTLS", "HALO", "CRI", "LNTH", "PIPR", "KRG"
]

iwm2 = [
    "NVAX", "AMC", "SAGE", "PTON", "ZI", "DOCU", "FSLY", "CHWY", "CRWD", "DDOG",
    "NET", "SNOW", "PLTR", "RBLX", "UPST", "AFRM", "BILL", "ASAN", "MSTR", "ZI",
    "DOCU", "FSLY", "CHWY", "CRWD", "DDOG"
]


rates = [
    'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'AGG', 'SHY', 'SHV', 'JNK' ## Expand to include more
]

commodities = [
    'GLD', 'SLV', 'USO', 'DBA', 'DBO', 'DBC', 'UGA', 'UNG', 'GDX', 'XLP', 'XLE', 'XOP', 'CORN', 'WEAT', "WOOD"
]


basic_test_universe = {
    'US EQ': ['AAPL', 'MSFT', 'JNJ', 'JPM', 'SBUX', 'NVDA', 'BAC', 'BMY', 'CSCO', 'DVN'],
    'US RATES': ['TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'AGG', 'SHY', 'SHV', 'JNK'],
    'US COMMODITIES': ['GLD', 'SLV', 'USO', 'DBA',  'DBC', 'UNG', 'GDX','XLE', 'CORN', 'WEAT'],
}