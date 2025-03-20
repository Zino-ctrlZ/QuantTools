class YFinanceEmptyData(Exception):
    pass

class OpenBBEmptyData(Exception):
    pass

class SymbolChangeError(Exception):
    pass

def raise_tick_name_change(tick, new_tick):
    raise SymbolChangeError(f"Tick name changed from {tick} to {new_tick}, access the new tick instead")