import pandas as pd
import pandas_ta as ta
from backtesting.lib import crossover # noqa: F401

def create_bbands_dataframe(data, length=20, std=2) -> pd.DataFrame:
    bbands = ta.bbands(pd.Series(data["close"]), length=length, std=std)
    bbands_cols = ["bband_lower", "bband_mid", "bband_upper", "bband_wband", "bband_pband"]
    bbands.columns = bbands_cols
    return bbands


def create_atx_dataframe(data, length=15) -> pd.DataFrame:
    atx = ta.adx(pd.Series(data["high"]), pd.Series(data["low"]), pd.Series(data["close"]), length=length)
    atx_cols = ["adx", "dmp", "dmn"]
    atx.columns = atx_cols
    return atx


def create_atr_dataframe(data, length=21) -> pd.DataFrame:
    atr = ta.atr(pd.Series(data["high"]), pd.Series(data["low"]), pd.Series(data["close"]), length=length)
    atr = atr.to_frame(name="atr")
    return atr
