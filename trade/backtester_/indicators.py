import pandas as pd
import pandas_ta as ta
from backtesting.lib import crossover # noqa: F401
from copy import deepcopy
import numpy as np
import warnings
from typing import Optional
warnings.filterwarnings("ignore")

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


def compute_atr_loss(
    df: pd.DataFrame,
    *,
    atr_period: int = 20,
    atr_factor: float = 3.0,
    trail_type: str = "m",  # "m" modified, "u" unmodified
    average_type: str = "w",  # "w" wilders, "e"/"h"/"s" if you want
) -> pd.Series:
    """
    Computes loss = atr_factor * ATR, once. Use your existing TR + averaging code.
    Returns a pd.Series aligned to df.index.
    """
    d = df.copy()
    d.columns = list(map(str.lower, d.columns))

    # --- reuse your TR logic ---
    hi_lo = np.minimum(
        d["high"] - d["low"], 1.5 * d["high"].sub(d["low"]).rolling(atr_period).mean()
    )
    h_ref = np.where(
        d["low"] <= d["high"].shift(1),
        d["high"] - d["close"].shift(1),
        (d["high"] - d["close"].shift(1)) - 0.5 * (d["low"] - d["high"].shift(1)),
    )
    l_ref = np.where(
        d["high"] >= d["low"].shift(1),
        d["close"].shift(1) - d["low"],
        (d["close"].shift(1) - d["low"]) - 0.5 * (d["low"].shift(1) - d["high"]),
    )

    if trail_type == "m":
        tr = np.maximum(hi_lo, np.maximum(h_ref, l_ref))
        tr = pd.Series(tr, index=d.index)
    elif trail_type == "u":
        tr = np.maximum(
            np.maximum(d["high"] - d["low"], (d["high"] - d["close"].shift(1)).abs()),
            (d["low"] - d["close"].shift(1)).abs(),
        )
        tr = pd.Series(tr, index=d.index)
    else:
        raise ValueError("trail_type must be 'm' or 'u'")

    # average_type dispatch (uses your existing funcs)
    if average_type == "w":
        atr = wilders_average(tr, atr_period)
    elif average_type == "e":
        atr = exponential_average(tr, atr_period)
    elif average_type == "h":
        atr = hull_average(tr, atr_period)
    elif average_type == "s":
        atr = simple_average(tr, atr_period)
    else:
        raise ValueError("average_type must be 'w','e','h','s'")

    loss = atr_factor * atr
    return loss


def update_atr_trail_long(
    *,
    close: float,
    loss: float,
    prev_trail: Optional[float],
    reset: bool = False,
) -> float:
    """
    O(1) trailing stop update for LONG positions.

    Parameters
    ----------
    close : float
        Current bar close price.
    loss : float
        Stop distance for this bar (typically atr_factor * ATR).
    prev_trail : Optional[float]
        Previous trailing stop level (None / nan if uninitialized).
    reset : bool
        If True, initializes the trail fresh from this bar (used on entry).

    Returns
    -------
    float
        Updated trailing stop level for a LONG position.
    """
    if np.isnan(loss):
        return np.nan

    candidate = close - loss  # today’s raw stop level

    if (
        reset
        or prev_trail is None
        or (isinstance(prev_trail, float) and np.isnan(prev_trail))
    ):
        # Initialize at entry (or first valid ATR bar)
        return candidate

    # Ratchet up only (never decreases)
    return max(prev_trail, candidate)


def update_atr_trail_short(
    *,
    close: float,
    loss: float,
    prev_trail: Optional[float],
    reset: bool = False,
) -> float:
    """
    O(1) trailing stop update for SHORT positions.
    """
    if np.isnan(loss):
        return np.nan

    candidate = close + loss

    if (
        reset
        or prev_trail is None
        or (isinstance(prev_trail, float) and np.isnan(prev_trail))
    ):
        return candidate

    # Ratchet down only (never increases)
    return min(prev_trail, candidate)


def wilders_average(data, length):
    alpha = 1 / length
    return data.ewm(alpha=alpha, adjust=False).mean()


def exponential_average(data, length):
    alpha = 2 / (length + 1)
    return data.ewm(alpha=alpha, adjust=False).mean()


def hull_average(data, length):
    half_length = int(length / 2)
    wma1 = wilders_average(data, half_length * 2)
    wma2 = wilders_average(data, half_length)
    return wilders_average(2 * wma1 - wma2, int(np.sqrt(length)))


def simple_average(data, length):
    return data.rolling(length).mean()


def atr_trailing_stop(df: pd.DataFrame,
                      trail_type: str = 'm',
                      atr_period: int = 20,
                      atr_factor: float = 3.0,
                      first_trade: str = 'long',
                      average_type: str = 'w'):
    """
    This function creates a trailing stop value based on the Average True Range. It is continuously updatingf as price moves

    Args:
    df: The df containing the close, high and low of the given underlier
    trail_type: modified or unmodified. It defaults to modified. Pass the first letter of the options 'm' or 'u'
    atr_period: Length of period to be used in the calculation. 
    atr_factor: Multiplier for Average True Range (atr)
    first_trade: The starting trade. Long or Short
    average_type: Methodology of average to be used. Defaults to wilders. Available options, Exponential ('e'), Hull ('h'), Simple ('s').


    """

    assert atr_factor > 0, "'atr factor' must be positive: " + str(atr_factor)
    df = deepcopy(df)
    df.columns = list(map(str.lower, df.columns))
    hi_lo = np.minimum(df['high'] - df['low'], 1.5 *
                       df['high'].sub(df['low']).rolling(atr_period).mean())

    h_ref = np.where(df['low'] <= df['high'].shift(1), df['high'] - df['close'].shift(
        1), (df['high'] - df['close'].shift(1)) - 0.5 * (df['low'] - df['high'].shift(1)))

    l_ref = np.where(df['high'] >= df['low'].shift(1), df['close'].shift(
        1) - df['low'], (df['close'].shift(1) - df['low']) - 0.5 * (df['low'].shift(1) - df['high']))

    if trail_type == 'm':
        true_range = np.maximum(hi_lo, np.maximum(h_ref, l_ref))
    elif trail_type == 'u':
        true_range = np.maximum(np.maximum(df['high'] - df['low'], np.abs(df['high'] - df['close'].shift(1))),
                                np.abs(df['close'] - df['low']))
    else:
        raise ValueError("Invalid trail_type")

    if average_type == 'w':
        loss = atr_factor * wilders_average(true_range, atr_period)
    elif average_type == 'e':
        loss = atr_factor * exponential_average(true_range, atr_period)
    elif average_type == 'h':
        loss = atr_factor * hull_average(true_range, atr_period)
    elif average_type == 's':
        loss = atr_factor * simple_average(true_range, atr_period)
    else:
        raise ValueError(
            "Invalid average type. Choose 'wilders', 'exponential', 'hull', or 'simple'.")
    state = pd.Series(index=df.index, dtype='object')
    trail = pd.Series(index=df.index)

    for i in range(1, len(df)):

        if pd.isna(loss[i]):
    
            state.iloc[i] = 'init'
            trail.iloc[i] = np.nan
        else:
    
            if state.iloc[i - 1] == 'init':
                if first_trade == 'long':
                    state.iloc[i] = 'long'
                    trail.iloc[i] = df['close'].iloc[i] - loss[i]

                elif first_trade == 'short':
                    state.iloc[i] = 'short'
                    trail.iloc[i] = df['close'].iloc[i] + loss[i]

            elif state.iloc[i - 1] == 'long':
                if df['close'].iloc[i] > trail.iloc[i - 1]:
                    state.iloc[i] = 'long'
                    trail.iloc[i] = max(trail.iloc[i - 1],
                                        df['close'].iloc[i] - loss[i])

                else:
                    state.iloc[i] = 'short'
                    trail.iloc[i] = df['close'].iloc[i] + loss[i]

            elif state.iloc[i - 1] == 'short':
                if df['close'].iloc[i] < trail.iloc[i - 1]:
                    state.iloc[i] = 'short'
                    trail.iloc[i] = min(trail.iloc[i - 1],
                                        df['close'].iloc[i] + loss[i])
                else:
                    state.iloc[i] = 'long'
                    trail.iloc[i] = df['close'].iloc[i] - loss[i]

    buy_signal = (state == 'long') & (state.shift() != 'long')
    sell_signal = (state == 'short') & (state.shift() != 'short')

    results_df = pd.DataFrame(
        {'TrailingStop': trail, 'BuySignal': buy_signal, 'SellSignal': sell_signal, 'State': state})
    df = pd.concat([df, results_df], axis=1)

    return df