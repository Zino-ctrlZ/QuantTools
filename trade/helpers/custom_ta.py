import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


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
                      atr_period: str = 20,
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
    average_type: Methodology of average to be used. Defaults to wilders. Available options, Exponential ('e'), Hull ('e'), Simple ('s').


    """

    assert atr_factor > 0, "'atr factor' must be positive: " + str(atr_factor)
    df = df.sort_values(by='timestamp', ascending=True, ignore_index=True)
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
    # print('hi',loss)
    state = pd.Series(index=df.index, dtype='object')
    trail = pd.Series(index=df.index)

    for i in range(1, len(df)):

        if pd.isna(loss[i]):
            # print(i, len(loss), len(true_range))
            state.iloc[i] = 'init'
            trail.iloc[i] = np.nan
        else:
            # print(loss[i])
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
