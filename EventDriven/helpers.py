import pandas as pd

def generate_signal_id(underlier, date, signal_type):
    signal_date = pd.to_datetime(date).strftime('%Y%m%d')
    key = underlier.upper() + signal_date + signal_type
    return key


def normalize_dollar_amount_to_decimal(price: float | int ) -> float | int:
    """
    divide by 100
    """
    return price / 100

def normalize_dollar_amount( price: float | int) -> float | int :
    """
    multiply by 100
    """
    return price * 100