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

def parse_signal_id(id):
    if 'SHORT' in id:
        return  dict(direction = id[-5:], date = pd.to_datetime(id[-13:-5]), ticker = id[:-13])
    elif 'LONG' in id:
        return  dict(direction = id[-4:], date = pd.to_datetime(id[-12:-4]), ticker = id[:-12])
    else:
        raise ValueError(f'Invalid signal id `{id}`, neither LONG nor SHORT was found in the id')
    
