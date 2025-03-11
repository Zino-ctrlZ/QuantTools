import pandas as pd

def generate_signal_id(underlier, date, signal_type):
    signal_date = pd.to_datetime(date).strftime('%Y%m%d')
    key = underlier.upper() + signal_date + signal_type
    return key