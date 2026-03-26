import pandas as pd
from typing import Tuple
from trade.helpers.helper import parse_option_tick
from trade.helpers.Logging import setup_logger

logger = setup_logger("EventDriven.helpers")


def generate_signal_id(underlier, date, signal_type):
    signal_date = pd.to_datetime(date).strftime("%Y%m%d")
    key = underlier.upper() + signal_date + signal_type
    return key


def normalize_dollar_amount_to_decimal(price: float | int) -> float | int:
    """
    divide by 100
    """
    return price / 100


def normalize_dollar_amount(price: float | int) -> float | int:
    """
    multiply by 100
    """
    return price * 100


def parse_signal_id(id):
    if "SHORT" in id:
        return dict(direction=id[-5:], date=pd.to_datetime(id[-13:-5]), ticker=id[:-13])
    elif "LONG" in id:
        return dict(direction=id[-4:], date=pd.to_datetime(id[-12:-4]), ticker=id[:-12])
    else:
        raise ValueError(f"Invalid signal id `{id}`, neither LONG nor SHORT was found in the id")


def parse_position_id(positionID: str) -> Tuple[dict, list]:
    position_str = positionID
    position_list = position_str.split("&")
    position_list = [x.split(":") for x in position_list if x]
    position_list_parsed = [(x[0], parse_option_tick(x[1])) for x in position_list]
    position_dict = dict(L=[], S=[])
    for x in position_list_parsed:
        position_dict[x[0]].append(x[1])
    return position_dict, position_list
