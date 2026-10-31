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


def _parse_signal_date(date_str: str) -> pd.Timestamp:
    """Parse the YYYYMMDD date slice of a signal id.

    Args:
        date_str: The 8-character date portion of a signal id.

    Returns:
        The date as a ``pd.Timestamp``.

    Raises:
        ValueError: If neither the fast path nor ``pd.to_datetime`` can parse it.
    """
    ## generate_signal_id always writes YYYYMMDD, so build the Timestamp from the digits.
    ## pd.to_datetime guesses the format on every call (about 1 ms); this is about 2 us.
    ## Deliberately not trade.helpers.helper.to_datetime: it tries %Y-%m-%d first and
    ## then guesses, which is the slow path this avoids.
    if len(date_str) == 8 and date_str.isdigit():
        try:
            return pd.Timestamp(year=int(date_str[:4]), month=int(date_str[4:6]), day=int(date_str[6:]))
        except ValueError:
            pass
    ## Anything else keeps the original behavior, including its errors.
    return pd.to_datetime(date_str)


def parse_signal_id(id):
    """Split a signal id into direction, date, ticker, and optional strategy slug.

    Args:
        id: Signal id like ``AAPL20200102LONG`` or ``slug::AAPL20200102SHORT``.

    Returns:
        Dict with ``direction``, ``date`` (``pd.Timestamp``), ``ticker``, and ``strategy_slug``.

    Raises:
        ValueError: If the id has neither LONG nor SHORT, or its date does not parse.
    """
    if "::" in id:
        strategy_slug, id = id.split("::", 1)  # Remove strategy slug if present
    else:
        strategy_slug = None
    if "SHORT" in id:
        return dict(direction=id[-5:], date=_parse_signal_date(id[-13:-5]), ticker=id[:-13], strategy_slug=strategy_slug)
    elif "LONG" in id:
        return dict(direction=id[-4:], date=_parse_signal_date(id[-12:-4]), ticker=id[:-12], strategy_slug=strategy_slug)
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
