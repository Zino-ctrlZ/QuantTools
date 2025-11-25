import pandas as pd
from EventDriven.riskmanager.utils import parse_position_id
from EventDriven.riskmanager.actions import HOLD, ROLL, EXERCISE, ADJUST, RMAction, Changes
from trade.assets.helpers.utils import swap_ticker
from trade.helpers.helper import compare_dates, parse_option_tick
from .vars import ACTION_PRIORITY, MEASURES
from EventDriven.configs.core import StrategyLimitsEnabled
from EventDriven._vars import load_riskmanager_cache
from trade.helpers.Logging import setup_logger




SPLITS_CACHE = load_riskmanager_cache("splits_raw")
logger = setup_logger("EventDriven.riskmanager.position.cogs.analyze_utils", stream_log_level="WARNING")


def get_dte_from_trade_id(trade_id: str, check_date: pd.Timestamp) -> int:
    parsed_trade_id = parse_position_id(trade_id)[0]
    dte_list = []
    for meta_list in parsed_trade_id.values():
        for meta in meta_list:
            dte_list.append((pd.to_datetime(meta["exp_date"]).date() - check_date.date()).days)
    dte = min(dte_list)
    return dte


def get_moneyness_from_trade_id(
    trade_id: str,
    check_price: float,
    check_date: pd.Timestamp,
    start: pd.Timestamp = None,
) -> float:
    parsed_trade_id = parse_position_id(trade_id)[0]
    moneyness_list = []
    for meta_list in parsed_trade_id.values():
        for meta in meta_list:
            ## Adjust for corporate events if start and end dates are provided
            if start:
                meta = adjust_for_events(start=start, date=check_date, option=meta)

            strike = meta["strike"]
            option_type = meta["put_call"]
            if option_type.lower() == "c":
                moneyness = check_price / strike
            else:
                moneyness = strike / check_price
            moneyness_list.append(moneyness)
    return moneyness_list


def adjust_for_events(
        start: str,
        date: str,
        option: str | dict,
    ):
        """
        Adjusts the option tick for events like splits or dividends.
        """
        if isinstance(option, str):
            meta = parse_option_tick(option)
        elif isinstance(option, dict):
            meta = option
        else:
            raise ValueError("Option must be a string or a dictionary.")
        split = SPLITS_CACHE.get(swap_ticker(meta["ticker"]), None)
        if split is None:
            return meta
        for pack in split:
            if compare_dates.is_before(start, pack[0]) and compare_dates.is_after(date, pack[0]):
                meta["strike"] /= pack[1]
        return meta


def dte_check(dte: int, dte_threshold: int) -> bool:
    """Check if the position's days to expiration (DTE) is below a specified threshold.
    Args:
        dte (int): The days to expiration of the position.
        dte_threshold (int): The threshold value for DTE.
    Returns:
        bool: True if the position's DTE is below the threshold, False otherwise.
    """
    return dte < dte_threshold


def exercise_check(dte: int, t_plus_n: int) -> bool:
    """Check if the position is within a specified number of days to expiration (DTE).
    Args:
        position (EODPositionData): The position data containing DTE information.
        t_plus_n (int): The number of days to expiration threshold.
    Returns:
        bool: True if the position's DTE is less than or equal to t_plus_n, False otherwise.
    """
    dte = dte - t_plus_n
    return dte <= 0


def moneyness_check(moneyness: list, moneyness_threshold: float) -> bool:
    """Check if the position's moneyness is below a specified threshold.
    Args:
        moneyness (list): A list of moneyness values for the options in the position.
        moneyness_threshold (float): The threshold value for moneyness.
    Returns:
        bool: True if the position's moneyness is below the threshold, False otherwise.
    """

    return any(abs(m) > abs(moneyness_threshold) for m in moneyness)


def greek_check(greek_value: float, greek_threshold: float, qty: int = 1, greater_than: bool = True) -> bool:
    """Check if the position's Greek value is above or below a specified threshold.
    Args:
        greek_value (float): The Greek value of the position (e.g., delta, gamma, vega, theta).
        greek_threshold (float): The threshold value for the Greek.
        greater_than (bool): If True, check if the absolute Greek value is greater than the threshold.
                             else, check if it is less than the threshold.
                             True is for upper limits, False is for lower limits.
    Returns:
        bool: True if the condition is met, False otherwise.
    """
    if greater_than:
        per_greek = greek_value / qty
        _bool = abs(greek_value) > abs(greek_threshold)
        required_qty = max(int(abs(greek_threshold) // abs(per_greek)), 1)
        quantity_diff = abs(qty) - abs(required_qty)
        return _bool, quantity_diff
    else:
        return abs(greek_value) < abs(greek_threshold)


def analyze_position(
    dte: int,
    trade_id: str,
    position_greek_limit: dict,
    dte_limit: int,
    moneyness_limit: float,
    greeks: dict,
    qty: int,
    moneyness_list: list,
    strategy_enabled_actions: StrategyLimitsEnabled,
    t_plus_n: int,
) -> RMAction:
    """
    Analyzing position data for a given date.
    """

    position_actions = []
    ## DTE Check
    if strategy_enabled_actions.dte:
        if dte_check(dte, dte_limit):
            changes: Changes = {
                "quantity_diff": 0,
                "new_quantity": qty,
            }
            action = ROLL(trade_id=trade_id, action=changes)
            action.reason = f"not enough DTE ({dte} < {dte_limit})"
            logger.debug(f"Added ROLL action for {trade_id} due to DTE check.")
            position_actions.append(action)

    ## Exercise Check
    if strategy_enabled_actions.exercise:
        if exercise_check(dte, t_plus_n):
            changes = Changes(
                quantity_diff=qty,
                new_quantity=0,
            )
            action = EXERCISE(trade_id=trade_id, action=changes)
            action.reason = f"position is expiring (DTE: {dte} <= {t_plus_n})"
            logger.debug(f"Added EXERCISE action for {trade_id} due to Exercise check.")
            position_actions.append(action)

    ## Moneyness Check
    if strategy_enabled_actions.moneyness:
        if moneyness_check(moneyness_list, moneyness_limit):
            changes: Changes = {
                "quantity_diff": 0,
                "new_quantity": qty,
            }
            action = ROLL(trade_id=trade_id, action=changes)
            action.reason = f"position is too ITM (moneyness exceeds {moneyness_limit})"
            position_actions.append(action)

    ## Greek Checks
    ## Loop through all greek measures
    for greek in MEASURES:
        ## Skip if not enabled
        if strategy_enabled_actions.__dict__.get(greek, False) is False:
            logger.debug(f"Skipping greek check for {greek} on {trade_id} as it is not enabled in strategy_enabled_actions.")
            continue

        ## Skip if no limit set
        if greek not in position_greek_limit:
            logger.debug(f"Skipping greek check for {greek} on {trade_id} as no limit is set in position_greek_limit.")
            continue

        ## Run greek check if limit is set
        greek_v = greeks.get(greek)
        if greek_v is None:
            logger.debug(f"Skipping greek check for {greek} on {trade_id} as its value is None in greeks.")
            continue

        greek_limit_v = position_greek_limit[greek]
        if greek_limit_v is None:
            logger.debug(f"Skipping greek check for {greek} on {trade_id} as its limit value is None in position_greek_limit.")
            continue

        _greek_bool, q_diff = greek_check(
            greek_threshold=position_greek_limit[greek], greek_value=greek_v, greater_than=True, qty=qty
        )

        ## Positive q_diff if qty is negative (short position is reduced by buying)
        ## Negative q_diff if qty is positive (long position is reduced by selling)
        q_diff = abs(q_diff) if qty < 0 else -abs(q_diff)
        if _greek_bool:  ## Only upper limits for now
            changes = Changes(
                quantity_diff=q_diff,
                new_quantity=qty + q_diff,
            )
            action = ADJUST(trade_id=trade_id, action=changes)
            action.reason = f"position {greek} exceeds limit ({greeks[greek]} > {position_greek_limit[greek]})"
            position_actions.append(action)
            logger.debug(f"Added ADJUST action for {trade_id} due to {greek} limit exceedance.")
    
    ## Finalize action
    ## If no actions, HOLD
    if not position_actions:
        action = HOLD(trade_id=trade_id, action=Changes(quantity_diff=0, new_quantity=qty))
        action.reason = "position within limits"
        return action

    ## Else prioritize actions
    ## Prioritize actions:
    ## P1. EXERCISE
    ## P2. ROLL
    ## P3. ADJUST
    else:
        action_priority = ACTION_PRIORITY
        position_actions = sorted(position_actions, key=lambda x: action_priority[x.type])

        ## If multiple adjust actions, keep only one with max abs quantity diff
        if type(position_actions[0]) in [EXERCISE, ROLL]:
            return position_actions[0]
        elif type(position_actions[0]) == ADJUST: #noqa
            adjust_actions = [act for act in position_actions if type(act) == ADJUST] #noqa
            if len(adjust_actions) > 1:
                adjust_actions = sorted(adjust_actions, key=lambda x: abs(x.action["quantity_diff"]), reverse=True)
            return adjust_actions[0]
