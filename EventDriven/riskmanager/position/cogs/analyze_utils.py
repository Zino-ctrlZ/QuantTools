import pandas as pd
from EventDriven.riskmanager.utils import parse_position_id
from EventDriven.riskmanager.actions import HOLD, ROLL, EXERCISE, ADJUST, RMAction, Changes
from trade.helpers.helper import parse_option_tick, generate_option_tick_new
from .vars import MEASURES
from EventDriven.configs.core import StrategyLimitsEnabled
from EventDriven._vars import load_riskmanager_cache
from trade.helpers.Logging import setup_logger




SPLITS_CACHE = load_riskmanager_cache("splits_raw")
ADJUSTED_STRIKE_CACHE = load_riskmanager_cache("adjusted_strike_cache")
logger = setup_logger("EventDriven.riskmanager.position.cogs.analyze_utils", stream_log_level="WARNING")


def get_dte_and_moneyness_from_trade_id(
    trade_id: str,
    check_date: pd.Timestamp,
    check_price: float,
    start: pd.Timestamp = None,
    is_backtest: bool = True,
) -> tuple[int, list[float]]:
    """
    Combined function to parse trade_id once and compute both DTE and moneyness.
    This eliminates redundant parsing that occurred when calling get_dte_from_trade_id 
    and get_moneyness_from_trade_id separately.
    
    Returns:
        tuple: (dte, moneyness_list)
    """
    parsed_trade_id = parse_position_id(trade_id)[0]
    dte_list = []
    moneyness_list = []
    
    for meta_list in parsed_trade_id.values():
        for meta in meta_list:
            # Calculate DTE
            dte_list.append((pd.to_datetime(meta["exp_date"]).date() - check_date.date()).days)
            
            # Calculate moneyness with optional corporate event adjustment
            if start:
                meta = adjust_for_events(start=start, date=check_date, option=meta, is_backtest=is_backtest)
            
            strike = meta["strike"]
            option_type = meta["put_call"]
            if option_type.lower() == "c":
                moneyness = check_price / strike
            else:
                moneyness = strike / check_price
            moneyness_list.append(moneyness)
    
    dte = min(dte_list)
    return dte, moneyness_list


def get_dte_from_trade_id(trade_id: str, check_date: pd.Timestamp) -> int:
    """Legacy function - consider using get_dte_and_moneyness_from_trade_id for better performance"""
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
    """Legacy function - consider using get_dte_and_moneyness_from_trade_id for better performance"""
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
        is_backtest: bool = True,
    ):
        """
        Adjusts the option tick for events like splits or dividends.
        This function searches for splits in the SPLITS_CACHE and adjusts the strike price accordingly.
        Args:
            start (str): The start date of the position.
            date (str): The date for which the adjustment is to be made.
            option (str | dict): The option tick as a string or a dictionary.
            is_backtest (bool): Flag indicating if the context is backtesting or live trading.
        Returns:
            dict: The adjusted option tick as a dictionary. If no adjustment is made, returns the original option tick as a dictionary.
        """
        if isinstance(option, str):
            meta = option
        elif isinstance(option, dict):
            meta = generate_option_tick_new(
                symbol=option["ticker"],
                exp=option["exp_date"],
                strike=option["strike"],
                right=option["put_call"],
            )
        else:
            raise ValueError("Option must be a string or a dictionary.")

        strike_series = ADJUSTED_STRIKE_CACHE.get(meta, None)
        if strike_series is None:
            if is_backtest:
                raise ValueError(f"No adjusted strike data found for option: {meta}")
            else:
                logger.warning(f"No adjusted strike data found for option: {meta}. Returning original meta.")
                meta_dict = parse_option_tick(meta)
                return meta_dict
            
        adj_strike = strike_series.get(pd.to_datetime(date), None)
        if adj_strike is not None:
            meta_dict = parse_option_tick(meta)
            meta_dict["strike"] = adj_strike
            return meta_dict
        else:
            if is_backtest:
                raise ValueError(f"No adjusted strike found for option: {meta} on date: {date}")
            else:
                logger.warning(f"No adjusted strike found for option: {meta} on date: {date}. Returning original meta.")
                meta_dict = parse_option_tick(meta)
                return meta_dict
        # split = SPLITS_CACHE.get(swap_ticker(meta["ticker"]), None)
        # if split is None:
        #     return meta
        # for pack in split:
        #     if compare_dates.is_before(start, pack[0]) and compare_dates.is_after(date, pack[0]):
        #         print(f"Adjusting {meta['ticker']} strike {meta['strike']} for event on {pack[0]} with factor {pack[1]}")
        #         meta["strike"] /= pack[1]
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
        logger.info(f"Greek Check: greek_value={greek_value}, greek_threshold={greek_threshold}, per_greek={per_greek}, _bool={_bool}")
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
    tick: str = None,
    signal_id: str = None,
) -> RMAction:
    """
    Analyzing position data for a given date.
    Optimized with early returns based on action priority to eliminate list building and sorting.
    
    Priority order:
    1. EXERCISE (highest)
    2. ROLL (DTE or Moneyness)
    3. ADJUST (Greek limits)
    4. HOLD (default)
    """

    logger.info(f"Analyzing position {trade_id} at tick {tick} with signal {signal_id}")
    
    ## P1: EXERCISE Check (highest priority - return immediately if triggered)
    if strategy_enabled_actions.exercise:
        if dte - t_plus_n <= 0:  # Inline the check for performance
            action = EXERCISE(
                trade_id=trade_id,
                action=Changes(quantity_diff=qty, new_quantity=0)
            )
            action.reason = f"position is expiring (DTE: {dte} <= {t_plus_n})"
            logger.debug(f"EXERCISE action for {trade_id} due to expiration.")
            return action

    ## P2: ROLL Checks (second priority - return immediately if triggered)
    # DTE Check
    if strategy_enabled_actions.dte:
        if dte < dte_limit:  # Inline the check for performance
            action = ROLL(
                trade_id=trade_id,
                action=Changes(quantity_diff=0, new_quantity=qty)
            )
            action.reason = f"not enough DTE ({dte} < {dte_limit})"
            logger.debug(f"ROLL action for {trade_id} due to DTE check.")
            return action

    # Moneyness Check
    if strategy_enabled_actions.moneyness:
        if any(abs(m) > abs(moneyness_limit) for m in moneyness_list):  # Inline the check
            m = min(moneyness_list, key=abs)
            action = ROLL(
                trade_id=trade_id,
                action=Changes(quantity_diff=0, new_quantity=qty)
            )
            action.reason = f"position is too ITM ({m} exceeds {moneyness_limit})"
            logger.debug(f"ROLL action for {trade_id} due to moneyness check.")
            return action

    ## P3: ADJUST Checks (lowest actionable priority)
    ## Find the most significant Greek limit breach (max quantity adjustment needed)
    max_adjust_action = None
    max_qty_diff = 0
    
    for greek in MEASURES:
        # Skip if not enabled
        if not strategy_enabled_actions.__dict__.get(greek, False):
            continue
        
        # Get Greek value and limit
        greek_v = greeks.get(greek)
        greek_limit_v = position_greek_limit.get(greek)
        
        # Skip if either is None
        if greek_v is None or greek_limit_v is None:
            continue
        
        # Check if limit is breached
        _greek_bool, q_diff = greek_check(
            greek_threshold=greek_limit_v,
            greek_value=greek_v,
            greater_than=True,
            qty=qty
        )
        
        if _greek_bool:
            abs_q_diff = abs(q_diff)
            # Keep track of the largest adjustment needed
            if abs_q_diff > max_qty_diff:
                max_qty_diff = abs_q_diff
                # Adjust sign based on position direction
                q_diff = abs_q_diff if qty < 0 else -abs_q_diff
                max_adjust_action = ADJUST(
                    trade_id=trade_id,
                    action=Changes(quantity_diff=q_diff, new_quantity=qty + q_diff)
                )
                max_adjust_action.reason = f"position {greek} exceeds limit ({greek_v} > {greek_limit_v})"
                logger.debug(f"ADJUST action candidate for {trade_id}: {greek} limit breach.")
        else:
            logger.debug(f"No {greek} limit breach for {trade_id}: {greek_v} within {greek_limit_v}.")
    
    # Return the most significant ADJUST action if found
    if max_adjust_action:
        return max_adjust_action
    
    ## P4: HOLD (default - no limits breached)
    action = HOLD(trade_id=trade_id, action=Changes(quantity_diff=0, new_quantity=qty))
    action.reason = "position within limits"
    return action
