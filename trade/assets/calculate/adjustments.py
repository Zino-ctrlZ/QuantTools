from typing import List
import pandas as pd
from trade.assets.calculate.data_classes import TradePnlInfo
from trade.helpers.Logging import setup_logger
logger = setup_logger('trade.assets.calculate.adjustments')

def trade_pnl_adjustment(
    attribution_table: pd.DataFrame,
    entry_info: List[TradePnlInfo],
    scale_qty: float = 1.0
):
    """
    Adjust the attribution table for trade PnL.
    Args:
        attribution_table (pd.DataFrame): The original attribution table.
        entry_info (Dict[str, Tuple[float, datetime]]): A dictionary containing
            trade entry information with keys as identifiers and values as tuples
            of (entry_price, entry_date).
    Returns:
        pd.DataFrame: The adjusted attribution table.
    """ 
    if 'trade_pnl_adjustment' not in attribution_table.columns:
        attribution_table['trade_pnl_adjustment'] = 0.0

    for info in entry_info:
        trade_pnl = info.calculate_trade_pnl() * scale_qty
        effect_date = info.effect_date

        # Update the attribution table
        if effect_date in attribution_table.index:
            attribution_table.at[effect_date, 'trade_pnl_adjustment'] += trade_pnl
        else:
            logger.warning("Effect date %s not in attribution table index.", effect_date)
    attribution_table['total_pnl'] = attribution_table['total_pnl_excl_trade_pnl'] + attribution_table['trade_pnl_adjustment']

    return attribution_table
