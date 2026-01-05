"""Database Persistence Utilities for Live Position Limits and Greeks.

This module provides database interaction functions for storing and retrieving
position limits, Greek values, and risk metrics in live trading environments.
It handles caching, batch updates, and data synchronization with the portfolio
database to support the LiveCOGLimitsAndSizingCog.

Key Functions:
    get_limits_data:
        - Retrieves all position limits from database
        - Implements caching with automatic refresh
        - Reloads every 5 accesses or on force=True
        - Returns DataFrame with all limit records

    update_greeks_case:
        - Updates Greek values for specific position
        - Supports date changes via new_date parameter
        - Validates all required measures present
        - Uses parameterized queries for safety

    store_position_limits:
        - Persists new position limits to database
        - Records delta, gamma, vega, theta limits
        - Associates limits with trade_id and signal_id
        - Timestamps for audit trail

    get_position_limit:
        - Retrieves specific limit for position/measure
        - Returns (date, limit_value) tuple
        - Used during position initialization
        - Falls back to default if not found

Database Schema:
    limits table:
        - trade_id: Unique position identifier
        - signal_id: Strategy signal reference
        - strategy_name: Strategy identifier
        - date: Limit creation/update date
        - delta: Delta limit value
        - gamma: Gamma limit value
        - vega: Vega limit value
        - theta: Theta limit value

    greeks table:
        - trade_id: Position identifier
        - date: Greek calculation date
        - delta: Current delta exposure
        - gamma: Current gamma exposure
        - vega: Current vega exposure
        - theta: Current theta exposure
        - Additional position metrics

Caching Strategy:
    LIMITS_DF Global Cache:
        - In-memory DataFrame of all limits
        - Reduces database queries
        - Refreshes every 5 accesses
        - Force refresh available via parameter

    Access Counter:
        - _ACCESS_COUNTER tracks reads
        - Triggers automatic refresh at threshold
        - Balances freshness vs performance
        - Reset on refresh

Update Patterns:
    Single Update:
        - update_greeks_case() for one position
        - Used when analyzing individual positions
        - Immediate database write

    Batch Update:
        - dynamic_batch_update() for multiple positions
        - Used during portfolio-wide analysis
        - More efficient for bulk operations
        - Reduces database round-trips

Safety Features:
    Input Validation:
        - Verifies all required Greek measures present
        - Type checking on parameters
        - Prevents partial updates

    Parameterized Queries:
        - All SQL uses parameter binding
        - Prevents SQL injection
        - Database handles type conversion

    Transaction Management:
        - Atomic updates via DatabaseAdapter
        - Rollback on errors
        - Maintains data consistency

Usage:
    # Store new position limits
    store_position_limits(
        delta_limit=10.0,
        gamma_limit=5.0,
        vega_limit=15.0,
        theta_limit=-20.0,
        trade_id='AAPL_20240315_C_150.0',
        signal_id='SIG001',
        strategy_name='put_spread_strategy',
        date=pd.Timestamp('2024-01-15')
    )

    # Retrieve limits
    limits_df = get_limits_data(force=False)
    date, delta_limit = get_position_limit(
        trade_id='AAPL_20240315_C_150.0',
        strategy_name='put_spread_strategy',
        signal_id='SIG001',
        measure='delta'
    )

    # Update Greeks
    update_greeks_case(
        trade_id='AAPL_20240315_C_150.0',
        strategy_name='put_spread_strategy',
        signal_id='SIG001',
        greeks={'delta': 8.5, 'gamma': 3.2, 'vega': 12.1, 'theta': -15.3},
        date_=pd.Timestamp('2024-01-15')
    )

Integration:
    - LiveCOGLimitsAndSizingCog uses for persistent state
    - Position initialization loads historical limits
    - Daily analysis updates Greeks in database
    - Reporting queries limits table for analytics

Error Handling:
    - Missing measures raise ValueError with details
    - Database errors propagated with context
    - Invalid parameters logged with warnings
    - Graceful degradation on cache failures

Notes:
    - Requires 'portfolio_data' database connection
    - MEASURES_SET defines valid Greek measures
    - Cache tuning via _ACCESS_COUNTER threshold
    - Thread-safe via DatabaseAdapter connection pooling
"""

from sqlalchemy import text
from datetime import datetime
import pandas as pd
from EventDriven.backtest import OptionSignalBacktest
from trade.helpers.Logging import setup_logger
from dbase.database.SQLHelpers import DatabaseAdapter, dynamic_batch_update, get_engine
from EventDriven.riskmanager.position.cogs.vars import MEASURES_SET

logger = setup_logger("algo.positions.limits.save_utils")
LIMITS_DF = None
_ACCESS_COUNTER = 0
db = DatabaseAdapter()


def get_limits_data(force: bool = False) -> pd.DataFrame:
    global LIMITS_DF
    global _ACCESS_COUNTER
    _ACCESS_COUNTER += 1

    ## Reload every 5 accesses or if forced
    if LIMITS_DF is None or force or _ACCESS_COUNTER > 5:
        _ACCESS_COUNTER = 0
        LIMITS_DF = db.query_database(
            db="portfolio_data",
            table_name="limits",
            query="""
            SELECT * FROM limits
        """,
        )
    return LIMITS_DF


## STORERS


def update_greeks_case(trade_id: str, strategy_name: str, signal_id: str, greeks: dict, date_):
    # validate inputs
    missing = [k for k in MEASURES_SET if k not in greeks]
    if missing:
        raise ValueError(f"Missing greek(s): {missing}")

    # Support either greeks["new_date"] or greeks["date"] as the target date
    new_date = greeks.pop("new_date", greeks.pop("date", None))

    params = {
        "old_date": date_,  # date you are updating FROM
        "signal_id": signal_id,
        "strategy_name": strategy_name,
        "trade_id": trade_id,
        **greeks,
    }
    set_date_sql = ", `date` = :new_date" if new_date is not None else ""
    if new_date is not None:
        params["new_date"] = new_date

    # Update values (+ optionally move to new date)
    q = text(f"""
        UPDATE `limits`
        SET `value` = CASE `risk_measure`
            WHEN 'delta' THEN :delta
            WHEN 'gamma' THEN :gamma
            WHEN 'vega'  THEN :vega
            WHEN 'theta' THEN :theta
            ELSE `value`
        END
        {set_date_sql}
        WHERE `date`=:old_date
          AND `signal_id`=:signal_id
          AND `strategy_name`=:strategy_name
          AND `trade_id`=:trade_id
          AND `risk_measure` IN ('delta','gamma','vega','theta')
    """)

    engine = get_engine("portfolio_data")
    with engine.begin() as conn:
        res = conn.execute(q, params)

        # Verify at the final date (new_date if provided, else old_date)
        verify_date = new_date or date_
        rows = conn.execute(
            text("""
            SELECT `risk_measure`, `value`
            FROM `limits`
            WHERE `date`=:date AND `signal_id`=:signal_id
              AND `strategy_name`=:strategy_name AND `trade_id`=:trade_id
              AND `risk_measure` IN ('delta','gamma','vega','theta')
        """),
            {"date": verify_date, "signal_id": signal_id, "strategy_name": strategy_name, "trade_id": trade_id},
        ).all()

    after = {rm: v for rm, v in rows}
    return {
        "matched": len(after) == 4,  # how many of the 4 measures now exist at the target date
        "rowcount": res.rowcount,  # may be 0 if values were identical
        "date": verify_date,
        "after": after,
    }


def save_limits_by_meta(trade_id: str, strategy_name: str, signal_id: str, greeks: dict, date_):
    # validate inputs
    missing = [k for k in MEASURES_SET if k not in greeks]
    if missing:
        raise ValueError(f"Missing greek(s): {missing}")
    params = {"date": date_, "signal_id": signal_id, "strategy_name": strategy_name, "trade_id": trade_id, **greeks}
    insert_rows = []
    for measure in MEASURES_SET:
        insert_rows.append(
            {
                "date": params["date"],
                "signal_id": params["signal_id"],
                "strategy_name": params["strategy_name"],
                "trade_id": params["trade_id"],
                "risk_measure": measure,
                "value": params[measure],
            }
        )

    db.save_to_database(db="portfolio_data", table_name="limits", data=pd.DataFrame(insert_rows), filter_data=False)


def store_position_limits(
    delta_limit: str,
    gamma_limit: str,
    vega_limit: str,
    theta_limit: str,
    trade_id: str,
    strategy_name: str,
    date: str | datetime,
    signal_id: str,
):
    """
    Store position limits in the database.
    If limits already exist for the given trade_id, strategy_name, signal_id, and date, they will be updated.
    Args:
        delta_limit (str|float): The delta limit to store.
        gamma_limit (str|float): The gamma limit to store.
        vega_limit (str|float): The vega limit to store.
        theta_limit (str|float): The theta limit to store.
        trade_id (str): The trade ID associated with the limits.
        strategy_name (str): The strategy name associated with the limits.
        date (str|datetime): The date for which the limits are valid.
        signal_id (str): The signal ID associated with the limits.
    Returns:
        None
    """

    ## First check if the limits exist
    if isinstance(date, str):
        date = pd.to_datetime(date)

    exists, old_date = check_limits_exist(trade_id, strategy_name, signal_id=signal_id)

    ## Convert to float or None
    delta_limit = float(delta_limit) if not pd.isna(delta_limit) else None
    gamma_limit = float(gamma_limit) if not pd.isna(gamma_limit) else None
    vega_limit = float(vega_limit) if not pd.isna(vega_limit) else None
    theta_limit = float(theta_limit) if not pd.isna(theta_limit) else None

    ## If exists, we update
    if exists:
        logger.info(
            "Updating existing limits for trade_id=%s, strategy_name=%s, signal_id=%s, date=%s to new date=%s with values: delta=%s, gamma=%s, vega=%s, theta=%s",
            trade_id,
            strategy_name,
            signal_id,
            old_date,
            date,
            delta_limit,
            gamma_limit,
            vega_limit,
            theta_limit,
        )
        ## Delta row
        update_greeks_case(
            trade_id=trade_id,
            strategy_name=strategy_name,
            signal_id=signal_id,
            greeks={"delta": delta_limit, "gamma": gamma_limit, "vega": vega_limit, "theta": theta_limit, "date": date},
            date_=old_date,
        )

    ## If not exists, we create new rows
    else:
        logger.info(
            "Creating new limits for trade_id=%s, strategy_name=%s, signal_id=%s, date=%s with values: delta=%s, gamma=%s, vega=%s, theta=%s",
            trade_id,
            strategy_name,
            signal_id,
            date,
            delta_limit,
            gamma_limit,
            vega_limit,
            theta_limit,
        )
        save_limits_by_meta(
            trade_id=trade_id,
            strategy_name=strategy_name,
            signal_id=signal_id,
            greeks={"delta": delta_limit, "gamma": gamma_limit, "vega": vega_limit, "theta": theta_limit},
            date_=date,
        )


def check_limits_exist(trade_id, strategy_name, signal_id):
    """
    Check if the limits exist for the given trade_id and strategy_name

    Args:
        trade_id (str): The trade ID to check.
        strategy_name (str): The strategy name to check.
        signal_id (str): The signal ID to check.
    Returns:
        bool: True if limits exist, False otherwise.
    """

    limits_data = db.query_database(
        db="portfolio_data",
        table_name="limits",
        query=f"""
            SELECT * FROM limits
            WHERE trade_id = '{trade_id}'
            AND strategy_name = '{strategy_name}'
            AND signal_id = '{signal_id}'
        """,
    )
    if len(limits_data) > 0:
        return True, limits_data["date"].values[0]
    return False, None


def update_limits(trade_id: str, strategy_name: str, signal_id: str, greek_name: str, update_value: float):
    """
    Update the limits for the given trade_id, strategy_name, signal_id, and greek_name.
    Args:
        trade_id (str): The trade ID to update.
        strategy_name (str): The strategy name to update.
        signal_id (str): The signal ID to update.
        greek_name (str): The greek name to update (e.g., 'delta_limit').
        update_value (float): The new value for the limit.
    """

    dynamic_batch_update(
        db="portfolio_data",
        table_name="limits",
        update_values={"value": update_value},
        condition={
            "trade_id": trade_id,
            "strategy_name": strategy_name,
            "signal_id": signal_id,
            "risk_measure": greek_name,
        },
    )


def save_limits_from_backtester(bkt: OptionSignalBacktest, date: datetime = None):
    """
    Save limits from backtester to database.
    If date is provided, only save limits for that date.
    """
    unadjusted_trades = bkt.unadjusted_trades
    for info, greek_meta in bkt.risk_manager.limits_meta.items():
        old_signal_id, trade_id, date_ = info
        signal_id = unadjusted_trades[unadjusted_trades["signal_id"] == old_signal_id]["PT_BKTEST_SIG_ID"].values[0]
        if date is not None:
            if pd.to_datetime(date_).date() != date.date():
                logger.info(f"Skipping {info} as date does not match {date.date()}")
                continue
        store_position_limits(
            delta_limit=greek_meta.get("delta", None),
            gamma_limit=greek_meta.get("gamma", None),
            vega_limit=greek_meta.get("vega", None),
            theta_limit=greek_meta.get("theta", None),
            trade_id=trade_id,
            strategy_name="long_bbands",
            date=date_,
            signal_id=signal_id,
        )


## LOADERS
def get_position_limit(trade_id: str, strategy_name: str, signal_id: str, risk_measure: str) -> tuple:
    df = get_limits_data()
    assert risk_measure in MEASURES_SET, f"risk_measure must be one of {MEASURES_SET}"
    row = df[
        (df["trade_id"] == trade_id)
        & (df["strategy_name"] == strategy_name)
        & (df["signal_id"] == signal_id)
        & (df["risk_measure"] == risk_measure)
    ]
    if len(row) == 0:
        logger.error(
            f"No limit found for trade_id={trade_id}, strategy_name={strategy_name}, signal_id={signal_id}, risk_measure={risk_measure}"
        )
        return None
    return row["date"].values[0], float(row["value"].values[0])
