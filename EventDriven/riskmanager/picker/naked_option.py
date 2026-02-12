import numpy as np
import pandas as pd
from EventDriven.riskmanager.picker import _order_formatting, create_trade_id
from EventDriven.types import ResultsEnum, OrderDict
from EventDriven.riskmanager.picker import OrderSchema



def naked_option_by_exp(
    row: pd.Series,
    min_total_price: float = 0.5,
    max_total_price: float = 1.0,
) -> pd.DataFrame:
    """
    For a given row (option contract), find the corresponding leg of the naked option based on the spread_tick.
    Calculate spread metrics such as spread mid, spread bid, spread ask, bid-ask spread, spread percentage ratio, and combined open interest.
    Filter the resulting paired DataFrame based on the total spread mid price being within the specified min
    and max total price range.
    Args:
        row (pd.Series): A row from the sorted_chain DataFrame representing an option contract.
        spread_tick (int): The number of ticks between the legs of the spread.
        min_total_price (float): Minimum total price of the spread to filter the results.
        max_total_price (float): Maximum total price of the spread to filter the results.
    Returns:
        pd.DataFrame: A DataFrame containing the paired legs of the vertical spread and their calculated metrics, filtered by the total spread mid price.
    """
    tgt_details = ["opttick", "midpoint", "closebid", "closeask", "open_interest"]
    long_leg_details = row[tgt_details].reset_index(drop=True)

    ## Produce relevant spread information.
    spread_bid = long_leg_details["closebid"]
    spread_ask = long_leg_details["closeask"]
    spread_mid = long_leg_details["midpoint"]
    bid_ask_spread = spread_ask - spread_bid
    spread_pct_ratio = abs(spread_bid - spread_ask) / spread_mid.replace(0, np.nan)  # Avoid division by zero.
    spread_oi = abs(long_leg_details["open_interest"])

    ## Combine into a DataFrame for analysis.
    paired_opttick = pd.concat(
        (long_leg_details["opttick"], spread_mid, spread_bid, spread_ask, bid_ask_spread, spread_pct_ratio, spread_oi),
        axis=1,
    )
    paired_opttick.columns = [
        "long_leg_opttick",
        "spread_mid",
        "spread_bid",
        "spread_ask",
        "bid_ask_spread",
        "spread_pct_ratio",
        "spread_oi",
    ]
    return paired_opttick[paired_opttick["spread_mid"].between(min_total_price, max_total_price)].reset_index(drop=True)


def _naked_option_finder(
    filtered_chain: pd.DataFrame,
    schema: OrderSchema,
) -> pd.DataFrame:
    """
    For a given filtered option chain, find the best option contract based on the spread_tick.
    Args:
        filtered_chain (pd.DataFrame): The filtered option chain DataFrame.
        schema (OrderSchema): The order schema containing parameters for building the naked option.
    Returns:
        pd.DataFrame: A Series containing the picked naked option contract details.
    """

    # Start by ordering by strike, from ITM to OTM.
    #   For calls, ITM is lower strike, for puts, ITM is higher strike.

    is_call = schema["option_type"].lower() == "c"
    sorted_chain = filtered_chain.sort_values(
        by="strike",
        ascending=is_call,  # Calls: Ascending (lower strike = ITM). Puts: Descending (higher strike = ITM).
    ).reset_index(drop=True)

    naked_option_chain = (
        sorted_chain.groupby("expiration")
        .apply(
            naked_option_by_exp,
            min_total_price=schema["min_total_price"],
            max_total_price=schema["max_total_price"],
        )
        .reset_index(level=1, drop=True)
        .sort_index()
    )

    ## Now we have our naked option chain with spread metrics for analysis.
    ## We pick the option we want based on specific criteria. We sort based on (this is by priority):
    ## 1. spread_pct_ratio (we want this to be low, meaning the spread is relatively tight compared to its midpoint)
    ## 2. spread_oi (we want this to be high, meaning there's good liquidity in the spread)
    ## Finally, pick the top row as our chosen spread.
    naked_option_chain.sort_values(by=["spread_pct_ratio", "spread_oi"], ascending=[True, False], inplace=True)
    picked_spread = naked_option_chain.iloc[0] if not naked_option_chain.empty else pd.Series()

    return picked_spread


def _extract_order_for_naked_option(
    picked_spread: pd.Series,
    schema: OrderSchema,
) -> OrderDict:
    """
    Extract order details for a naked option based on the picked spread and the provided schema.
    Args:
        picked_spread (pd.Series): A Series containing the details of the picked naked option contract.
        schema (OrderSchema): The order schema containing parameters for building the naked option.
    Returns:
        OrderDict: A dictionary containing the result status and order data.
    """
    ## Extract order
    is_long = schema["structure_direction"].upper() == "LONG"
    if not picked_spread.empty:
        ## Determine leg info for order formatting
        long = [{"opttick": picked_spread["long_leg_opttick"]}] if is_long else []
        short = [{"opttick": picked_spread["short_leg_opttick"]}] if not is_long else []

        ## Determine leg info for order formatting
        leg_info = []
        if is_long:
            leg_info.append(("L", picked_spread["long_leg_opttick"]))
        else:
            leg_info.append(("S", picked_spread["long_leg_opttick"]))

        ## Other details from spread
        pct_ratio = picked_spread["spread_pct_ratio"]
        spread_oi = picked_spread["spread_oi"]

        close_price = picked_spread["spread_mid"]
        trade_id = create_trade_id(
            legs={
                "long": long,
                "short": short,
            }
        )
        data = _order_formatting(trade_id=trade_id, legs=leg_info, close=close_price, dir=schema["structure_direction"])
        order = {
            "result": ResultsEnum.SUCCESSFUL.value,
            "data": data,
            "metrics": {
                "spread_pct_ratio": pct_ratio,
                "spread_oi": spread_oi,
            },
        }
    else:
        order = {
            "result": ResultsEnum.UNSUCCESSFUL.value,
            "data": None,
        }

    return order


def naked_option_order_builder(
    filtered_chain: pd.DataFrame,
    schema: OrderSchema,
) -> OrderDict:
    """
    Build a vertical spread order based on the filtered option chain and the provided schema.
    Args:
        filtered_chain (pd.DataFrame): The filtered option chain DataFrame.
        schema (OrderSchema): The order schema containing parameters for building the vertical spread.
    Returns:
        OrderDict: A dictionary containing the result status and order data.
    """
    picked_spread = _naked_option_finder(
        filtered_chain=filtered_chain,
        schema=schema,
    )
    order = _extract_order_for_naked_option(picked_spread, schema=schema)
    return order