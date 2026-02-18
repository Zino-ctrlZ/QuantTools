
import numpy as np
import pandas as pd
from EventDriven.riskmanager.picker import _order_formatting, create_trade_id
from EventDriven.types import ResultsEnum, OrderDict
from EventDriven.riskmanager.picker import OrderSchema


def vertical_spread_pairer_by_exp(
    row: pd.Series,
    spread_tick: int = 1,
    min_total_price: float = 0.5,
    max_total_price: float = 1.0,
) -> pd.DataFrame:
    """
    For a given row (option contract), find the corresponding leg of the vertical spread based on the spread_tick.
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
    short_leg_details = row[tgt_details].shift(-spread_tick).reset_index(drop=True)

    ## Drop last spread_tick rows which will have NaNs in the short leg details after the shift.
    valid_length = len(row) - spread_tick
    long_leg_details = long_leg_details.iloc[:valid_length]
    short_leg_details = short_leg_details.iloc[:valid_length]

    ## Produce relevant spread information.
    spread_bid = long_leg_details["closebid"] - short_leg_details["closeask"]
    spread_ask = long_leg_details["closeask"] - short_leg_details["closebid"]
    spread_mid = long_leg_details["midpoint"] - short_leg_details["midpoint"]
    bid_ask_spread = spread_ask - spread_bid
    spread_pct_ratio = abs(spread_bid - spread_ask) / spread_mid.replace(0, np.nan)  # Avoid division by zero.
    spread_oi = abs(long_leg_details["open_interest"] + short_leg_details["open_interest"])

    ## Combine into a DataFrame for analysis.
    paired_opttick = pd.concat(
        (
            long_leg_details["opttick"],
            short_leg_details["opttick"],
            spread_mid,
            spread_bid,
            spread_ask,
            bid_ask_spread,
            spread_pct_ratio,
            spread_oi,
        ),
        axis=1,
    )
    paired_opttick.columns = [
        "long_leg_opttick",
        "short_leg_opttick",
        "spread_mid",
        "spread_bid",
        "spread_ask",
        "bid_ask_spread",
        "spread_pct_ratio",
        "spread_oi",
    ]
    ## Ensure bid, ask > 0
    ## Ensure spread_pct <= 1.0 (we want tight spreads relative to the mid price)
    paired_opttick = paired_opttick[
        (paired_opttick["spread_bid"] > 0)
        & (paired_opttick["spread_ask"] > 0)
        & (paired_opttick["spread_pct_ratio"] <= 1.0)
    ].reset_index(drop=True)
    return paired_opttick[paired_opttick["spread_mid"].between(min_total_price, max_total_price)].reset_index(drop=True)


def _vertical_spread_pairer(
    filtered_chain: pd.DataFrame,
    schema: OrderSchema,
) -> pd.DataFrame:
    """
    For a given filtered option chain, find the best option contract based on the spread_tick.
    Args:
        filtered_chain (pd.DataFrame): The filtered option chain DataFrame.
        schema (OrderSchema): The order schema containing parameters for building the vertical spread.
    Returns:
        pd.DataFrame: A Series containing the picked vertical spread contract details.
    """
    # Start by ordering by strike, from ITM to OTM.
    #   For calls, ITM is lower strike, for puts, ITM is higher strike.

    spread_tick = schema["spread_ticks"]
    is_call = schema["option_type"].lower() == "c"
    sorted_chain = filtered_chain.sort_values(
        by="strike",
        ascending=is_call,  # Calls: Ascending (lower strike = ITM). Puts: Descending (higher strike = ITM).
    ).reset_index(drop=True)

    # spread_ticks is the number of ticks between the legs of the spread.
    # For a call spread with spread_ticks=1, we buy the ITM call and sell the next lower strike call.
    # For a put spread with spread_ticks=1, we buy the ITM put and sell the next higher strike put.
    # For vertical spreads it is important that the legs are paired to expiration.
    vertical_chain = (
        sorted_chain.groupby("expiration")
        .apply(
            vertical_spread_pairer_by_exp,
            spread_tick=spread_tick,
            min_total_price=schema["min_total_price"],
            max_total_price=schema["max_total_price"],
        )
        .reset_index(level=1, drop=True)
        .sort_index()
    )

    ## Now we have our vertical spread chain with paired optticks and spread metrics for analysis.
    ## We pick the spread we want based on specific criteria. We sort based on (this is by priority):
    ## 1. spread_pct_ratio (we want this to be low, meaning the spread is relatively tight compared to its midpoint)
    ## 2. spread_oi (we want this to be high, meaning there's good liquidity in the spread)
    ## Finally, pick the top row as our chosen spread.
    vertical_chain.sort_values(by=["spread_pct_ratio", "spread_oi"], ascending=[True, False], inplace=True)
    picked_spread = vertical_chain.iloc[0] if not vertical_chain.empty else pd.Series()
    return picked_spread


def _extract_order_for_vertical_spread(picked_spread: pd.Series, schema: OrderSchema) -> OrderDict:
    """ 
     Extract order details for a vertical spread based on the picked spread and the provided schema.
     Args:
        picked_spread (pd.Series): A Series containing the details of the picked vertical spread contract.
        schema (OrderSchema): The order schema containing parameters for building the vertical spread.
     Returns:
        "OrderResult": A dictionary containing the result status and order data.
    """
    ## Extract order
    if not picked_spread.empty:
        long = [{"opttick": picked_spread["long_leg_opttick"]}]
        short = [{"opttick": picked_spread["short_leg_opttick"]}]
        leg_info = [("L", picked_spread["long_leg_opttick"]), ("S", picked_spread["short_leg_opttick"])]
        close_price = picked_spread["spread_mid"]

        ## Other details from spread
        pct_ratio = picked_spread["spread_pct_ratio"]
        spread_oi = picked_spread["spread_oi"]
        trade_id = create_trade_id(
            legs={
                "long": long,
                "short": short,
            }
        )
        data = _order_formatting(
            trade_id=trade_id, legs=leg_info, close=close_price, dir=schema["structure_direction"]
        )
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


def vertical_spread_order_builder(
    filtered_chain: pd.DataFrame,
    schema: dict,
) ->  OrderDict:
    """
    Build a vertical spread order based on the filtered option chain and the provided schema.
    Args:
        filtered_chain (pd.DataFrame): The filtered option chain DataFrame.
        schema (dict): The order schema containing parameters for building the vertical spread.
    Returns:
        dict: A dictionary containing the result status and order data.
    """
    picked_spread = _vertical_spread_pairer(
        filtered_chain=filtered_chain,
        schema=schema,
    )
    order = _extract_order_for_vertical_spread(picked_spread, schema=schema)
    return order
