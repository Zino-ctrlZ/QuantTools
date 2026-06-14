import numpy as np
import pandas as pd
from typing import Optional
from EventDriven.riskmanager.picker import _order_formatting, create_trade_id
from EventDriven.riskmanager.picker.utils import (
    _delta_lmt,
    _verify_delta_in_chain,
    _build_common_pair_masks,
    _finalize_paired_output,
)
from EventDriven.types import ResultsEnum, OrderDict
from EventDriven.riskmanager.picker import OrderSchema
from trade.helpers.Logging import setup_logger

logger = setup_logger("EventDriven.riskmanager.picker.vertical_spread")


def vertical_spread_pairer_by_exp(
    row: pd.DataFrame,
    spread_tick: int = 1,
    min_total_price: float = 0.5,
    max_total_price: float = 1.0,
    max_pct_width: float = np.inf,
    min_oi: int = 0,
    delta_lmt: Optional[float] = None,
    return_mask: bool = False,
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
        delta_lmt (Optional[float]): Optional delta limit for the spread.
    Returns:
        pd.DataFrame: A DataFrame containing the paired legs of the vertical spread and their calculated metrics, filtered by the total spread mid price.
    """
    logger.debug(f"vertical_spread_pairer_by_exp recieved delta_lmt: {delta_lmt}")  ## DEBUG
    tgt_details = ["opttick", "midpoint", "closebid", "closeask", "open_interest", "delta", "moneyness", "dte", "theta"]

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
    spread_moneyness = long_leg_details["moneyness"]
    dte = long_leg_details["dte"]
    spread_delta = (long_leg_details["delta"] - short_leg_details["delta"]).fillna(
        np.inf
    )  # If delta is missing, set to infinity to fail delta filter.
    bid_ask_spread = spread_ask - spread_bid
    spread_pct_ratio = abs(spread_bid - spread_ask) / spread_mid.replace(0, np.nan)  # Avoid division by zero.
    spread_oi = abs(long_leg_details["open_interest"] + short_leg_details["open_interest"])
    spread_theta = long_leg_details["theta"] - short_leg_details["theta"]

    if "volume" in long_leg_details.columns and "volume" in short_leg_details.columns:
        spread_volume = abs(long_leg_details["volume"] + short_leg_details["volume"])
    else:
        spread_volume = pd.Series([np.nan] * len(spread_mid))  # If volume data is not available, fill with NaN.

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
            spread_volume,
            spread_delta,
            spread_moneyness,
            dte,
            spread_theta,
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
        "spread_volume",
        "spread_delta",
        "spread_moneyness",
        "spread_dte",
        "spread_theta",
    ]

    shrunk_delta_lmt = round(_delta_lmt(delta_lmt) * 0.95, 2)

    mask_df = _build_common_pair_masks(
        spread_mid=paired_opttick["spread_mid"],
        spread_bid=paired_opttick["spread_bid"],
        spread_ask=paired_opttick["spread_ask"],
        spread_pct_ratio=paired_opttick["spread_pct_ratio"],
        spread_oi=paired_opttick["spread_oi"],
        spread_delta=paired_opttick["spread_delta"],
        min_total_price=min_total_price,
        max_total_price=max_total_price,
        max_pct_width=max_pct_width,
        min_oi=min_oi,
        delta_lmt=shrunk_delta_lmt,
    )
    full_mask = mask_df["full_mask"]
    logger.debug(f"Number of spreads after applying all filters: {full_mask.sum()}")  ## DEBUG
    logger.debug(
        f"mid_mask: {mask_df['mid_mask'].sum()} (between {min_total_price} and {max_total_price}), spread_oi_mask: {mask_df['spread_oi_mask'].sum()}, pct_width_mask: {mask_df['pct_width_mask'].sum()}, spread_bid_mask: {mask_df['spread_bid_mask'].sum()}, spread_ask_mask: {mask_df['spread_ask_mask'].sum()}, delta_mask: {mask_df['delta_mask'].sum()}"
    )  ## DEBUG
    return _finalize_paired_output(paired_opttick=paired_opttick, mask_df=mask_df, return_mask=return_mask)


def _vertical_spread_pairer(
    filtered_chain: pd.DataFrame,
    schema: OrderSchema,
    delta_lmt: Optional[float] = None,
) -> pd.Series:
    """
    For a given filtered option chain, find the best option contract based on the spread_tick.
    Args:
        filtered_chain (pd.DataFrame): The filtered option chain DataFrame.
        schema (OrderSchema): The order schema containing parameters for building the vertical spread.
        delta_lmt (Optional[float]): The delta limit for filtering options. Defaults to None.
    Returns:
        pd.Series: A Series containing the picked vertical spread contract details.
    """
    logger.debug(f"_vertical_spread_pairer recieved delta_lmt: {delta_lmt}")  ## DEBUG
    if filtered_chain.empty:
        return pd.Series()  # Return empty Series if no contracts are available after filtering.
    # Start by ordering by strike, from ITM to OTM.
    #   For calls, ITM is lower strike, for puts, ITM is higher strike.
    max_pct_width = schema.get("max_pct_width", 0.10)  ## NOTE: Add to schema
    min_oi = schema.get("min_oi", 25)  ## NOTE: Add to schema
    spread_tick = schema["spread_ticks"]
    is_call = schema["option_type"].lower() == "c"
    sorted_chain = filtered_chain.sort_values(
        by="strike",
        ascending=is_call,  # Calls: Ascending (lower strike = ITM). Puts: Descending (higher strike = ITM).
    ).reset_index(drop=True)
    if sorted_chain.empty:
        return pd.Series()  # Return empty Series if no contracts are available after filtering.

    # spread_ticks is the number of ticks between the legs of the spread.
    # For a call spread with spread_ticks=1, we buy the ITM call and sell the next lower strike call.
    # For a put spread with spread_ticks=1, we buy the ITM put and sell the next higher strike put.
    # For vertical spreads it is important that the legs are paired to expiration.
    vertical_chain = sorted_chain.groupby("expiration").apply(
        vertical_spread_pairer_by_exp,
        spread_tick=spread_tick,
        min_total_price=schema["min_total_price"],
        max_total_price=schema["max_total_price"],
        max_pct_width=max_pct_width,
        min_oi=min_oi,
        delta_lmt=delta_lmt,
    )
    vertical_chain = vertical_chain.reset_index(level=1, drop=True).sort_index()

    ## Now we have our vertical spread chain with paired optticks and spread metrics for analysis.
    ## We pick the spread we want based on specific criteria. We sort based on (this is by priority):
    ## 1. spread_pct_ratio (we want this to be low, meaning the spread is relatively tight compared to its midpoint)
    ## 2. spread_oi (we want this to be high, meaning there's good liquidity in the spread)
    ## Finally, pick the top row as our chosen spread.
    order_delta_ascending = True if is_call else False
    vertical_chain.sort_values(
        by=["spread_pct_ratio", "spread_oi", "spread_delta"],
        ascending=[True, False, order_delta_ascending],
        inplace=True,
    )
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
        delta = picked_spread["spread_delta"]
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
                "delta": delta,
            },
            "scores": {
                "moneyness_score": picked_spread.get("moneyness_score", np.nan),
                "dte_score": picked_spread.get("dte_score", np.nan),
                "mid_score": picked_spread.get("mid_score", np.nan),
                "pct_spread_score": picked_spread.get("pct_spread_score", np.nan),
                "oi_score": picked_spread.get("oi_score", np.nan),
                "theta_burden_score": picked_spread.get("theta_burden_score", np.nan),
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
    delta_lmt: Optional[float] = None,
) -> OrderDict:
    """
    Build a vertical spread order based on the filtered option chain and the provided schema.
    Args:
        filtered_chain (pd.DataFrame): The filtered option chain DataFrame.
        schema (dict): The order schema containing parameters for building the vertical spread.
    Returns:
        dict: A dictionary containing the result status and order data.
    """
    logger.debug(f"vertical_spread_order_builder recieved delta_lmt: {delta_lmt}")  ## DEBUG
    filtered_chain = _verify_delta_in_chain(
        filtered_chain
    )  # Ensure delta column exists before proceeding to finder function.
    picked_spread = _vertical_spread_pairer(
        filtered_chain=filtered_chain,
        schema=schema,
        delta_lmt=delta_lmt,
    )
    order = _extract_order_for_vertical_spread(picked_spread, schema=schema)
    return order
