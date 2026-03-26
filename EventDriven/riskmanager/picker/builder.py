
from typing import Optional
from trade.helpers.helper import parse_option_tick
from EventDriven.riskmanager.picker import filter_contracts, OrderSchema
import pandas as pd
from trade.helpers.Logging import setup_logger
import numpy as np
from .vertical_spread import vertical_spread_order_builder
from .naked_option import naked_option_order_builder
from ...types import ResultsEnum, OrderData
from .iv_helper import _add_greeks_and_iv_to_chain

logger = setup_logger("EventDriven.riskmanager.picker.builder")

BUILDER_FACTORY = {
    "vertical": vertical_spread_order_builder,
    "naked": naked_option_order_builder,
}


def validate_order(order: dict, date: pd.Timestamp, spot: Optional[float] = None) -> bool:
    """
    Validates the order dictionary structure and contents.
    Raises ValueError if validation fails.
    """
    assert "result" in order, "Order must have a 'result' key."
    assert order["result"] in [e.value for e in ResultsEnum], f"Invalid result value: {order['result']}."
    if order["result"] == ResultsEnum.SUCCESSFUL.value:
        assert "data" in order, "Successful order must have 'data' key."
        data = order["data"]
        assert "trade_id" in data, "Order data must have 'trade_id'."
        assert "close" in data, "Order data must have 'close' price."
        assert "long" in data or "short" in data, "Order data must have at least 'long' or 'short' legs."
        if "long" in data:
            for leg in data["long"]:
                assert isinstance(leg, str), "Each long leg must be a string."
                try:
                    parse_option_tick(leg)
                except Exception as e:
                    raise ValueError(f"Invalid long leg opttick: {leg}. Error: {e}") from e
        if "short" in data:
            for leg in data["short"]:
                assert isinstance(leg, str), "Each short leg must be a string."
                try:
                    parse_option_tick(leg)
                except Exception as e:
                    raise ValueError(f"Invalid short leg opttick: {leg}. Error: {e}") from e

    ## Add min_dte to order
    data = order.get("data", {})

    ## No data to validate, so we can skip DTE calculation. 
    if not data:
        return True
    
    ## If there is data, we want to calculate DTE for risk management purposes.
    trade_id = order["data"].get("trade_id", None)
    dt = []
    moneyness = []
    if trade_id is not None and hasattr(trade_id, "meta"):
        for _, meta in trade_id.meta.items():
            for m in meta:
                dt.append((pd.to_datetime(m["exp_date"]) - pd.to_datetime(date)).days)
                if spot is not None:
                    moneyness.append(m["strike"] / spot if m["put_call"].lower() == "p" else spot / m["strike"])

        order["metrics"]["min_dte"] = min(dt) if dt else None
        order["metrics"]["max_dte"] = max(dt) if dt else None
        order["metrics"]["min_moneyness"] = min(moneyness) if moneyness else None
        order["metrics"]["max_moneyness"] = max(moneyness) if moneyness else None
    return True


def order_builder(
    unfiltered_chain: pd.DataFrame,
    schema: OrderSchema,
    spot: float,
    date: pd.Timestamp,
    delta_lmt: Optional[float] = None,
) -> OrderData:
    """
    Build an order based on the unfiltered option chain and the provided schema.
    Args:
        unfiltered_chain (pd.DataFrame): The unfiltered option chain DataFrame.
        schema (OrderSchema): The order schema containing parameters for building the order.
    Returns:
        OrderData: Detailed trade execution data including positions and pricing.
    """
    # Step 1: Filter contracts based on schema
    filtered_chain = filter_contracts(
        df=unfiltered_chain,
        schema=schema,
        spot=spot,
    )
    logger.info(f"Recieved {len(unfiltered_chain)} contracts from data source. Delta limit for this order: {delta_lmt}")

    ## If delta filter is enabled, calculate Greeks and IV for the filtered chain to apply delta-based filtering in the builder functions.
    ## If the chain is empty after initial filtering, we can skip this step to save computation.
    ## If delta_lmt is None, it means the position manager did not provide a delta limit, so we should skip delta-based filtering in the builder functions as well.
    if schema.get("enable_delta_filter", False) and not filtered_chain.empty and delta_lmt is not None:
        logger.info(f"Calculating Greeks and IV for {len(filtered_chain)} contracts to apply delta filter...")
        filtered_chain = _add_greeks_and_iv_to_chain(filtered_chain, date, spot)
    else:
        logger.info("Delta filter not enabled, skipping Greeks and IV calculation.")
        filtered_chain[["iv", "delta", "gamma", "vega", "theta", "rho", "volga"]] = np.inf  # Ensure these columns exist for builder functions, even if we are not calculating them.

    
    logger.info(f"Filtered chain size: {len(filtered_chain)} contracts after applying schema filters.")

    # Step 2: Build order using the appropriate builder function
    structure_type = schema.get("strategy")
    if structure_type not in BUILDER_FACTORY:
        raise ValueError(
            f"Unsupported structure type: {structure_type}. Supported types are: {list(BUILDER_FACTORY.keys())}"
        )

    builder_function = BUILDER_FACTORY[structure_type]
    order = builder_function(
        filtered_chain=filtered_chain,
        schema=schema,
        delta_lmt=delta_lmt,
    )

    # Step 3: Validate the constructed order
    try:
        validate_order(order, date, spot)
    except AssertionError as e:
        raise ValueError(f"Order validation failed: {e}") from e

    return order
