from .vertical_spread import vertical_spread_order_builder
from .naked_option import naked_option_order_builder
from ...types import ResultsEnum, OrderData
from trade.helpers.helper import parse_option_tick
from EventDriven.riskmanager.picker import filter_contracts, OrderSchema
import pandas as pd

BUILDER_FACTORY = {
    "vertical": vertical_spread_order_builder,
    "naked": naked_option_order_builder,
}


def validate_order(order: dict) -> bool:
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
    return True

def order_builder(
    unfiltered_chain: pd.DataFrame,
    schema: OrderSchema,
    spot: float,
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

    # Step 2: Build order using the appropriate builder function
    structure_type = schema.get("strategy")
    if structure_type not in BUILDER_FACTORY:
        raise ValueError(f"Unsupported structure type: {structure_type}. Supported types are: {list(BUILDER_FACTORY.keys())}")

    builder_function = BUILDER_FACTORY[structure_type]
    order = builder_function(
        filtered_chain=filtered_chain,
        schema=schema,
    )

    # Step 3: Validate the constructed order
    try:
        validate_order(order)
    except AssertionError as e:
        raise ValueError(f"Order validation failed: {e}") from e

    return order