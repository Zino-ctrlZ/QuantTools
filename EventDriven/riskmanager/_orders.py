from typing import Dict, Any, TYPE_CHECKING, Tuple
from datetime import datetime
import pandas as pd
from trade.helpers.Logging import setup_logger
from EventDriven.riskmanager.picker import OrderSchema, ResultsEnum
from EventDriven.riskmanager._order_validator import OrderInputs
from EventDriven.riskmanager.utils import get_persistent_cache, set_use_temp_cache
from EventDriven.dataclasses.orders import OrderRequest

if TYPE_CHECKING:
    from EventDriven.riskmanager.picker.order_picker import OrderPicker

logger = setup_logger('EventDriven.riskmanager._orders', stream_log_level="WARNING")

def order_failed(order: Dict[str, Any]) -> bool:
    """
    Check if the order result indicates a failure.

    Args:
        order (Dict[str, Any]): The order dictionary containing the result.
    Returns:
        bool: True if the order result indicates failure, False otherwise.
    """
    return order['result'] != ResultsEnum.SUCCESSFUL.value

def resolve_schema(schema: OrderSchema, 
                   tries: int, 
                   max_dte_tolerance: int, 
                   otm_moneyness_width: float, 
                   itm_moneyness_width: float,
                   max_close: float, max_tries: int =6) -> Tuple[OrderSchema, int]:
    """
    Resolving schema by order of importance
    1. DTE Tolerance
    2. Min Moneyness width
    3. Max Moneyness width
    4. Max Close Price
    5. Max Schema Tries
    If no schema is found after max tries, return False and the number of tries.

    Args:
        schema (OrderSchema): The schema to resolve.
        tries (int): The number of tries already made.
        max_dte_tolerance (int): The maximum DTE tolerance to allow.
        moneyness_width (float): The moneyness width to allow.
        max_close (float): The maximum close price to allow.
        max_tries (int): The maximum number of tries allowed.

    Returns:
        tuple: A tuple containing the resolved schema or False if no schema was found, and the number of tries made.
    """
    tick = schema['tick']

    ##0). Max schema tries
    if tries >= max_tries:
        return False, tries

    #1). DTE Resolve
    tries +=1
    if schema['dte_tolerance'] <= max_dte_tolerance:
        logger.info(f"Resolving Schema ({tick}): {schema['dte_tolerance']} <= {max_dte_tolerance}, increasing DTE Tolerance by 10 from {schema['dte_tolerance']} to {schema['dte_tolerance'] + 20}")
        schema['dte_tolerance'] += 20
        return schema, tries
    
    #2). Min Moneyness Resolve
    elif 1 - schema['min_moneyness'] <= otm_moneyness_width:
        logger.info(f"Resolving Schema ({tick}): {1 - schema['min_moneyness']} <= {otm_moneyness_width}, decreasing Min Moneyness by 0.1 from {schema['min_moneyness']} to {schema['min_moneyness'] - 0.1}")
        schema['min_moneyness'] -= 0.1
        return schema, tries    

    #3). Max Moneyness Resolve
    elif schema['max_moneyness'] - 1 <= itm_moneyness_width:
        logger.info(f"Resolving Schema ({tick}): {schema['max_moneyness'] - 1} <= {itm_moneyness_width}, increasing Max Moneyness by 0.1 from {schema['max_moneyness']} to {schema['max_moneyness'] + 0.1}")
        schema['max_moneyness'] += 0.1
        return schema, tries
    
    #4). Close Resolve
    elif schema['max_total_price'] <= max_close:
        logger.info(f"Resolving Schema ({tick}): {schema['max_total_price']} <= {max_close}, increasing Max Close by 0.5 from {schema['max_total_price']} to {schema['max_total_price'] + 1}")
        schema['max_total_price'] += 1
        return schema, tries
    
    return False, tries


def order_resolve_loop(
        order: Dict[str, Any],
        schema: OrderSchema,
        date: pd.Timestamp|str|datetime,
        spot: float,
        max_close: float,
        max_dte_tolerance: int,
        max_tries: int,
        otm_moneyness_width: float,
        itm_moneyness_width: float,
        logger,
        signalID: str,
        schema_cache: dict,
        picker: "OrderPicker",
        request: OrderRequest = None,
):
    """
    Attempt to resolve an order schema until a successful order is produced or maximum tries are exceeded.
    Args:
    
        order (Dict[str, Any]): The initial order dictionary.
        schema (OrderSchema): The initial order schema.
        date (pd.Timestamp|str|datetime): The date for the order.
        spot (float): The current chain spot price.
        max_close (float): The maximum total price for the order.
        max_dte_tolerance (int): The maximum DTE tolerance for the order. (This should actually be minimum. It is the least we can tolerate)
        max_tries (int): The maximum number of tries to resolve the schema before giving up.
        otm_moneyness_width (float): The max width btwn OTM strikes and 1 to tolerate.
        itm_moneyness_width (float): The max width btwn ITM strikes and 1 to tolerate.
        logger: Logger object for logging information.
        signalID (str): Unique identifier for the signal/order.
        schema_cache (dict): Cache to store resolved schemas for specific dates and signal IDs. It will store in place.
        picker (OrderPicker): The OrderPicker instance to use for getting the order.

    Returns:
        Dict[str, Any]: The final order dictionary, either successful or indicating failure.
        """
    if picker.__class__.__name__ != "OrderPicker":
        raise ValueError("picker must be an instance of OrderPicker")
    
    tries = 0
    use_request = False


    if request is not None:
        ## This is a patch to use new get order method with request
        ## Technically it is still relevant to the previous method, but we use request to get schema
        ## And transform to tuple for old method
        logger.info(f"Attempting to resolve order for request: {request}")
        schema = picker.get_order_schema(
            ticker=request.symbol, 
            option_type=request.option_type, 
            max_total_price=request.max_close
        )
        schema_as_tuple = tuple(schema.data.items())
        use_request = True

    while order_failed(order):
        logger.info(f"Failed to produce order with schema: {schema}, trying to resolve schema, on try {tries}")
        pack = resolve_schema(schema,
                                        tries = tries,
                                        max_dte_tolerance = max_dte_tolerance,
                                        max_close = max_close,
                                        max_tries = max_tries,
                                        otm_moneyness_width = otm_moneyness_width,
                                        itm_moneyness_width = itm_moneyness_width)
        schema, tries = pack

        if schema is False:
            logger.info(f"Unable to resolve schema after {tries} tries, returning None")
            schema_cache.setdefault(date, {}).update({signalID: schema})
            return {
                'result': ResultsEnum.NO_CONTRACTS_FOUND.value,
                'data': None
            }
        logger.info(f"Resolved Schema: {schema}, tries: {tries}")

        if use_request:
            ## When using request, we have to use the _get_order method directly
            ## Previously, .get_order_new took in schema as a dict, converted to tuple internally
            ## And then called _get_order. But here we already have the tuple form
            ## So we call _get_order directly
            order = picker._get_order(
                schema=schema_as_tuple, 
                date=request.date, 
                spot=request.spot, 
                chain_spot=request.chain_spot, 
                print_url=False
            )
        else:
            order = picker.get_order_new(schema, date, spot, print_url = False) ## Get the order from the OrderPicker
    schema_cache.setdefault(date, {}).update({signalID: schema})
    return order



def get_open_order(
        picker: "OrderPicker",
        spot: float,
        date: datetime|str,
        schema: OrderSchema,
        inputs: OrderInputs,
        schema_cache: dict = None,
        test: bool = False
        
) -> dict:
    
    ## Initialize schema cache
    if schema_cache is None:
        schema_cache = {}

    ## Set caching parameters to temp
    set_use_temp_cache(True)

    ## Clear persistent cache if not in test mode
    if not test:
        print("INFO: Not in test mode, clearing persistent cache")
        get_persistent_cache().clear()
    else:
        print("WARNING: In test mode, using persistent cache")

    ## Utilize picker to get order
    order=picker.get_order_new(
        schema=schema,
        date=date,
        spot=spot,
    )

    ## Resole order if failed
    order = order_resolve_loop(
        order=order,
        schema=schema,
        date=inputs.date,
        spot=spot,
        max_close=inputs.tick_cash/100, ## Use tick cash to determine max close. Normalize to 100 contracts
        max_dte_tolerance=inputs.max_dte_tolerance,
        max_tries=inputs.max_tries,
        otm_moneyness_width=inputs.otm_moneyness_width,
        itm_moneyness_width=inputs.itm_moneyness_width,
        logger=logger,
        signalID=inputs.signal_id,
        schema_cache=schema_cache,
        picker=picker
    )

    ## Add necessary tags for identification
    order['signal_id'] = inputs.signal_id
    order['map_signal_id'] = inputs.signal_id
    return order

