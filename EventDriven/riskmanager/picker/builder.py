import numpy as np
from typing import Optional
from EventDriven.riskmanager.picker import filter_contracts
from EventDriven.configs.core import ScoringConfigs
import pandas as pd
from trade.helpers.Logging import setup_logger
from trade.helpers.helper import parse_option_tick, to_datetime
from EventDriven.riskmanager.utils import populate_cache_with_chain
from EventDriven.dataclasses.orders import OrderRequest
from .vertical_spread import _extract_order_for_vertical_spread, vertical_spread_pairer_by_exp
from .naked_option import _extract_order_for_naked_option, naked_option_by_exp
from ...types import ResultsEnum, OrderData
from .iv_helper import _add_greeks_and_iv_to_chain
from .chain_scoring import _score_chain


logger = setup_logger("EventDriven.riskmanager.picker.builder", stream_log_level="WARNING")


def validate_order(order: dict, date: pd.Timestamp, spot: Optional[float] = None) -> bool:
    """Validate constructed order payload and derive basic risk metrics.

    Args:
        order: The order dictionary returned by order extraction helpers.
        date: Trade date used for DTE computation.
        spot: Optional underlying spot used for moneyness computation.

    Returns:
        True when validation passes.
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
        date_dt = to_datetime(date)
        for _, meta in trade_id.meta.items():
            for m in meta:
                dt.append((to_datetime(m["exp_date"]) - date_dt).days)
                if spot is not None:
                    moneyness.append(m["strike"] / spot if m["put_call"].lower() == "p" else spot / m["strike"])

        order.setdefault("metrics", {})
        order["metrics"]["min_dte"] = min(dt) if dt else None
        order["metrics"]["max_dte"] = max(dt) if dt else None
        order["metrics"]["min_moneyness"] = min(moneyness) if moneyness else None
        order["metrics"]["max_moneyness"] = max(moneyness) if moneyness else None
    return True


def order_builder(
    req: OrderRequest,
    configs: ScoringConfigs,
) -> OrderData:
    """Build an order from an OrderRequest using scoring-based selection.

    Args:
        req: The order request containing symbol, date, option type, and pricing constraints.
        configs: Scoring configuration defining moneyness, DTE, spread, and pricing targets.

    Returns:
        OrderData: Detailed trade execution data including positions and pricing.
    """
    order = _order_builder_with_scoring(req=req, configs=configs)
    validate_order(order, date=req.date, spot=req.chain_spot)
    return order


def _order_builder_with_scoring(
    req: OrderRequest,
    configs: ScoringConfigs,
) -> OrderData:
    """
    Build an order using the scored chain, which includes additional scoring metrics for better decision-making.
    This function is currently not used in the main builder flow but can be integrated for enhanced order construction based on scoring.
    """

    ## Step 1: Build scored chain
    scored_chain = build_scored_chain(req=req, configs=configs)
    if scored_chain.empty:
        logger.warning(
            f"Following are used for filtering but resulted in empty chain: min_moneyness={configs.min_moneyness}, max_moneyness={configs.max_moneyness}, target_dte={configs.target_dte}, dte_tolerance={configs.dte_tolerance}, spread_ratio_max={configs.pct_spread_max}, mid_price_range=({configs.mid_lower_limit}, {configs.mid_upper_limit})"
        )
        order = _extract_order_with_scoring_config(chain_row=pd.Series(), configs=configs)
        order["result"] = ResultsEnum.NO_CONTRACTS_FOUND.value
        return order

    ## Step 2: Extract top scored row and build order from it
    top_scored_row = scored_chain.iloc[0]

    ## Note: The order extraction functions will need to be updated to accept the additional scoring metrics if we want to utilize them in the order construction logic. For now, we will just pass the top scored row as is, and the extraction functions can decide how to use the scoring metrics if needed.
    order = _extract_order_with_scoring_config(chain_row=top_scored_row, configs=configs)
    validate_order(order, date=req.date, spot=req.chain_spot)
    return order


def build_scored_chain(req: OrderRequest, configs: ScoringConfigs) -> pd.DataFrame:
    """
    Build a scored option chain based on the request and scoring configurations. This involves fetching the chain, filtering it, adding Greeks and IV, pairing contracts if necessary, and then applying the scoring functions to rank the contracts.
        The resulting DataFrame will include the original contract data along with the calculated scores for each contract, allowing for informed decision-making in order construction.
    Steps:
    1. Fetch the option chain data for the given ticker and date.
    2. Filter the chain based on the request parameters (option type, moneyness, DTE, etc.).
    3. Add Greeks and implied volatility to the filtered chain.
    4. If the strategy involves spreads, pair the contracts accordingly.
    5. Apply the scoring functions to the resulting chain to calculate scores for moneyness, DTE, mid price, percentage spread, open interest, and theta burden.
    6. Return the scored chain sorted by total score in descending order.

    Additional scoring functions can be defined in the chain_scoring module, and the _score_chain function will apply these functions to the appropriate columns in the DataFrame to calculate the scores. The final scored chain will have additional columns for each score, as well as a total score that can be used for ranking the contracts.
    Args:
        req (OrderRequest): The order request containing parameters for fetching and filtering the option chain.
        configs (ScoringConfigs): The scoring configurations that define how the scores are calculated and weighted
    Returns:
        pd.DataFrame: A DataFrame containing the scored option chain, with additional columns for each
    """
    ## Step 1: Get chain
    chain = populate_cache_with_chain(tick=req.symbol, date=req.date, chain_spot=req.chain_spot)

    ## Step 2: Filter chain
    filtered = filter_contracts(
        df=chain,
        option_type=req.option_type,
        spot=req.chain_spot,
        min_moneyness=configs.min_moneyness,
        max_moneyness=configs.max_moneyness,
        target_dte=configs.target_dte,
        dte_tol=configs.dte_tolerance,
    )
    if filtered.empty:
        logger.warning(
            f"Filtering resulted in empty chain for {req.symbol} on {req.date}. Parameters used for filtering: option_type={req.option_type}, min_moneyness={configs.min_moneyness}, max_moneyness={configs.max_moneyness}, target_dte={configs.target_dte}, dte_tolerance={configs.dte_tolerance}"
        )
        return filtered

    ## Step 3: Add Greeks and IV
    filtered = _add_greeks_and_iv_to_chain(filtered=filtered, date=req.date, chain_spot=req.chain_spot)

    ## Step 4: Pair vertical spreads or naked options
    is_call = req.option_type.lower() == "c"
    filtered = filtered.sort_values(by="strike", ascending=is_call).reset_index(drop=True)
    vertical_chain = (
        filtered.groupby("expiration")
        .apply(
            naked_option_by_exp if configs.strategy == "naked" else vertical_spread_pairer_by_exp,
            spread_tick=configs.spread_ticks,
            min_total_price=configs.mid_lower_limit,
            max_total_price=configs.mid_upper_limit,
            max_pct_width=configs.pct_spread_max,
            min_oi=-25,
            delta_lmt=np.inf,
        )
        .reset_index(drop=True)
    )

    ## Step 5: Score pairs
    scored_chain = _score_chain(vertical_chain, configs=configs)
    scored_chain = scored_chain[
        [
            "long_leg_opttick",
            "short_leg_opttick",
            "spread_moneyness",
            "moneyness_score",
            "dte_score",
            "spread_dte",
            "mid_score",
            "spread_mid",
            "spread_pct_ratio",
            "pct_spread_score",
            "oi_score",
            "spread_oi",
            "theta_burden_score",
            "spread_delta",
            "total_score",
        ]
    ].sort_values("total_score", ascending=False)
    return scored_chain


def _extract_order_with_scoring_config(chain_row: pd.Series, configs: ScoringConfigs) -> OrderData:
    """
    Extract order schema from a scored chain row, applying any necessary adjustments based on scoring configs.
    """

    schema = {"structure_direction": configs.structure_direction}
    return (
        _extract_order_for_naked_option(chain_row, schema=schema)
        if configs.strategy == "naked"
        else _extract_order_for_vertical_spread(chain_row, schema=schema)
    )
