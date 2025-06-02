import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict
from EventDriven.types import ResultsEnum
from .utils import (logger, 
                    get_cache, 
                    LOOKBACKS, 
                    precompute_lookbacks, 
                    populate_cache_with_chain, 
                    time_logger, 
                    produce_order_candidates,
                    refresh_cache,
                    populate_cache,
                    logger)
from .utils import *
from datetime import datetime, timedelta
from trade.helpers.helper import CustomCache
order_cache = CustomCache(BASE, fname = "order")
# --------- OrderSchema ---------
@dataclass
class OrderSchema:
    data: Dict[str, Any]

    def __post_init__(self):
        required = ["strategy", 
                    "option_type", 
                    "target_dte", 
                    "dte_tolerance", 
                    "structure_direction", 
                    "max_total_price",
                    "min_total_price", 
                    "tick"]
        for key in required:
            if key not in self.data:
                raise ValueError(f"Missing required field: {key}")

        if self.data["strategy"] == "vertical" and not ("spread_pct" in self.data or "spread_ticks" in self.data):
            raise ValueError("Vertical strategies require either 'spread_pct' or 'spread_ticks'")
        
        optional = {"min_moneyness": 0.9, "max_moneyness": 1.1, "max_attempts": 3, "increment": 0.25}
        for key, default in optional.items():
            if key not in self.data:
                self.data[key] = default

        

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data
    
    def __setitem__(self, key, value):
        if key not in self.data:
            raise KeyError(f"Key '{key}' not found in OrderSchema.")
        self.data[key] = value

    def __repr__(self):
        return repr(self.data)
    
    def get(self, key, default=None):
        return self.data.get(key, default)

# --------- Utilities ---------
def resolve_ordering(option_type, structure_direction):
    if structure_direction == "long":
        return (True, ("long", "short")) if option_type.lower() == "c" else (False, ("long", "short"))
    else:
        return (False, ("short", "long")) if option_type.lower() == "p" else (True, ("short", "long"))

def filter_contracts(df: pd.DataFrame, schema: OrderSchema, spot: float, min_moneyness: float = 0.5, max_moneyness: float = 1.5, increment =0.25) -> pd.DataFrame:
    target_dte = schema["target_dte"]
    dte_tol = schema["dte_tolerance"]
    filtered = pd.DataFrame()
    attempt = 0
    factor = 1
    max_attempts = schema.get("max_attempts", 3)
    min_moneyness = schema.get("min_moneyness", min_moneyness)
    max_moneyness = schema.get("max_moneyness", max_moneyness)
    increment = schema.get("increment", increment)
    while filtered.empty and attempt < max_attempts:
        lower_strike = spot * (min_moneyness * factor)
        upper_strike = spot * (max_moneyness * factor)
        filtered = df[
            (df["dte"].between(target_dte - dte_tol, target_dte + dte_tol)) &
            (df["strike"].between(lower_strike, upper_strike))
        ].copy()
        attempt += 1
        factor *= (1 + increment)  # Increase the range by a factor of (1 + increment) each attempt
    if filtered.empty:
        logger.critical(f"Warning: No contracts found for {schema['option_type']} with DTE {target_dte} ± {dte_tol} and strike range [{lower_strike:.2f}, {upper_strike:.2f}] after {attempt} attempts.")
    return filtered.reset_index(drop=True)

def build_spread_by_ticks(df, schema, cache):
    df = df[df["right"].str.lower() == schema["option_type"].lower()].copy()
    df["mid"] = df["chain_id"].map(cache)
    df = df.dropna(subset=["mid"])
    ascending, _ = resolve_ordering(schema["option_type"], schema["structure_direction"])

    spreads = []
    for exp, group in df.groupby("expiration"):
        group = group.sort_values("strike", ascending=ascending).reset_index(drop=True)
        for i in range(len(group) - schema["spread_ticks"]):
            leg1, leg2 = group.iloc[i], group.iloc[i + schema["spread_ticks"]]
            long, short = (leg1, leg2)
            spread_price = long["mid"] - short["mid"]
            if abs(spread_price) <= schema["max_total_price"] and abs(spread_price) >= schema["min_total_price"]:
                spreads.append({
                    "long": long, "short": short,
                    "spread_price": spread_price,
                    "width": abs(short["strike"] - long["strike"]),
                    "dte": int(long["dte"]),
                    "expiration": long["expiration"],
                    "option_type": schema["option_type"],
                    "type": "vertical",
                    "legs": [long, short],
                })

    if schema["structure_direction"] == "long":
        pick = min((s for s in spreads if s["spread_price"] > 0), key=lambda s: s["spread_price"], default=None)
    else:
        pick = min((s for s in spreads if s["spread_price"] < 0), key=lambda s: s["spread_price"], default=None)
    return [pick] if pick else []

def build_spread_by_pct(df, schema, spot, cache):
    df = df[df["right"].str.lower() == schema["option_type"].lower()].copy()
    df["mid"] = df["chain_id"].map(cache)
    df = df.dropna(subset=["mid"])
    ascending, _ = resolve_ordering(schema["option_type"], schema["structure_direction"])
    spreads = []
    for exp, group in df.groupby("expiration"):
        group = group.sort_values("strike", ascending=ascending).reset_index(drop=True)
        for i in range(len(group)):
            leg1 = group.iloc[i]
            target_strike = leg1["strike"] + (spot * schema["spread_pct"] if ascending else -spot * schema["spread_pct"])
            group_slice = group.iloc[i+1:]  # only look ahead to maintain spread structure
            if group_slice.empty:
                continue

            leg2_idx = (group_slice["strike"] - target_strike).abs().idxmin()
            leg2 = group.loc[leg2_idx]
            error = (leg2["strike"] - target_strike) ** 2 
            
            ## Controlling distance apart. Avoiding spreads that are too wide or too narrow.
            actual_width = abs(leg2["strike"] - leg1["strike"])
            min_width = spot * schema["spread_pct"] * 0.10
            max_error = (spot * schema["spread_pct"] * 1.5) ** 2

            if actual_width < min_width or error > max_error:
                logger.info(f"Skipping spread due to width or error: {actual_width:.2f} < {min_width:.2f} or error {error:.2f} > {max_error:.2f}")
                continue

            long, short = (leg1, leg2)
            spread_price = long["mid"] - short["mid"]
            
            if abs(spread_price) <= schema["max_total_price"]:
                spreads.append({
                    "long": long, "short": short,
                    "spread_price": spread_price,
                    "width": abs(short["strike"] - long["strike"]),
                    "dte": int(long["dte"]),
                    "expiration": long["expiration"],
                    "option_type": schema["option_type"],
                    "type": "vertical",
                    "legs": [long, short],
                })
    if schema["structure_direction"] == "long":
        pick = min((s for s in spreads if s["spread_price"] > 0), key=lambda s: s["spread_price"], default=None)
    else:
        pick = min((s for s in spreads if s["spread_price"] < 0), key=lambda s: s["spread_price"], default=None)
    return [pick] if pick else []

def build_vertical_spread(df, schema, spot, cache):
    df = filter_contracts(df, schema, spot)
    return build_spread_by_ticks(df, schema, cache) if "spread_ticks" in schema else build_spread_by_pct(df, schema, spot, cache)

def build_naked_option(df, schema, spot, cache):
    df = filter_contracts(df, schema, spot)
    df = df[df["right"].str.lower() == schema["option_type"].lower()].copy()
    df["mid"] = df["chain_id"].map(cache)
    df = df.dropna(subset=["mid"])
    df = df[df["mid"] <= schema["max_total_price"]]
    df = df.sort_values("mid", ascending=(schema["structure_direction"] == "long"))
    pick = df.iloc[0] if not df.empty else None
    return [{schema["structure_direction"]: pick}] if pick is not None else []

def build_strategy(df, schema, spot, cache):
    strategy_map = {
        "vertical": build_vertical_spread,
        "naked": build_naked_option,
    }
    builder = strategy_map.get(schema["strategy"])
    return builder(df, schema, spot, cache) if builder else []

def extract_order(obj):
    order = {}
    id =''

    ## If no contracts found, return early
    if not obj:
        order['result'] = ResultsEnum.NO_CONTRACTS_FOUND.value
        order['data'] = None
        return order
    
    ## If contracts found, build the order
    order['result'] = ResultsEnum.SUCCESSFUL.value
    order['data'] = {"trade_id": "", 
                      "close": 0,
                      'long': [],
                      'short': [],}
    for pack in obj:
        for direction, data in pack.items():
            if direction not in ('long', 'short'):
                continue
            id+= f"&{direction[0].upper()}:{data['opttick']}"
            order['data'][direction].append(data["opttick"])
            mid = data["mid"]
            order['data']['close'] += mid if direction == 'long' else -mid
    order['data']['trade_id'] = id
    return order
