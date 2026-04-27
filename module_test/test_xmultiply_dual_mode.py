"""Run side-by-side evidence tests for xmultiply attribution payload paths.

This script runs two test formats for the same option/date inputs:
1) Direct mode: call load_option_pnl_data with opttick only.
2) Payload mode: preload all required data via trade.datamanager managers,
   build OptionPnlPayload, then pass payload into load_option_pnl_data.

Usage:
    /Users/chiemelienwanisobi/miniconda3/envs/openbb_new_use/bin/python \
        module_test/test_xmultiply_dual_mode.py \
        --opttick HD20250620C500 \
        --start 2024-12-03 \
        --end 2024-12-31
"""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime
from typing import Any, Dict

from pandas.tseries.offsets import BDay

from trade.helpers.helper import parse_option_tick, change_to_last_busday
from trade.assets.calculate.data_classes import OptionPnlPayload, SymbolPayload
from trade.assets.calculate.xmultiply_attr_v2 import load_option_pnl_data
from trade.datamanager import (
    SpotDataManager,
    RatesDataManager,
    VolDataManager,
    OptionSpotDataManager,
    GreekDataManager,
)


def _summarize_output(mode: str, payload: OptionPnlPayload) -> Dict[str, Any]:
    """Build a compact evidence payload from function output."""
    attribution = payload.attribution
    dod_change = payload.dod_change
    summary: Dict[str, Any] = {
        "mode": mode,
        "status": "SUCCESS",
        "attribution_rows": 0 if attribution is None else len(attribution),
        "dod_rows": 0 if dod_change is None else len(dod_change),
        "attribution_cols": [],
        "attribution_index_min": None,
        "attribution_index_max": None,
        "aggregates": {},
    }

    if attribution is not None and not attribution.empty:
        summary["attribution_cols"] = list(attribution.columns)
        summary["attribution_index_min"] = str(attribution.index.min())
        summary["attribution_index_max"] = str(attribution.index.max())

        aggregate_columns = [
            "opt_dod_change",
            "total_pnl_excl_trade_pnl",
            "trade_pnl_adjustment",
            "total_pnl",
            "unexplained_pnl",
        ]
        aggregates = {}
        for col in aggregate_columns:
            if col in attribution.columns:
                aggregates[col] = float(attribution[col].sum())
        summary["aggregates"] = aggregates

    return summary


def run_direct(opttick: str, start: datetime, end: datetime) -> Dict[str, Any]:
    """Run direct mode (no prebuilt payload)."""
    out = load_option_pnl_data(yesterday=start, today=end, opttick=opttick)
    return _summarize_output(mode="direct", payload=out)


def run_payload(opttick: str, start: datetime, end: datetime) -> Dict[str, Any]:
    """Run payload mode with all factors loaded from dedicated data managers."""
    meta = parse_option_tick(opttick)
    query_start = change_to_last_busday(start - BDay(1))

    symbol = meta["ticker"]
    exp = meta["exp_date"]
    strike = meta["strike"]
    right = meta["put_call"]

    spot_mgr = SpotDataManager(symbol)
    rates_mgr = RatesDataManager()
    vol_mgr = VolDataManager(symbol)
    opt_spot_mgr = OptionSpotDataManager(symbol)
    greeks_mgr = GreekDataManager(symbol)

    asset_spot = spot_mgr.get_spot_timeseries(
        start_date=query_start,
        end_date=end,
        undo_adjust=True,
    ).timeseries
    asset_spot.name = "spot"

    rates_spot = rates_mgr.get_risk_free_rate_timeseries(
        start_date=query_start,
        end_date=end,
    ).timeseries
    rates_spot.name = "rates"

    vol_data = vol_mgr.get_implied_volatility_timeseries(
        start_date=query_start,
        end_date=end,
        expiration=exp,
        strike=strike,
        right=right,
    ).timeseries
    vol_data.name = "vol"

    option_spot_result = opt_spot_mgr.get_option_spot_timeseries(
        start_date=query_start,
        end_date=end,
        expiration=exp,
        strike=strike,
        right=right,
    )
    option_spot = option_spot_result.price
    option_spot.name = "spot"

    greeks_data = greeks_mgr.get_greeks_timeseries(
        start_date=query_start,
        end_date=end,
        expiration=exp,
        strike=strike,
        right=right,
    ).timeseries

    preloaded_payload = OptionPnlPayload(
        opttick=opttick,
        date=end,
        vol=vol_data,
        spot=option_spot,
        greeks=greeks_data,
        asset_payload=SymbolPayload(symbol=symbol, datetime=end, spot=asset_spot),
        rates_payload=SymbolPayload(symbol="RATES_USD", datetime=end, spot=rates_spot),
    )

    out = load_option_pnl_data(
        yesterday=start,
        today=end,
        opttick=opttick,
        payload=preloaded_payload,
    )
    return _summarize_output(mode="payload_datamanagers", payload=out)


def _run_and_capture(label: str, fn, *args):
    """Run a test mode and capture success/error in a JSON-serializable form."""
    print(f"\n===== {label} =====")
    try:
        result = fn(*args)
        print(json.dumps(result, indent=2))
        return result
    except Exception as exc:  # pragma: no cover
        error = {
            "mode": label,
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(error, indent=2))
        return error


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-mode evidence test for load_option_pnl_data")
    parser.add_argument("--opttick", required=True, help="Option ticker, e.g. HD20250620C500")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    print("Running dual-mode evidence test")
    print(f"opttick={args.opttick}, start={start.date()}, end={end.date()}")

    direct_result = _run_and_capture("direct", run_direct, args.opttick, start, end)
    payload_result = _run_and_capture("payload_datamanagers", run_payload, args.opttick, start, end)

    print("\n===== summary =====")
    summary = {
        "direct_status": direct_result.get("status"),
        "payload_status": payload_result.get("status"),
        "direct_rows": direct_result.get("attribution_rows"),
        "payload_rows": payload_result.get("attribution_rows"),
        "direct_aggregates": direct_result.get("aggregates"),
        "payload_aggregates": payload_result.get("aggregates"),
        "aggregate_differences": {},
    }

    direct_aggregates = summary.get("direct_aggregates") or {}
    payload_aggregates = summary.get("payload_aggregates") or {}
    all_keys = sorted(set(direct_aggregates.keys()) | set(payload_aggregates.keys()))
    for key in all_keys:
        direct_value = direct_aggregates.get(key, 0.0)
        payload_value = payload_aggregates.get(key, 0.0)
        summary["aggregate_differences"][key] = float(direct_value - payload_value)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
