"""Model data orchestration: timeseries loading, point-in-time clip, and sync checks.

Resolves fetch windows, chains factor preloads (D→R→S→F→option_spot→vol→greeks), and
clips loaded windows to an anchor for ``rt``/``as_of`` without re-fetching.

Comment density: orchestration

Core Functions:
    _resolve_model_load_window: Hist uses request dates; rt/as_of expands lookback from anchor.
    _load_model_data_timeseries: Main orchestrator; always returns full-window timeseries.
    _load_model_data_as_date: Calls orchestrator, clips each factor to anchor.
    _load_model_data: Public router (hist vs point-in-time).
    assert_synchronized_model: Cross-factor alignment and metadata checks.

Processing Flow:
    1. Resolve window (explicit range or anchor + lookback).
    2. Load missing factors via timeseries getters with chained preloads.
    3. For rt/as_of: clip in-memory via ``_resolve_from_window`` (no second fetch).
"""

import time
from trade.helpers.Logging import setup_logger
from trade.datamanager.result import (
    DividendsResult,
    ModelResultPack,
    SpotResult,
    ForwardResult,
    RatesResult,
    VolatilityResult,
    OptionSpotResult,
    GreekResultSet,
)
import os
from trade.datamanager._enums import (
    ModelPrice,
    OptionPricingModel,
    OptionSpotEndpointSource,
    RealTimeFallbackOption,
    SeriesId,
    VolatilityModel,
)
from trade.datamanager.requests import LoadRequest
from trade.datamanager.config import OptionDataConfig
from copy import copy
from typing import Optional, Tuple, Union
import pandas as pd
from trade.optionlib.config.types import DivType
from trade.datamanager.utils.logging import get_logging_level, UTILS_LOGGER_NAME
from trade.datamanager.utils.na_logging import log_model_result_pack_na
from trade.datamanager.utils.point_in_time import _lookback_window_bounds, _resolve_from_window
from trade.datamanager.utils.date import DATE_HINT
from trade.helpers.helper import to_datetime
from trade.datamanager.vars import add_to_log_bucket

logger = setup_logger(UTILS_LOGGER_NAME, stream_log_level=get_logging_level())

def _get_print_diagnostics() -> bool:
    """
    Gets the print diagnostics flag from the environment variable.
    """
    return os.environ.get("PRINT_DIAGNOSTICS", "False").lower() == "true"

def log_model_load_info(
    log_info: dict,
    is_rt: bool,
    is_timeseries: bool,
    symbol: str,
    expiration: str,
    strike: float,
    right: str,
    dividend_type: str,
    market_model: str,
) -> None:
    """Logs model load information in a structured format."""
    log_info["symbol"] = symbol
    log_info["expiration"] = expiration
    log_info["strike"] = strike
    log_info["right"] = right
    log_info["dividend_type"] = dividend_type
    log_info["market_model"] = market_model
    log_info["is_rt"] = is_rt
    log_info["is_timeseries"] = is_timeseries
    log_info["date"] = pd.Timestamp.now().date().strftime("%Y-%m-%d")
    add_to_log_bucket(log_info)


def _adjust_div_yield_for_spot_shock(
    shock: float,
    div: float,
) -> float:
    """Adjust dividend yield based on spot price shock for continuous dividends."""
    adjusted_div = div / shock
    return adjusted_div


def assert_synchronized_model(
    packet: Optional[ModelResultPack] = None,
    *,
    # Hard-required guiding attributes (per your instruction)
    symbol: str,
    undo_adjust: bool,
    dividend_type: DivType,
    # Optional guiding attributes (enable if you want stricter checks)
    series_id: Optional[SeriesId] = None,
    endpoint_source: Optional[OptionSpotEndpointSource] = None,
    market_model: Optional[OptionPricingModel] = None,
    vol_model: Optional[VolatilityModel] = None,
    model_price: Optional[ModelPrice] = None,
    # Individual results (override packet fields if provided)
    spot: Optional[SpotResult] = None,
    dividend: Optional[DividendsResult] = None,
    rates: Optional[RatesResult] = None,
    forward: Optional[ForwardResult] = None,
    option_spot: Optional[OptionSpotResult] = None,
    vol: Optional[VolatilityResult] = None,
    greek: Optional[GreekResultSet] = None,
    # Point in time check
    is_rt: bool = True,
    check_fallback_option: bool = False,
    # Alignment policy
    anchor: str = "option_spot_midpoint",
    require_anchor: bool = True,
) -> None:
    """
    Authoritative synchronization checks for model inputs.

    Accepts either:
      - packet: ModelResultPack
      - and/or individual result overrides (spot=..., dividend=..., ...)

    Hard requirements (must not be None):
      - symbol: non-empty string
      - undo_adjust: bool
      - dividend_type: DivType

    Skips any individual result object that is None.

    Checks:
      1) Symbol consistency across all present results (allows result.symbol=None)
      2) Dividend type consistency across (dividend, forward, vol, packet.dividend_type)
      3) undo_adjust consistency across (spot, dividend, forward, packet.undo_adjust)
      4) Canon hard contract: dividend.undo_adjust must equal undo_adjust when dividend exists
      5) Date/index alignment: anchored on option_spot.midpoint by default
    """

    # -------------------------
    # 0) Validate required args
    # -------------------------
    if symbol is None or not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("assert_synchronized_model: `symbol` must be a non-empty string.")
    if undo_adjust is None or not isinstance(undo_adjust, bool):
        raise ValueError("assert_synchronized_model: `undo_adjust` must be a bool (True/False).")
    if dividend_type is None:
        raise ValueError("assert_synchronized_model: `dividend_type` must not be None.")

    # -------------------------
    # 1) Load from packet first
    # -------------------------
    if packet is not None:
        if spot is None:
            spot = packet.spot
        if dividend is None:
            dividend = packet.dividend
        if rates is None:
            rates = packet.rates
        if forward is None:
            forward = packet.forward
        if option_spot is None:
            option_spot = packet.option_spot
        if vol is None:
            vol = packet.vol
        if greek is None:
            greek = packet.greek

        # Optional strictness knobs (only check if caller passed them)
        # If caller provided series_id/endpoint_source etc, verify packet matches.
        if series_id is not None and packet.series_id is not None and packet.series_id != series_id:
            raise ValueError(f"series_id mismatch: expected {series_id}, packet has {packet.series_id}")
        if (
            endpoint_source is not None
            and packet.endpoint_source is not None
            and packet.endpoint_source != endpoint_source
        ):
            raise ValueError(
                f"endpoint_source mismatch: expected {endpoint_source}, packet has {packet.endpoint_source}"
            )

        # If packet carries dividend_type and caller supplied dividend_type, enforce.
        if packet.dividend_type is not None and packet.dividend_type != dividend_type:
            raise ValueError(f"dividend_type mismatch: expected {dividend_type}, packet has {packet.dividend_type}")

        # If packet carries undo_adjust and caller supplied undo_adjust, enforce.
        if packet.undo_adjust is not None and packet.undo_adjust != undo_adjust:
            raise ValueError(f"undo_adjust mismatch: expected {undo_adjust}, packet has {packet.undo_adjust}")

    results = {
        "spot": spot,
        "dividend": dividend,
        "rates": rates,
        "forward": forward,
        "option_spot": option_spot,
        "vol": vol,
        "greek": greek,
    }

    dividend_factors = [
        "dividend",
        "forward",
        "vol",
        "greek",
    ]

    vol_model_factors = [
        "vol",
        "greek",
    ]

    market_model_factors = [
        "vol",
        "greek",
    ]

    undo_adjust_factors = [
        "spot",
        "dividend",
        "forward",
        "greek",
        "vol",
    ]

    model_price_factors = ["vol", "option_spot", "greek"]

    fallback_option_factors = [
        "spot",
        "dividend",
        "rates",
        "forward",
        "vol",
        "greek",
    ]

    rt_factors = [
        "spot",
        "dividend",
        "rates",
        "forward",
        "option_spot",
        "vol",
        "greek",
    ]

    # -------------------------
    # 2) Symbol consistency
    # -------------------------
    for name, res in results.items():
        if res is None or name == "rates":
            continue
        res_sym = getattr(res, "symbol", None)
        if res_sym is None:
            continue
        if res_sym != symbol:
            raise ValueError(f"Symbol mismatch: expected symbol={symbol}, but {name}.symbol={res_sym}")

    # -------------------------
    # 3) Dividend type consistency
    # -------------------------

    # Generic dividend_type checks
    if dividend is not None:
        # Loop through all results that have dividend_type attribute
        for name in dividend_factors:
            res = results.get(name)
            if res is None:
                continue
            res_div_type = getattr(res, "dividend_type", None)
            if res_div_type is None:
                raise ValueError(f"{name} missing dividend_type attribute.")
            if res_div_type != dividend_type:
                raise ValueError(f"Dividend type mismatch: expected {dividend_type}, {name} has {res_div_type}")

    # Generic vol_model checks
    if vol_model is not None:
        for name in vol_model_factors:
            res = results.get(name)
            if res is None:
                continue
            res_vol_model = getattr(res, "vol_model", None)
            if vol_model is not None and res_vol_model is not None and res_vol_model != vol_model:
                raise ValueError(f"vol_model mismatch: expected {vol_model}, {name} has {res_vol_model}")

    # Generic market_model checks
    if market_model is not None:
        for name in market_model_factors:
            res = results.get(name)
            if res is None:
                continue
            res_market_model = getattr(res, "market_model", None)
            if market_model is not None and res_market_model is not None and res_market_model != market_model:
                raise ValueError(f"market_model mismatch: expected {market_model}, {name} has {res_market_model}")

    # Generic model_price checks
    if model_price is not None:
        for name in model_price_factors:
            res = results.get(name)
            if res is None:
                print("Skipping, result is None")
                continue
            res_model_price = getattr(res, "model_price", None)
            if res_model_price is None or res_model_price != model_price:
                raise ValueError(f"model_price mismatch: expected {model_price}, {name} has {res_model_price}")

    # -------------------------
    # 4) undo_adjust consistency + canon hard contract
    # -------------------------
    for name in undo_adjust_factors:
        res = results.get(name)
        if res is None:
            continue
        res_undo_adjust = getattr(res, "undo_adjust", None)
        if res_undo_adjust is None:
            raise ValueError(f"{name} missing undo_adjust attribute.")
        if res_undo_adjust != undo_adjust:
            raise ValueError(f"undo_adjust mismatch: expected {undo_adjust}, {name} has {res_undo_adjust}")

    if is_rt:
        for name in rt_factors:
            res = results.get(name)
            if res is None:
                continue
            res_rt = getattr(res, "rt", None)
            if res_rt is None:
                raise ValueError(f"{name} missing rt attribute.")
            if res_rt != is_rt:
                raise ValueError(f"rt mismatch: expected {is_rt}, {name} has {res_rt}")

    if check_fallback_option:
        for name in fallback_option_factors:
            res = results.get(name)
            if res is None:
                continue
            res_fallback_option = getattr(res, "fallback_option", None)
            if res_fallback_option is None:
                raise ValueError(f"{name} missing fallback_option attribute.")
            if not res_fallback_option:
                raise ValueError(f"fallback_option mismatch: expected True, {name} has {res_fallback_option}")

    # -------------------------
    # 5) Timeseries alignment checks
    # -------------------------
    def _assert_dt_index(x: Union[pd.Series, pd.DataFrame], label: str) -> pd.DatetimeIndex:
        if not isinstance(x.index, pd.DatetimeIndex):
            raise TypeError(f"{label} index must be DatetimeIndex; got {type(x.index)}")
        if not x.index.is_monotonic_increasing:
            raise ValueError(f"{label} index must be sorted increasing.")
        return x.index

    for name, res in results.items():
        if res is None:
            continue
        if res.timeseries is None:
            raise ValueError(f"{name} timeseries is None.")
        _assert_dt_index(res.timeseries, name)

    series_map = {
        "spot": spot.timeseries if spot is not None else None,
        "dividend": dividend.timeseries if dividend is not None else None,
        "rates": rates.daily_risk_free_rates if rates is not None else None,
        "forward": forward.timeseries if forward is not None else None,
        "option_spot_midpoint": option_spot.timeseries if option_spot is not None else None,
        "vol": vol.timeseries if vol is not None else None,
    }

    # Determine anchor
    if anchor not in series_map:
        raise ValueError(f"Unknown anchor='{anchor}'. Valid anchors: {list(series_map.keys())}")

    anchor_series = series_map[anchor]
    if require_anchor:
        if anchor_series is None:
            raise ValueError(f"Anchor '{anchor}' is None but require_anchor=True.")
        if isinstance(anchor_series, pd.Series) and anchor_series.empty:
            raise ValueError(f"Anchor '{anchor}' is empty but require_anchor=True.")
        if isinstance(anchor_series, pd.DataFrame) and anchor_series.empty:
            raise ValueError(f"Anchor '{anchor}' is empty but require_anchor=True.")

    # If no anchor (require_anchor=False) and no series, nothing to check.
    if anchor_series is None:
        return

    anchor_idx = _assert_dt_index(anchor_series, anchor)

    # Require overlap (intersection) with anchor for all other present series
    for name, s in series_map.items():
        if name == anchor or s is None:
            continue

        # empty is allowed (you may want to tighten this later)
        if isinstance(s, (pd.Series, pd.DataFrame)) and s.empty:
            continue

        idx = _assert_dt_index(s, name)
        inter = idx.intersection(anchor_idx)
        if len(inter) == 0:
            raise ValueError(
                f"Index intersection empty: '{name}' has [{idx.min().date()}..{idx.max().date()}], "
                f"anchor '{anchor}' has [{anchor_idx.min().date()}..{anchor_idx.max().date()}]."
            )

    # Optional: global intersection check for vectorized kernels
    common = None
    for _, s in series_map.items():
        if s is None or (isinstance(s, (pd.Series, pd.DataFrame)) and s.empty):
            continue
        idx = s.index
        common = idx if common is None else common.intersection(idx)

    if common is not None and len(common) == 0:
        raise ValueError(
            "All detected non-empty timeseries have an empty global index intersection. "
            f"Non-empty series: {[k for k,v in series_map.items() if isinstance(v,(pd.Series,pd.DataFrame)) and not v.empty]}"
        )

_DEFAULT_ORCHESTRATOR_LOOKBACK_BDATES = 10


def _resolve_model_load_window(
    load_request: LoadRequest,
    lookback_bdays: int = _DEFAULT_ORCHESTRATOR_LOOKBACK_BDATES,
) -> Tuple[Optional[DATE_HINT], DATE_HINT, DATE_HINT]:
    """Resolve anchor and fetch window for the model orchestrator.

    Historical requests use explicit ``start_date``/``end_date``. ``rt``/``as_of``
    requests peg start=end=anchor on ``LoadRequest``; the orchestrator expands to
    a lookback window for loading while the anchor remains the clip target.

    Args:
        load_request: Caller request with date mode flags.
        lookback_bdays: Business-day lookback for point-in-time modes.

    Returns:
        ``(anchor, start_date, end_date)`` where ``anchor`` is ``None`` for hist.
    """
    if load_request.on_date or load_request.rt:
        anchor = load_request.as_of or load_request.end_date
        start_str, end_str, anchor_dt = _lookback_window_bounds(
            anchor,
            lookback_bdays=lookback_bdays,
        )
        return anchor_dt, start_str, end_str
    return None, load_request.start_date, load_request.end_date


def _clip_result_at_anchor(
    result: Optional[Union[DividendsResult, RatesResult, SpotResult, ForwardResult, OptionSpotResult, VolatilityResult, GreekResultSet]],
    anchor: DATE_HINT,
    fallback_option: RealTimeFallbackOption,
) -> Optional[Union[DividendsResult, RatesResult, SpotResult, ForwardResult, OptionSpotResult, VolatilityResult, GreekResultSet]]:
    """Clip one loaded result to a single row at ``anchor`` without re-fetching.

    Args:
        result: Loaded factor result with a timeseries window.
        anchor: Valuation date to resolve.
        fallback_option: Policy when exact date is missing or non-trading.

    Returns:
        Copy of ``result`` with a single-row timeseries, or ``None``/empty input.
    """
    if result is None or result.is_empty():
        return result
    anchor_ts = pd.Timestamp(to_datetime(anchor)).normalize()
    row, _, _ = _resolve_from_window(
        result.timeseries,
        anchor_ts,
        fallback_option,
        key=result.key or result.__class__.__name__,
    )
    clipped = copy(result)
    clipped.timeseries = row
    clipped.rt = True
    clipped.fallback_option = fallback_option
    return clipped


def _clip_model_packet_at_anchor(
    packet: ModelResultPack,
    anchor: DATE_HINT,
    fallback_option: RealTimeFallbackOption,
) -> ModelResultPack:
    """Clip every loaded factor in a model packet to one row at ``anchor``.

    Args:
        packet: Full-window ``ModelResultPack`` from the timeseries orchestrator.
        anchor: Valuation date to resolve.
        fallback_option: Policy when exact date is missing or non-trading.

    Returns:
        Copy of ``packet`` with single-row factors and ``rt=True``.
    """
    clipped = copy(packet)
    clipped.dividend = _clip_result_at_anchor(packet.dividend, anchor, fallback_option)
    clipped.rates = _clip_result_at_anchor(packet.rates, anchor, fallback_option)
    clipped.spot = _clip_result_at_anchor(packet.spot, anchor, fallback_option)
    clipped.forward = _clip_result_at_anchor(packet.forward, anchor, fallback_option)
    clipped.option_spot = _clip_result_at_anchor(packet.option_spot, anchor, fallback_option)
    clipped.vol = _clip_result_at_anchor(packet.vol, anchor, fallback_option)
    clipped.greek = _clip_result_at_anchor(packet.greek, anchor, fallback_option)
    clipped.rt = True
    return clipped


def _load_model_data_timeseries(load_request: LoadRequest) -> ModelResultPack:
    """Load all requested factors over the orchestrator-resolved window.

    Always uses timeseries getters and chains preloads (D→R→S→F→option_spot→vol→greeks).
    For ``rt``/``as_of``, ``LoadRequest`` pegs start=end=anchor; this function expands
    the fetch window via lookback. Clip to anchor via ``_load_model_data_as_date``.

    Args:
        load_request: Flags and payloads describing which factors to load.

    Returns:
        ``ModelResultPack`` with full-window timeseries for each requested factor.
    """
    ## Import here to avoid circular dependencies
    from trade.datamanager.dividend import DividendDataManager
    from trade.datamanager.rates import RatesDataManager
    from trade.datamanager.spot import SpotDataManager
    from trade.datamanager.forward import ForwardDataManager
    from trade.datamanager.option_spot import OptionSpotDataManager
    from trade.datamanager.vol import VolDataManager
    from trade.datamanager.greeks import GreekDataManager

    _, start_date, end_date = _resolve_model_load_window(load_request)
    start_date = to_datetime(start_date)
    end_date = to_datetime(end_date)

    load_info = {}
    start_time = time.time()
    load_info["date_range_packet"] = time.time() - start_time
    symbol = load_request.symbol
    expiration = load_request.expiration
    d = load_request.load_dividend
    r = load_request.load_rates
    s = load_request.load_spot
    f = load_request.load_forward
    vol = load_request.load_vol
    opt_spot = load_request.load_option_spot
    greek = load_request.load_greek
    model_price = load_request.model_price
    dividend_type = load_request.dividend_type or OptionDataConfig().dividend_type
    D, R, S, F, V, G, OPTION_SPOT = (
        load_request.dividend_timeseries,
        load_request.rates_timeseries,
        load_request.spot_timeseries,
        load_request.forward_timeseries,
        load_request.vol_timeseries,
        load_request.greek_timeseries,
        load_request.option_spot_timeseries,
    )
    if _get_print_diagnostics():
        print("Parameters for request:")
        for k, v in load_request.__dict__.items():
            print(f"{k}: {v}")
        print("\n")
    model_data = ModelResultPack()

    if d:
        logger.info(f"Loading dividend data for symbol: {symbol}, expiration: {expiration}, dividend_type: {dividend_type}, undo_adjust: {load_request.undo_adjust}")
        start_time = time.time()
        d_params = {
            "maturity_date": expiration,
            "dividend_type": dividend_type,
            "undo_adjust": load_request.undo_adjust,
        }
        if _get_print_diagnostics():
            print(f"d_params: {d_params}")
        D = DividendDataManager(symbol).get_schedule_timeseries(
            start_date=start_date,
            end_date=end_date,
            **d_params,
        )
        load_info["dividend_load_time"] = time.time() - start_time

    if r:
        logger.info(f"Loading rates data for symbol: {symbol}, expiration: {expiration}, dividend_type: {dividend_type}, undo_adjust: {load_request.undo_adjust}")
        start_time = time.time()
        if _get_print_diagnostics():
            print(
                f"Rates params: start_date={start_date}, end_date={end_date}, "
                f"fallback_option={load_request.fall_back_option}"
            )
        R = RatesDataManager().get_risk_free_rate_timeseries(
            start_date=start_date,
            end_date=end_date,
        )
        load_info["rates_load_time"] = time.time() - start_time

    if s:
        logger.info(f"Loading spot data for symbol: {symbol}, expiration: {expiration}, dividend_type: {dividend_type}, undo_adjust: {load_request.undo_adjust}")
        start_time = time.time()
        if _get_print_diagnostics():
            print(
                f"Spot params: start_date={start_date}, end_date={end_date}, "
                f"undo_adjust={load_request.undo_adjust}"
            )
        S = SpotDataManager(symbol).get_spot_timeseries(
            start_date=start_date,
            end_date=end_date,
            undo_adjust=load_request.undo_adjust,
        )
        load_info["spot_load_time"] = time.time() - start_time

    if f:
        logger.info(f"Loading forward data for symbol: {symbol}, expiration: {expiration}, dividend_type: {dividend_type}, undo_adjust: {load_request.undo_adjust}")
        start_time = time.time()
        f_params = {
            "maturity_date": expiration,
            "use_chain_spot": load_request.undo_adjust,
            "dividend_type": dividend_type,
            "dividend_result": D,
            "spot": S,
            "rates": R,
        }
        if _get_print_diagnostics():
            print(f"Forward params: {f_params}")
        F = ForwardDataManager(symbol=symbol).get_forward_timeseries(
            start_date=start_date,
            end_date=end_date,
            **f_params,
        )
        load_info["forward_load_time"] = time.time() - start_time

    if opt_spot:
        logger.info(f"Loading option spot data for symbol: {symbol}, expiration: {expiration}, dividend_type: {dividend_type}, undo_adjust: {load_request.undo_adjust}")
        start_time = time.time()
        opt_params = {
            "expiration": expiration,
            "strike": load_request.strike,
            "right": load_request.right,
            "endpoint_source": load_request.endpoint_source,
            "model_price": model_price,
        }
        if _get_print_diagnostics():
            print(f"Option spot params: {opt_params}")
        market_price = OptionSpotDataManager(symbol=symbol).get_option_spot_timeseries(
            start_date=start_date,
            end_date=end_date,
            **opt_params,
        )
        load_info["option_spot_load_time"] = time.time() - start_time
        OPTION_SPOT = market_price

    if vol:
        logger.info(f"Loading implied volatility data for symbol: {symbol}, expiration: {expiration}, dividend_type: {dividend_type}, undo_adjust: {load_request.undo_adjust}")
        start_time = time.time()
        v_params = {
            "expiration": expiration,
            "strike": load_request.strike,
            "right": load_request.right,
            "market_model": load_request.market_model,
            "vol_model": load_request.vol_model,
            "undo_adjust": load_request.undo_adjust,
            "endpoint_source": load_request.endpoint_source,
            "model_price": model_price,
            "dividend_type": dividend_type,
            "dividends": D,
            "F": F,
            "S": S,
            "r": R,
            "market_price": OPTION_SPOT,
        }
        if _get_print_diagnostics():
            print(f"Vol params: {v_params}")
        V = VolDataManager(symbol=symbol).get_implied_volatility_timeseries(
            start_date=start_date,
            end_date=end_date,
            **v_params,
        )
        load_info["vol_load_time"] = time.time() - start_time
        model_data.vol = V

    if greek:
        logger.info(f"Loading greek data for symbol: {symbol}, expiration: {expiration}, dividend_type: {dividend_type}, undo_adjust: {load_request.undo_adjust}")
        start_time = time.time()
        grk_params = {
            "expiration": expiration,
            "strike": load_request.strike,
            "right": load_request.right,
            "market_model": load_request.market_model,
            "dividend_type": dividend_type,
            "undo_adjust": load_request.undo_adjust,
            "endpoint_source": load_request.endpoint_source,
            "model_price": model_price,
            "d": D,
            "f": F,
            "S": S,
            "r": R,
            "vol": V,
        }
        G = GreekDataManager(symbol=symbol).get_greeks_timeseries(
            start_date=start_date,
            end_date=end_date,
            **grk_params,
        )
        load_info["greek_load_time"] = time.time() - start_time
        model_data.greek = G

    model_data.dividend = D
    model_data.dividend_type = dividend_type
    model_data.forward = F
    model_data.rates = R
    model_data.spot = S
    model_data.option_spot = OPTION_SPOT
    model_data.series_id = SeriesId.HIST
    model_data.undo_adjust = load_request.undo_adjust
    model_data.time_to_load = load_info
    model_data.endpoint_source = load_request.endpoint_source

    if not any(
        [
            load_request.load_dividend,
            load_request.load_rates,
            load_request.load_spot,
            load_request.load_forward,
            load_request.load_option_spot,
            load_request.load_vol,
            load_request.load_greek,
        ]
    ):
        logger.info(
            (
                "No data requested to load in _load_model_data_timeseries()."
                f" Option: Symbol={symbol}, exp={expiration}, strike={load_request.strike} "
                f"right={load_request.right}"
                f" Load bools: d={d}, r={r}, s={s}, f={f}, opt_spot={opt_spot}, vol={vol}, greek={greek}"
            )
        )
        return model_data

    assert_synchronized_model(
        packet=model_data,
        symbol=symbol,
        undo_adjust=load_request.undo_adjust,
        dividend_type=dividend_type,
        require_anchor=model_data.option_spot is not None,
        is_rt=False,
        check_fallback_option=False,
    )
    model_data.rt = False
    model_data.market_model = load_request.market_model or OptionDataConfig().option_model
    model_data.vol_model = load_request.vol_model or OptionDataConfig().volatility_model
    model_data.model_price = load_request.model_price or OptionDataConfig().model_price

    if load_info:
        log_model_load_info(
            log_info=load_info,
            is_rt=False,
            is_timeseries=True,
            symbol=symbol,
            expiration=expiration,
            strike=load_request.strike,
            right=load_request.right,
            dividend_type=dividend_type,
            market_model=load_request.market_model.name if load_request.market_model else "N/A",
        )
    log_model_result_pack_na(model_data)
    return model_data


def _load_model_data_as_date(load_request: LoadRequest) -> ModelResultPack:
    """Load full lookback window, then clip each factor to the request anchor.

    Args:
        load_request: Point-in-time or real-time request (``on_date`` or ``rt``).

    Returns:
        ``ModelResultPack`` with single-row factors at the anchor date.
    """
    anchor, _, _ = _resolve_model_load_window(load_request)
    fallback = load_request.fall_back_option or RealTimeFallbackOption.USE_LAST_AVAILABLE

    ## Orchestrator loads the expanded window; anchor on the request is only the clip target.
    packet = _load_model_data_timeseries(load_request)

    clipped = _clip_model_packet_at_anchor(packet, anchor, fallback)
    clipped.series_id = SeriesId.AT_TIME
    clipped.market_model = packet.market_model
    clipped.vol_model = packet.vol_model
    clipped.model_price = packet.model_price
    clipped.time_to_load = packet.time_to_load

    assert_synchronized_model(
        packet=clipped,
        symbol=load_request.symbol,
        undo_adjust=load_request.undo_adjust,
        dividend_type=clipped.dividend_type,
        require_anchor=clipped.option_spot is not None,
        is_rt=True,
        check_fallback_option=True,
    )
    log_model_result_pack_na(clipped)
    return clipped


def _load_model_data(load_request: LoadRequest) -> ModelResultPack:
    """Route model loading by date mode.

    Args:
        load_request: Historical window or point-in-time/real-time anchor request.

    Returns:
        Full-window pack for hist; single-row clipped pack for ``rt``/``as_of``.
    """
    if load_request.on_date or load_request.rt:
        return _load_model_data_as_date(load_request)
    return _load_model_data_timeseries(load_request)
