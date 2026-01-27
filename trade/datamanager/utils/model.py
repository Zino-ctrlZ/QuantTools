import time
from trade.helpers.Logging import setup_logger
from trade.datamanager.result import DividendsResult, ModelResultPack, SpotResult, ForwardResult, RatesResult, VolatilityResult, OptionSpotResult
from trade.datamanager._enums import SeriesId
from trade.datamanager.requests import LoadRequest
from trade.datamanager.utils.date import DateRangePacket
from trade.datamanager.config import OptionDataConfig
from typing import Optional, Union
import pandas as pd
from trade.datamanager._enums import OptionSpotEndpointSource, VolatilityModel, OptionPricingModel
from trade.optionlib.config.types import DivType

logger = setup_logger("trade.datamanager.utils", stream_log_level="INFO")



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

    # Individual results (override packet fields if provided)
    spot: Optional[SpotResult] = None,
    dividend: Optional[DividendsResult] = None,
    rates: Optional[RatesResult] = None,
    forward: Optional[ForwardResult] = None,
    option_spot: Optional[OptionSpotResult] = None,
    vol: Optional[VolatilityResult] = None,

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

        # Optional strictness knobs (only check if caller passed them)
        # If caller provided series_id/endpoint_source etc, verify packet matches.
        if series_id is not None and packet.series_id is not None and packet.series_id != series_id:
            raise ValueError(f"series_id mismatch: expected {series_id}, packet has {packet.series_id}")
        if endpoint_source is not None and packet.endpoint_source is not None and packet.endpoint_source != endpoint_source:
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
    }

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
    # DividendsResult.dividend_type
    if dividend is not None and dividend.dividend_type is not None and dividend.dividend_type != dividend_type:
        raise ValueError(f"Dividend type mismatch: expected {dividend_type}, dividend has {dividend.dividend_type}")

    # ForwardResult.dividend_type
    if forward is not None and forward.dividend_type is not None and forward.dividend_type != dividend_type:
        raise ValueError(f"Dividend type mismatch: expected {dividend_type}, forward has {forward.dividend_type}")

    # VolatilityResult.dividend_type
    if vol is not None and vol.dividend_type is not None and vol.dividend_type != dividend_type:
        raise ValueError(f"Dividend type mismatch: expected {dividend_type}, vol has {vol.dividend_type}")

    # Optional strict checks for market model / vol model
    if market_model is not None and vol is not None and vol.market_model is not None and vol.market_model != market_model:
        raise ValueError(f"market_model mismatch: expected {market_model}, vol has {vol.market_model}")
    if vol_model is not None and vol is not None and vol.model is not None and vol.model != vol_model:
        raise ValueError(f"vol_model mismatch: expected {vol_model}, vol has {vol.model}")

    # -------------------------
    # 4) undo_adjust consistency + canon hard contract
    # -------------------------
    if spot is not None and spot.undo_adjust is not None and spot.undo_adjust != undo_adjust:
        raise ValueError(f"undo_adjust mismatch: expected {undo_adjust}, spot has {spot.undo_adjust}")

    if dividend is not None:
        if dividend.undo_adjust is None:
            raise ValueError("Dividend result present but dividend.undo_adjust is None (required for canon checks).")
        if dividend.undo_adjust != undo_adjust:
            raise ValueError(
                f"Spot/Dividend unit contract violated: dividend.undo_adjust={dividend.undo_adjust} "
                f"must equal undo_adjust={undo_adjust}."
            )

    if forward is not None and forward.undo_adjust is not None and forward.undo_adjust != undo_adjust:
        raise ValueError(f"undo_adjust mismatch: expected {undo_adjust}, forward has {forward.undo_adjust}")

    # -------------------------
    # 5) Timeseries alignment checks
    # -------------------------
    def _assert_dt_index(x: Union[pd.Series, pd.DataFrame], label: str) -> pd.DatetimeIndex:
        if not isinstance(x.index, pd.DatetimeIndex):
            raise TypeError(f"{label} index must be DatetimeIndex; got {type(x.index)}")
        if not x.index.is_monotonic_increasing:
            raise ValueError(f"{label} index must be sorted increasing.")
        return x.index

    # Build specific series for each result
    # NOTE: These are now specific and fast, no guessing.
    spot_s = spot.daily_spot if (spot is not None and spot.daily_spot is not None) else None

    # Dividends can be discrete or continuous depending on dividend_type
    div_s = None
    if dividend is not None:
        if dividend_type == DivType.DISCRETE:
            div_s = dividend.daily_discrete_dividends
        elif dividend_type == DivType.CONTINUOUS:
            div_s = dividend.daily_continuous_dividends

    # Rates
    rates_s = rates.daily_risk_free_rates if (rates is not None and rates.daily_risk_free_rates is not None) else None

    # Forward depends on dividend_type
    fwd_s = None
    if forward is not None:
        if dividend_type == DivType.DISCRETE:
            fwd_s = forward.daily_discrete_forward
        elif dividend_type == DivType.CONTINUOUS:
            fwd_s = forward.daily_continuous_forward

    # Option spot: anchor recommended
    opt_mid = option_spot.midpoint if (option_spot is not None) else None  # property returns empty Series if empty
    vol_s = vol.timeseries if (vol is not None and vol.timeseries is not None) else None

    series_map = {
        "spot": spot_s,
        "dividend": div_s,
        "rates": rates_s,
        "forward": fwd_s,
        "option_spot_midpoint": opt_mid,
        "vol": vol_s,
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




def _load_model_data_timeseries(load_request: LoadRequest) -> ModelResultPack:
    """
    Loads model data based on the provided load request.

    Parameters:
        load_request (LoadRequest): The request specifying what data to load.

    Returns:
        ModelResultPack: A container with the loaded model data.
    """
    ## Import here to avoid circular dependencies
    from trade.datamanager.dividend import DividendDataManager
    from trade.datamanager.rates import RatesDataManager
    from trade.datamanager.spot import SpotDataManager
    from trade.datamanager.forward import ForwardDataManager
    from trade.datamanager.option_spot import OptionSpotDataManager
    from trade.datamanager.vol import VolDataManager
    load_info = {}
    start_time = time.time()
    packet = DateRangePacket(
        start_date=load_request.start_date, end_date=load_request.end_date, maturity_date=load_request.expiration
    )
    load_info["date_range_packet"] = time.time() - start_time
    symbol = load_request.symbol
    start_date = packet.start_date
    end_date = packet.end_date
    expiration = packet.maturity_date
    d = load_request.load_dividend
    r = load_request.load_rates
    s = load_request.load_spot
    f = load_request.load_forward
    vol = load_request.load_vol
    opt_spot = load_request.load_option_spot
    dividend_type = load_request.dividend_type or OptionDataConfig().dividend_type
    D, R, S, F, V = None, None, None, None, None

    model_data = ModelResultPack()

    # Load BSM-specific data
    if d:
        start_time = time.time()
        D = DividendDataManager(symbol).get_schedule_timeseries(
            start_date=start_date,
            end_date=end_date,
            maturity_date=expiration,
            dividend_type=dividend_type,
            undo_adjust=load_request.undo_adjust,
        )
        load_info["dividend_load_time"] = time.time() - start_time

    if r:
        start_time = time.time()
        R = RatesDataManager().get_risk_free_rate_timeseries(start_date=start_date, end_date=end_date)
        load_info["rates_load_time"] = time.time() - start_time

    if s:
        start_time = time.time()
        S = SpotDataManager(symbol=symbol).get_spot_timeseries(
            start_date=start_date, end_date=end_date, undo_adjust=load_request.undo_adjust
        )
        load_info["spot_load_time"] = time.time() - start_time

    if f:
        start_time = time.time()
        F = ForwardDataManager(symbol=symbol).get_forward_timeseries(
            start_date=start_date,
            end_date=end_date,
            dividend_result=D,
            maturity_date=expiration,
            use_chain_spot=load_request.undo_adjust,
            dividend_type=dividend_type,
        )
        load_info["forward_load_time"] = time.time() - start_time

    if opt_spot:
        start_time = time.time()
        market_price = OptionSpotDataManager(symbol=symbol).get_option_spot_timeseries(
            start_date=start_date,
            end_date=end_date,
            expiration=expiration,
            strike=load_request.strike,
            right=load_request.right,
            endpoint_source=load_request.endpoint_source,
        )
        load_info["option_spot_load_time"] = time.time() - start_time
        model_data.option_spot = market_price

    if vol:
        start_time = time.time()
        V = VolDataManager(symbol=symbol).get_implied_volatility_timeseries(
            start_date=start_date,
            end_date=end_date,
            expiration=expiration,
            strike=load_request.strike,
            right=load_request.right,
            market_model=load_request.market_model,
            vol_model=load_request.vol_model,
            dividends=D,
            F=F,
            S=S,
            r=R,
            dividend_type=dividend_type,
            market_price=model_data.option_spot,
            undo_adjust=load_request.undo_adjust,
            endpoint_source=load_request.endpoint_source,
        )
        load_info["vol_load_time"] = time.time() - start_time
        model_data.vol = V

    model_data.dividend = D
    model_data.dividend_type = dividend_type
    model_data.forward = F
    model_data.rates = R
    model_data.spot = S
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
        ]
    ):
        logger.warning("No data requested to load in _load_model_data_timeseries().")
        return model_data
    ## Print what was loaded
    assert_synchronized_model(
        packet=model_data,
        symbol=symbol,
        undo_adjust=load_request.undo_adjust,
        dividend_type=dividend_type,
        require_anchor=False,
    )

    return model_data



