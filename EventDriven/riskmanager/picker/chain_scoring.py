import numpy as np
from EventDriven.configs.core import ScoringConfigs
import pandas as pd
from trade.helpers.Logging import setup_logger

logger = setup_logger("EventDriven.riskmanager.picker.chain_scoring")


def directional_moneyness(strike, forward, option_type):
    option_type = option_type.lower()
    if option_type == "put":
        return strike / forward
    elif option_type == "call":
        return forward / strike
    raise ValueError("option_type must be 'call' or 'put'")


def moneyness_score(
    m,
    target_m=0.8,
    sigma=0.1,
    tilt="otm",
    tilt_strength=0.1,
):

    base = np.exp(-((m - target_m) ** 2) / (2 * sigma**2))

    if tilt == "flat":
        mult = 1.0
    elif tilt == "otm":
        mult = 1.0 + tilt_strength if m < 1.0 else 1.0 - tilt_strength
    elif tilt == "itm":
        mult = 1.0 + tilt_strength if m > 1.0 else 1.0 - tilt_strength
    elif tilt == "atm":
        mult = 1.0 + tilt_strength * np.exp(-((m - 1.0) ** 2) / (2 * sigma**2))
    else:
        raise ValueError("tilt must be one of: flat, otm, itm, atm")

    return 10.0 * base * mult


def dte_score(
    dte,
    target_dte=60,
    sigma=10,
    tilt="flat",
    tilt_strength=0.1,
):
    """
    Score DTE around a target using a Gaussian-style score.

    Parameters
    ----------
    dte : float
        Contract DTE in calendar days.
    target_dte : float
        Desired DTE target.
    sigma : float
        Width / tolerance in days.
    tilt : str
        One of: 'flat', 'short', 'long'
    tilt_strength : float
        Preference strength when DTE is on one side of target.

    Returns
    -------
    float
        Score scaled to max about 10.
    """
    base = np.exp(-((dte - target_dte) ** 2) / (2 * sigma**2))

    if tilt == "flat":
        mult = 1.0
    elif tilt == "short":
        mult = 1.0 + tilt_strength if dte < target_dte else 1.0 - tilt_strength
    elif tilt == "long":
        mult = 1.0 + tilt_strength if dte > target_dte else 1.0 - tilt_strength
    else:
        raise ValueError("tilt must be one of: 'flat', 'short', 'long'")

    return 10.0 * base * mult


def mid_band_score(mid, min_mid=1.0, max_mid=5.0, sigma=1.0):
    if mid <= 0:
        return 0.0

    if min_mid <= mid <= max_mid:
        return 10.0

    if mid < min_mid:
        diff = min_mid - mid
    else:
        diff = mid - max_mid

    return 10.0 * np.exp(-(diff**2) / (2 * sigma**2))


def pct_spread_score(bid, ask, max_pct_spread=0.20, sigma=0.10):
    if bid <= 0 or ask <= 0 or ask < bid:
        return 0.0

    mid = 0.5 * (bid + ask)
    if mid <= 0:
        return 0.0

    pct_spread = (ask - bid) / mid

    if pct_spread <= max_pct_spread:
        return 10.0

    excess = pct_spread - max_pct_spread
    return 10.0 * np.exp(-(excess**2) / (2 * sigma**2))


def oi_score(oi, target_oi=500):
    """
    Soft score for open interest.
    - 0 if oi <= 0
    - approaches 10 as oi gets large
    - log scaling avoids huge OI dominating
    """
    if oi is None or oi <= 0:
        return 0.0

    return 10.0 * min(np.log1p(oi) / np.log1p(target_oi), 1.0)


def theta_burden_score_from_mid(theta, mid, max_theta_burden=0.03, sigma=0.02):
    if mid is None or mid <= 0:
        return 0.0

    if np.isnan(theta) or np.isnan(mid):
        return 0.0

    burden = abs(theta) / mid

    if burden <= max_theta_burden:
        return 10.0

    excess = burden - max_theta_burden
    return 10.0 * np.exp(-(excess**2) / (2 * sigma**2))


def _score_chain(structure_chain: pd.DataFrame, configs: ScoringConfigs) -> pd.DataFrame:
    """
    Score the chain using the defined scoring functions and configs.
    """

    structure_chain["moneyness_score"] = structure_chain["spread_moneyness"].apply(
        lambda x: (
            moneyness_score(
                m=np.log(x),
                target_m=np.log(configs.m_target),
                sigma=configs.m_sigma,
                tilt=configs.m_tilt,
                tilt_strength=configs.tilt_strength,
            )
            if pd.notna(x) and x > 0
            else np.nan
        )
    )
    structure_chain["dte_score"] = structure_chain["spread_dte"].apply(
        lambda x: (
            dte_score(
                dte=x,
                target_dte=configs.target_dte,
                sigma=configs.dte_sigma,
                tilt=configs.dte_tilt,
                tilt_strength=configs.tilt_strength,
            )
            if pd.notna(x)
            else np.nan
        )
    )
    structure_chain["mid_score"] = structure_chain["spread_mid"].apply(
        lambda x: mid_band_score(mid=x, min_mid=configs.mid_min, max_mid=configs.mid_max, sigma=configs.mid_sigma)
    )

    structure_chain["pct_spread_score"] = structure_chain.apply(
        lambda row: pct_spread_score(
            bid=row["spread_bid"],
            ask=row["spread_ask"],
            max_pct_spread=configs.target_spread_pct,
            sigma=configs.pct_spread_sigma,
        ),
        axis=1,
    )
    structure_chain["oi_score"] = structure_chain["spread_oi"].apply(
        lambda x: oi_score(
            oi=x,
            target_oi=configs.oi_target,
        )
    )

    structure_chain["theta_burden_score"] = structure_chain.apply(
        lambda row: theta_burden_score_from_mid(
            theta=row["spread_theta"],
            mid=row["spread_mid"],
            max_theta_burden=configs.theta_burden_max,
            sigma=configs.theta_burden_sigma,
        ),
        axis=1,
    )
    structure_chain["total_score"] = (
        structure_chain["moneyness_score"]
        + structure_chain["dte_score"]
        + structure_chain["mid_score"]
        + structure_chain["pct_spread_score"]
        + structure_chain["oi_score"]
        + structure_chain["theta_burden_score"]
    )
    return structure_chain
