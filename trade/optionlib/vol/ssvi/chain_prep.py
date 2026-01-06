"""
Module for preparing and checking option chains for SSVI volatility modeling.
Includes functions for intrinsic value calculation, European boundary checks,
and a ChainChecklist class for various chain transformations and validations.
Author: Chiemelie Nwanisobi
Date: 2025-10-01
"""

from typing import Literal, List
from datetime import datetime
import numpy as np
import pandas as pd
from trade.optionlib.config.ssvi.controller import logger
from trade.optionlib.pricing.black_scholes import black_scholes_vectorized
from trade.optionlib.vol.implied_vol import bsm_vol_est_brute_force, vector_vol_estimation
from trade.optionlib.greeks.numerical.binomial import binomial_tree_price_batch
from trade.optionlib.vol.ssvi.utils import (
    get_rates,
    format_chain,
    confine_chain_with_pricing_config,
    get_bs_vol_on_chain,
)
from trade.optionlib.utils.batch_operation import vector_batch_processor


def intrinsic_value(strike: float, spot: float, right: Literal["c", "p"]) -> float:
    """
    Calculate the intrinsic value of an option.

    Args:
        strike (float): The strike price of the option.
        spot (float): The current spot price of the underlying asset.
        right (Literal['c', 'p']): The type of option ('c' for call, 'p' for put).

    Returns:
        float: The intrinsic value of the option.
    """
    if right.lower() == "c":
        return max(0, spot - strike)
    elif right.lower() == "p":
        return max(0, strike - spot)
    else:
        raise ValueError(f"Invalid option type: {right}. Expected 'c' or 'p'.")


def vector_eu_boundary(
    f: np.ndarray,
    strike: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """
    Calculate the European option boundary values.

    Args:
        f (np.ndarray): Forward prices.
        strike (np.ndarray): Strike prices.
        t (np.ndarray): Time to maturity.
        r (np.ndarray): Risk-free rates.
        right (np.ndarray): Option types ('c' for call, 'p' for put).

    Returns:
        np.ndarray: The boundary values of the European options.
    """
    f = np.asarray(f)
    strike = np.asarray(strike)
    t = np.asarray(t)
    r = np.asarray(r)
    right = np.asarray(right)
    if f.shape != strike.shape or f.shape != t.shape or f.shape != r.shape or f.shape != right.shape:
        raise ValueError("All input arrays must have the same shape.")

    intrinsic_values = np.zeros_like(f)
    call = right == "c"
    put = right == "p"
    intrinsic_values[call] = np.maximum(0, f[call] - strike[call])
    intrinsic_values[put] = np.maximum(0, strike[put] - f[put])
    boundary = intrinsic_values * np.exp(-r * t)
    return boundary


class ChainChecklist:
    """
    A class to perform various checks and transformations on option chain data.
    This class includes methods to prepare the chain, remove junk quotes, and more.
    """

    @staticmethod
    def chain_prep(chain: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the option chain DataFrame for further processing.
        Runs through various transformations.

        Args:
            chain (pd.DataFrame): The option chain DataFrame.

        Returns:
            pd.DataFrame: The prepared option chain DataFrame.
        """
        raise NotImplementedError("chain_prep method is not implemented yet.")

    @staticmethod
    def remove_junk_quotes(chain: pd.DataFrame) -> pd.DataFrame:
        """
        Removes junk quotes from the option chain DataFrame.

        Args:
            chain (pd.DataFrame): The option chain DataFrame.

        Returns:
            pd.DataFrame: The cleaned option chain DataFrame.
        """

        chain = chain.copy()

        ## Format chain
        chain = format_chain(chain)
        logger.info("Initial chain length: %d", len(chain))
        ## Drop midpoint < intrinsic value
        chain["intrinsic_value"] = chain.apply(
            lambda x: intrinsic_value(
                x["strike"],
                x["f"],  ## Use Forward Price for intrinsic value instead of spot price
                x["right"],
            ),
            axis=1,
        )

        ## Drop below European lower bound
        chain["eu_lower_bound"] = vector_eu_boundary(
            chain["f"].tolist(),
            chain["strike"].tolist(),
            chain["t"].tolist(),
            [get_rates(chain["valuation_date"].iloc[0].strftime("%Y-%m-%d"))] * len(chain),
            chain["right"].str.lower().tolist(),
        )

        ## American Options cannot be worth less than max(intrinsic value, european lower bound, 0)
        ## Less than intrinsic value: Exercise
        ## Less than european lower bound: Arbitrage Violation
        chain["lower_bound"] = chain.apply(
            lambda x: max(
                # x['intrinsic_value'],
                x["eu_lower_bound"],
                0,
            ),
            axis=1,
        )

        ## Upper Bound is Spot for Call, Strike for Put
        chain["upper_bound"] = chain.apply(lambda x: x["spot"] if x["right"] == "c" else x["strike"], axis=1)
        chain = chain[chain["midpoint"] >= chain["lower_bound"]]
        logger.info("Chain length after dropping below lower bound: %d", len(chain))
        chain = chain[chain["midpoint"] <= chain["upper_bound"]]
        logger.info("Chain length after dropping above upper bound: %d", len(chain))

        ## Confine chain with pricing config
        chain = confine_chain_with_pricing_config(chain)
        logger.info("Chain length after confining with pricing config: %d", len(chain))

        return chain

    @staticmethod
    def get_european_price(
        chain: pd.DataFrame, bs_vol: np.ndarray, forward_col_name: str = "f", rates_col_name: str = None
    ) -> pd.Series:
        """
        Calculates the European price for the options in the chain.
        Args:
            chain (pd.DataFrame): The option chain DataFrame.
        Returns:
            pd.Series: The European price for each option in the chain.
        """
        if rates_col_name is None:
            _r = [get_rates(chain["valuation_date"].iloc[0].strftime("%Y-%m-%d"))] * len(chain)
        else:
            _r = chain[rates_col_name].tolist()

        chain = chain.copy()
        european_price_params = [
            chain[forward_col_name if forward_col_name in chain.columns else "f"].tolist(),
            chain["strike"].tolist(),
            chain["t"].tolist(),
            _r,  # Risk-free rate
            bs_vol,
            chain["right"].str.lower().tolist(),
        ]

        european_midpoint = black_scholes_vectorized(*european_price_params)
        return pd.Series(european_midpoint, index=chain.index)

    @staticmethod
    def get_american_price(
        chain: pd.DataFrame, sigmas: np.ndarray, rates_col_name: str = None, n: int = 500
    ) -> pd.Series:
        """
        Calculates the American price for the options in the chain using a binomial tree.

        Args:
            chain (pd.DataFrame): The option chain DataFrame.
            n (int): The number of steps in the binomial tree.

        Returns:
            pd.Series: The American price for each option in the chain.
        """
        chain = chain.copy()
        val_date = chain["valuation_date"].iloc[0].strftime("%Y-%m-%d")
        if rates_col_name is None:
            _r = [get_rates(val_date)] * len(chain)
        else:
            _r = chain[rates_col_name].tolist()
        crr_params = [
            chain["strike"].tolist(),
            chain["expiration"].tolist(),
            sigmas,
            _r,  # Risk-free rate
            [n] * len(chain),  # number of steps
            chain["spot"].tolist(),
            ["discrete"] * len(chain),  # Dividend type
            chain["div_schedule"].tolist(),  # Dividend schedules
            chain["right"].str.lower().tolist(),
            chain["valuation_date"].tolist(),  # Start dates
            chain["valuation_date"].tolist(),  # Valuation dates
            [True] * len(chain),  # American options
        ]

        def batch_hacked(*args):
            """
            A batch processor to handle the CRR binomial pricing.
            """
            return binomial_tree_price_batch(*args)[0]

        american_midpoint = vector_batch_processor(batch_hacked, *crr_params)
        chain["american_midpoint"] = american_midpoint
        return pd.Series(american_midpoint, index=chain.index)

    @staticmethod
    def run_calc_task(
        chain: pd.DataFrame,
        seed_vol: List[float],
        n: int = 500,
        forward_col_name: str = "f",
        rates_col_name: str = None,
    ) -> pd.DataFrame:
        """
        Calculates the European equivalent prices for the options in the chain.

        Args:
            chain (pd.DataFrame): The option chain DataFrame.
            N (int): The number of steps in the binomial tree.

        Returns:
            pd.DataFrame: The option chain DataFrame with European equivalent prices.
        """
        chain = chain.copy()
        mid = chain["midpoint"].to_numpy()

        ## Using bs_vol as seed because it is backed out of the midpoint
        # seed_vol = list(chain['bs_vol'].to_numpy())

        ## Using Midpoint as initial European price because seed_vol is backed out of it
        p_eu_init = ChainChecklist.get_european_price(
            chain=chain, bs_vol=seed_vol, forward_col_name=forward_col_name, rates_col_name=rates_col_name
        )

        ## Calculate American prices using CRR Binomial model and Seed Vol
        p_am = ChainChecklist.get_american_price(chain=chain, sigmas=seed_vol, n=n, rates_col_name=rates_col_name)

        ## Calculate Early Exercise Premium (EEP) and European Equivalent Price
        EEP = np.array(p_am - p_eu_init)
        euro_eq_mid = list(mid - EEP)

        ## Calculate European equivalent volatilities
        sigmas = ChainChecklist.get_bs_vol_on_chain(
            chain,
            chain["valuation_date"].iloc[0].strftime("%Y-%m-%d"),
            euro_eq_mid,
            rate_col_name=rates_col_name,
            forward_col_name=forward_col_name,
        )

        chain["european_midpoint"] = p_eu_init
        chain["european_vols_equiv"] = sigmas
        chain["american_midpoint"] = p_am
        chain["early_exercise_premium"] = EEP
        chain["european_equivalent_mid"] = euro_eq_mid
        return chain

    @staticmethod
    def calculate_european_equivalent_vols(
        chain: pd.DataFrame,
        n: int = 500,
        iteration: int = 4,
        seed_vol_col: str = None,
        forward_col_name: str = "f",
        rates_col_name: str = None,
        valuation_date: str | datetime = None,
    ) -> pd.DataFrame:
        """
        Iterates the run_calc_task to refine the European equivalent prices and volatilities.
        """

        def _name_not_include_error(col_name: str, columns: pd.Index) -> bool:
            if col_name not in columns:
                raise ValueError(f"{col_name} not found in chain columns: {columns.tolist()}")
            return False

        ## Valuation date validation
        if valuation_date is None:
            try:
                valuation_date = pd.to_datetime(chain["valuation_date"].iloc[0])
            except Exception as e:
                raise ValueError(
                    "valuation_date must be provided if chain does not contain 'valuation_date' column."
                ) from e
        else:
            valuation_date = pd.to_datetime(valuation_date)

        if rates_col_name is None:
            rates_col_name = "risk_free_rate"

        ## Rates column validation
        if rates_col_name not in chain.columns:
            if rates_col_name != "risk_free_rate":
                print(f"Warning: {rates_col_name} not found in chain columns. Defaulting to 'risk_free_rate'.")
            rates_col_name = "risk_free_rate"
            chain[rates_col_name] = get_rates(valuation_date.strftime("%Y-%m-%d"))

        ## Seed Vol column validation
        if seed_vol_col is None:
            seed_vol_col = "bs_vol"
            chain[seed_vol_col] = get_bs_vol_on_chain(
                chain=chain,
                valuation_date=chain["valuation_date"].iloc[0].strftime("%Y-%m-%d"),
                mid_col_name="midpoint",
                rate_col_name=rates_col_name,
                forward_col_name=forward_col_name,
            )

        ## Seed vol column validation P2
        elif seed_vol_col not in chain.columns:
            _name_not_include_error(seed_vol_col, chain.columns)

        ## Forward column validation
        _name_not_include_error(forward_col_name, chain.columns)

        ## Begin process
        seed_vol = list(chain[seed_vol_col].to_numpy())
        for i in range(iteration):
            print(f"Iteration {i+1} of {iteration}")
            chain = ChainChecklist.run_calc_task(
                chain, seed_vol, n, forward_col_name=forward_col_name, rates_col_name=rates_col_name
            )

            if i == iteration - 1:
                break  ## Last iteration, no need to reset variables

            ## Reset Variables for rerun
            seed_vol = list(chain["european_vols_equiv"].to_numpy())
        return chain

    @staticmethod
    def get_bs_vol_on_chain(
        chain: pd.DataFrame,
        valuation_date: str,
        midpoints: pd.Series,
        rate_col_name: str = None,
        forward_col_name: str = "f",
    ) -> pd.Series:
        """
        Estimates the Black-Scholes implied volatility for a given option chain.

        Args:
            chain (pd.DataFrame): The option chain DataFrame.
                Expected Columns: `f`, `strike`, `t`, `midpoint`, `right`.
            valuation_date (str): The date of valuation.

        Returns:
            pd.Series: The estimated Black-Scholes implied volatility for the option chain.
        """
        if rate_col_name is None:
            _r = [get_rates(valuation_date)] * len(chain)

        else:
            _r = chain[rate_col_name]
        params = list(
            zip(
                chain[forward_col_name if forward_col_name in chain.columns else "f"],
                chain["strike"],
                chain["t"],
                _r,
                midpoints,
                chain["right"].str.lower(),
            )
        )
        return vector_batch_processor(
            vector_vol_estimation,
            bsm_vol_est_brute_force,
            params,
        )
