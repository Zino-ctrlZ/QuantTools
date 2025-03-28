import pandas as pd
import numpy as np
from threading import Thread
from trade.helpers.helper import generate_option_tick_new, time_distance_helper, IV_handler, save_vol_resolve
from trade.models.VolSurface import fit_svi_model
from trade.helpers.Logging import setup_logger
from dbase.DataAPI.ThetaData import (
        list_contracts,
        retrieve_eod_ohlc
)

logger = setup_logger('trade.models.utils')


def resolve_missing_vol(
        underlier: str,
        expiration: str,
        strike: float,
        put_call: str,
        datetime: str,
        S: float,
        r: float,
        q: float,
        width = 3,
        **kwargs
) -> float:
        """
        Interpolates the implied volatility of a given option based on the volatility of the options with the closest strikes.

        params:
        -------
        underlier: str : Underlier symbol
        expiration: str : Expiration date
        strike: float : Strike price
        put_call: str : Put or Call
        datetime: str : Date to retrieve chain
        S: float : Spot price
        r: float : Risk free rate
        q: float : Dividend rate
        width: int : Width of the interpolation
        kwargs: dict : Additional arguments
                print_url: bool : Print the URL
                max_width: int : Maximum width of the interpolation
                return_full: bool : Return the full dataframe if True and only the interpolated value if False
                                    If True, but the interpolation fails, then the function will Vol and Model

        returns:
        --------
        float : Implied Volatility
        """
        # Get the strike of the next closest option
        range_ = np.arange(-width, width+1, 1)
        
        ## Set Variables
        opt_tick = generate_option_tick_new(underlier, put_call, expiration, strike)
        print_url = kwargs.get('print_url', False)
        max_width = kwargs.get('max_width', 5)
        return_full = kwargs.get('return_full', False)
        expiration = pd.to_datetime(expiration).strftime('%Y-%m-%d')
        datetime = pd.to_datetime(datetime).strftime('%Y-%m-%d')

        ## Get the list of contracts
        contracts = list_contracts(underlier, datetime, print_url = print_url)  ## Query Strikes from API

        proxy_strirke = False
        contracts.expiration = pd.to_datetime(contracts.expiration, format = '%Y%m%d')
        try:
                contracts_filtered = contracts[(contracts.expiration == expiration) & (contracts.right == put_call)].sort_values('strike').reset_index(drop = True)
                idx_tgt = contracts_filtered[contracts_filtered.strike ==  strike].index[0]
        except IndexError as _i:
                contracts_filtered['spread'] = (contracts_filtered.strike - strike) **2
                idx_tgt = contracts_filtered.sort_values('spread').index[0]
                proxy_strirke = True


        interpolate_idx = range_ + idx_tgt
        contracts_filtered = contracts_filtered.iloc[interpolate_idx]
        if proxy_strirke:
                contracts_filtered = pd.concat([contracts_filtered, pd.DataFrame({'root':underlier, 'expiration': pd.to_datetime(expiration), 'right': put_call, 'strike': strike, 'spread': 0}, index = [0])]).reset_index(drop = True)
                contracts_filtered.sort_values('strike', inplace = True)

        ## Get the midpoint of the options & Close of the options
        contracts_filtered['Midpoint'] = contracts_filtered.apply(lambda x: retrieve_eod_ohlc(
                symbol = x['root'],
                start_date = datetime,
                end_date = datetime,
                strike = x['strike'],
                exp = expiration,
                right = x['right'],
                print_url = print_url
                )['Midpoint'][0], axis = 1)


        contracts_filtered['Close'] = contracts_filtered.apply(lambda x: retrieve_eod_ohlc(
                symbol = x['root'],
                start_date = datetime,
                end_date = datetime,
                strike = x['strike'],
                exp = expiration,
                right = x['right'],
                print_url = print_url
                )['Close'][0], axis = 1)
        
        ## Calculate Implied Vol for the options on both Midpoint and Close
                
        contracts_filtered['mid_vol'] = contracts_filtered.apply(lambda x: IV_handler(
                                                        price = x['Midpoint'],
                                                        S = S,
                                                        K = x['strike'],
                                                        t = time_distance_helper(exp = expiration, strt = datetime),
                                                        r = r,
                                                        q = q,
                                                        flag = x['right'].lower()), axis = 1)

        contracts_filtered['close_vol'] = contracts_filtered.apply(lambda x: IV_handler(
                                                        price = x['Close'],
                                                        S = S,
                                                        K = x['strike'],
                                                        t = time_distance_helper(exp = expiration, strt = datetime),
                                                        r = r,
                                                        q = q,
                                                        flag = x['right'].lower()), axis = 1)
        
        ## Replace the zero values with the interpolated values
        contracts_filtered.mid_vol.replace(0, np.nan, inplace = True)
        contracts_filtered['mid_vol_interpolate'] = contracts_filtered['mid_vol'].interpolate()
        tgt_strike_vol = contracts_filtered.loc[idx_tgt, 'mid_vol_interpolate']
        
        if np.isnan(tgt_strike_vol):
                ## If the target strike is still zero, then we need to interpolate further
                logger.warning(f"{opt_tick}, {datetime} strike {strike} still has zero vol. Interpolating further, width = {width}")
                print(f"{opt_tick}, {datetime} strike {strike} still has zero vol. Interpolating further, width = {width}") if print_url else None
                
                if width >= max_width:
                        ## Fit a SVI model to the data if we have reached the maximum width, and still have zero vol
                        logger.warning(f"{opt_tick}, {datetime} could not interpolate within width of {width}. Trying SVI model")
                        print(f"{opt_tick}, {datetime} could not interpolate within width of {width}. Trying SVI model") if print_url else None
                        vol, model = fit_svi_model(
                                underlier = underlier,
                                expiration = expiration,
                                datetime = datetime,
                                r = r,
                                q = q,
                                spot = S,
                                Strike = strike,
                                return_model = True,
                                print_url = print_url
                                )

                        if model.preferred_mse >= 0.0184587085747542: ## Arbitarily chosen threshold. From Poorly fitted model
                                ## Poorly fitted model shouldn't be used
                                save_thread = Thread(target = save_vol_resolve, args = (opt_tick, datetime, 'POOR_FIT'))
                                save_thread.start()
                                logger.error(f"{opt_tick}, {datetime} has a poorly fitted model. Returning zero")
                                return 0
                        if np.isnan(vol) or vol == 0:
                                ## If the model still returns zero, then we have to return zero
                                save_thread = Thread(target = save_vol_resolve, args = (opt_tick, datetime, 'UNRESOLVED'))
                                save_thread.start()
                                logger.error(f"{opt_tick}, {datetime} could not fit SVI model. Returning zero")
                                return 0
                        else:
                                ## If the model returns a value, then we return that value
                                save_thread = Thread(target = save_vol_resolve, args = (opt_tick, datetime, 'SVI_FITTING'))
                                save_thread.start()
                                return vol if not return_full else (model, vol)
                return resolve_missing_vol(
                        underlier = underlier,
                        expiration = expiration,
                        strike = strike,
                        put_call = put_call,
                        datetime = datetime,
                        S = S,
                        r = r,
                        q = q,
                        width = width + 2,
                        **kwargs
                )
        
        save_thread = Thread(target = save_vol_resolve, args = (opt_tick, datetime, 'INTERPOLATED'))
        save_thread.start()
        return (contracts_filtered.reset_index(drop = True), tgt_strike_vol) if return_full else tgt_strike_vol