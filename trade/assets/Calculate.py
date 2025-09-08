
from threading import Thread
from datetime import datetime
import datetime as dt
from trade.assets.Stock import Stock
from trade.helpers.helper import time_distance_helper, vanna, volga, identify_interval, optionPV_helper
from trade.helpers.Context import Context
from trade.assets.rates import get_risk_free_rate_helper
from typing import Union
import pandas as pd
from py_vollib.black_scholes_merton.greeks.numerical import delta, vega, theta, rho, gamma
from py_vollib.black_scholes import black_scholes as bs
from dateutil.relativedelta import relativedelta
import numpy as np
from trade.helpers.Logging import setup_logger
import logging
from typing import Callable
from pandas.tseries.offsets import BDay
from trade.helpers.types import OptionModelAttributes

logger = setup_logger('trade.assets.Calculate', stream_log_level=logging.CRITICAL)


## TODO, recalculate the pv with the new vol values
## TODO, add replace option. Either to fill close with midpoint, use only close or use only midpoint
## TODO: Add volga, vanna, dividend attribution to GB
## TODO: Streamline GB attribution columns with RV
## TODO: Find a way to return data fill for GB
## TODO: Speed up RV PnL calc with Processing
## TODO: Add logs for attribution returning zero values
## TODO: No need to calculate greeks if using RV for attribution



def PatchedCalculateFunc( func: Callable, long_leg = [], short_leg = [], return_all = False, *args, **kwargs):
    """
    
    """
    from trade.assets.Option import Option
    alllowable_func = ['pct_spot_slides', 'attribution', 'pct_vol_slides' ]
    assert hasattr(Calculate, func.__name__), f"Function {func.__name__} not a member of {Calculate}"
    assert func.__name__ in alllowable_func, f"Function {func.__name__} not allowed for this operation"
    structure_dict = {'long': [], 'short': []}
    return_both_data = False
    def get_func_values(leg, leg_name, *args, **kwargs):
        values = []
        with Context():
            for l in leg: 
                assert isinstance(l, Option) or l.__class__.__name__ == 'Option', "Leg must be an Option object"

                if leg_name == 'long':
                    values.append(func(l, *args, **kwargs))
                else:
                    values.append(-func(l, *args, **kwargs))
            
            structure_dict[leg_name] = values
    
    long_leg_thread = Thread(target=get_func_values, args=(long_leg, 'long', *args), kwargs=kwargs, name = f'{func.__name__}_long')
    short_leg_thread = Thread(target=get_func_values, args=(short_leg, 'short', *args), kwargs=kwargs, name = f'{func.__name__}_long')
    long_leg_thread.start()
    short_leg_thread.start()
    long_leg_thread.join(timeout = 3 * 60)
    short_leg_thread.join(timeout = 3 * 60)

    ## Quick fix for pct_spot_slides, Need underlier_spot_slides column to NOT be summed
    
    if func.__name__ == 'pct_spot_slides':
        try:
            columns_to_sum = [x for x in structure_dict['long'][0].columns if x != 'underlier_spot_slides']
            key = 'long'
        except:
            columns_to_sum = [x for x in structure_dict['short'][0].columns if x != 'underlier_spot_slides']
            key = 'short'

        if key == 'long':
            underlier_series = structure_dict[key][0]['underlier_spot_slides'] 
        else:
            underlier_series = -structure_dict[key][0]['underlier_spot_slides']
        structure_dict['total'] = sum(structure_dict['long']) + sum(structure_dict['short'])
        structure_dict['total']['underlier_spot_slides'] = underlier_series

    else:
        if return_both_data:
            structure_dict['total'] = sum([x[1] for x in structure_dict['long']]) + sum([x[1] for x in structure_dict['short']])
        else:
            structure_dict['total'] = sum(structure_dict['long']) + sum(structure_dict['short'])



    if return_all:
        return structure_dict
    else:
        return structure_dict['total']





class Calculate:
    rf_rate = None  # Initializing risk free rate
    rf_ts = None
    init_date = None
    
    def __init__(self):
        """
        Calculate Class used for calculating Asset related items
        
        Static Class for handling calculations in Stock & Option Classes
        """
        
        today = datetime.today()
        start_date_date = today - relativedelta(months=6)
        start_date = datetime.strftime(start_date_date, format='%Y-%m-%d')
        end_date = datetime.strftime(today, format='%Y-%m-%d')

           ##Initializing Risk Free Rate as a variable across every Stock Intance, while Re-initializing after day has changed
        if Calculate.rf_rate is None:
            print("Calculate Initializing Risk Free Rate")
            self.init_rfrate_ts()
            self.init_date_method()
            self.init_risk_free_rate()
            if Calculate.init_date < today:
                print("Re-initializing Risk Rate")
                self.init_rfrate_ts()
                self.init_risk_free_rate()
                self.init_date_method()


    @classmethod
    def init_rfrate_ts(cls):
        cls.rf_ts = get_risk_free_rate_helper()

    @classmethod
    def init_risk_free_rate(cls):
        ts = cls.rf_ts
        cls.rf_rate = ts.iloc[len(ts)-1, 0]/100

    @classmethod
    def init_date_method(cls):
        cls.init_date = datetime.today()


    @staticmethod
    def pv(
           asset = None,
           K: Union[int, float] = None, 
           exp_date: str = None, 
           sigma: float = None, 
           S0: Union[int, float, None] = None,
           put_call=None,
           N: int = 100, 
           r: float = None,
           y: float = None, 
           start: str = None,
           model: str = None) -> float:
        '''
    Returns the price of an american option

    Parameters:
        model:
        K: Strike price
        exp_date: Expiration date
        S0: Spot at current time (Optional)
        r: Risk free rate (Optional). If no value passed, defaults to most recent risk free rate.
        N: Number of steps to use in the calculation (Optional)
        y: Dividend yield (Optional)
        Sigma: Implied Volatility of the option
        opttype: Option type ie put or call (Defaults to "P")
        start: Start date of the pricing model. If nothing is passed, defaults to today. If initiated within a context and nothing is passed, defaults to context start date (Optional)
        '''
        from trade.assets.Option import Option
        if asset is None:
            if all(arg is None for arg in (K, exp_date, sigma, S0, put_call, N, r, y, start)):
                raise Exception("Missing Assets & Missing params")
            else:
                return optionPV_helper(S0, K, exp_date, r, y, sigma, put_call, start)
        else:
            if asset.__class__.__name__ == 'Stock':
                raise Exception("Stock can't be priced with options model. This method isn't utilized for a Stock Instance")
            
            if isinstance(asset, Option):
                    if K is None:
                        K = asset.K
                    if exp_date is None:
                        exp_date = asset.exp
                    if sigma is None:
                        sigma = asset.sigma
                    if S0 is None:
                        S0 = asset.unadjusted_S0
                    if y is None:
                        y = asset.y
                    if put_call is None:
                        put_call = asset.put_call
                    if r is None:
                        r = asset.rf_rate
                    if start is None:
                        start = asset.end_date
                    return optionPV_helper(S0, K, exp_date, r, y, sigma, put_call, start)

    @staticmethod
    def pct_spot_slides(asset = None, pct_spot=[0.8, 0.9, 0.95, 1, 1.05, 1.1, 1.2], greeks_to_calc = ['delta', 'gamma'], asset_type = None, **kwargs):
        """
        Calculates Slide scenario based on provided lists of slides
        This assumes a percent drop in Spot

        Returns a dataframe containing both PnL and price post shock

        Parameters:
        pct_spot = An iterable with price shocks
        greeks_to_calc = Lists of greeks to return values
            Available greeks: delta, gamma, vega, rho, theta, vanna, volga
        asset_type = Use only when calculating spot without a base asset. Only applicable to options
        **kwargs = Keyword arguments as seen in Calculate.PV()
                For asset_type == 'stock' kwargs is S0
        """
        from trade.assets.Option import Option
        from trade.assets.OptionStructure import OptionStructure
        if asset is None:
            assert asset_type is not None, f'asset_type needed'
            assert asset_type.lower() in ['stock', 'option'], f'Invalid asset_type, expected "stock" or "option", recieved "{asset_type}"'
            if asset_type.lower() == 'stock':
                if kwargs:
                    s0 = kwargs.pop('S0')
                    slides = Calculate.scenario_helper('spot', pct_spot,S0 =  s0, modelType = 'stock')
                    return slides
                else:
                    raise TypeError('Expected S0 for stock, none given')
            else:
                if kwargs:
                    k = kwargs['K']
                    exp_date = kwargs['exp_date']
                    sigma = kwargs['sigma']
                    s0 = kwargs['S0']
                    put_call = kwargs['put_call']
                    r = kwargs['r']
                    y = kwargs['y']
                    start = kwargs['start']
                    modelType = 'option'
                    return Calculate.scenario_helper('spot', pct_spot, K = k,
                                S0=s0, exp_date=exp_date, sigma = sigma, y = y, put_call=put_call,
                                r = r, start = start, modelType = modelType, greeks_to_calc = greeks_to_calc).set_index('shocks')
                else:
                    raise TypeError('Expected kwargs for Calculate.PV. None was recieved.')

        
        elif asset.__class__.__name__ == 'Stock':
            s0 = list(asset.spot().values())[-1]
            slides = Calculate.scenario_helper('spot', pct_spot,S0 =  s0, modelType = 'stock').set_index('shocks')
            modelType = 'stock'
            return slides
        
        elif asset.__class__.__name__ == 'Option':

            k = getattr(asset, OptionModelAttributes.K.value)
            exp_date = getattr(asset, OptionModelAttributes.exp_date.value)
            sigma = getattr(asset, OptionModelAttributes.sigma.value)
            s0 = getattr(asset, OptionModelAttributes.S0.value)
            y = getattr(asset, OptionModelAttributes.y.value)
            put_call = getattr(asset, OptionModelAttributes.put_call.value)
            r = getattr(asset, OptionModelAttributes.r.value)
            start = getattr(asset, OptionModelAttributes.start.value)
            modelType = 'option'
            return Calculate.scenario_helper('spot', pct_spot, K = k,
                            S0=s0, exp_date=exp_date, sigma = sigma, y = y, put_call=put_call,
                            r = r, start = start, modelType = modelType, greeks_to_calc = greeks_to_calc).set_index('shocks')
        
        elif asset.__class__.__name__ == 'OptionStructure':
            from trade.assets.Option import Option
            return_all = kwargs.get('return_all', False)
            return PatchedCalculateFunc(Calculate.pct_spot_slides, 
                                        long_leg = asset.long, 
                                        short_leg = asset.short, 
                                        return_all = return_all, 
                                        pct_spot = pct_spot, 
                                        greeks_to_calc = greeks_to_calc)
            
            
    @staticmethod
    def pct_vol_slides(asset = None, pct_spot=[-0.05, -0.02, -0.01, 0, +0.01, +0.02, +0.05], greeks_to_calc = ['delta', 'gamma'], asset_type = None, **kwargs):
        """
        Calculates Slide scenario based on provided lists of vol slides.
        These assumes a drop in points 

        Returns a dataframe containing both PnL and price post shock

        Parameters:
        pct_spot = An iterable with price shocks
        asset_type = Use only when calculating spot without a base asset
        **kwargs = Keyword arguments as seen in Calculate.PV()
                For asset_type == 'stock' kwargs is S0
        """
        from trade.assets.Option import Option
        from trade.assets.OptionStructure import OptionStructure

        if asset is None:
            assert asset_type is not None, f'asset_type needed'
            assert asset_type.lower() in ['stock', 'option'], f'Invalid asset_type, expected "stock" or "option", recieved "{asset_type}"'
            if asset_type.lower() == 'stock':
                raise TypeError('Cannot Calculate vol shocks for Stock')
            else:
                if kwargs:
                    k = kwargs['K']
                    exp_date = kwargs['exp_date']
                    sigma = kwargs['sigma']
                    s0 = kwargs['S0']
                    put_call = kwargs['put_call']
                    r = kwargs['r']
                    y = kwargs['y']
                    start = kwargs['start']
                    modelType = 'option'
                    return Calculate.scenario_helper('vol', pct_spot, K = k,
                                S0=s0, exp_date=exp_date, sigma = sigma, y = y, put_call=put_call,
                                r = r, start = start, modelType = modelType, greeks_to_calc= greeks_to_calc).set_index('shocks')
                else:
                    raise TypeError('Expected kwargs for Calculate.PV. None was recieved.')
                    
        elif isinstance(asset, Stock):
            raise TypeError('Cannot Calculate vol shocks for Stock')
        elif isinstance(asset, Option) or asset.__class__.__name__ == 'Option':
            k = getattr(asset, OptionModelAttributes.K.value)
            exp_date = getattr(asset, OptionModelAttributes.exp_date.value)
            sigma = getattr(asset, OptionModelAttributes.sigma.value)
            s0 = getattr(asset, OptionModelAttributes.S0.value)
            y = getattr(asset, OptionModelAttributes.y.value)
            put_call = getattr(asset, OptionModelAttributes.put_call.value)
            r = getattr(asset, OptionModelAttributes.r.value)
            start = getattr(asset, OptionModelAttributes.start.value)
            modelType = 'option'
            return Calculate.scenario_helper('vol', pct_spot, K = k,
                            S0=s0, exp_date=exp_date, sigma = sigma, y = y, put_call=put_call,
                            r = r, start = start, modelType = modelType, greeks_to_calc = greeks_to_calc).set_index('shocks')

        elif isinstance(asset, OptionStructure) or asset.__class__.__name__ == 'OptionStructure':
            return_all = kwargs.get('return_all', False)
            return PatchedCalculateFunc(Calculate.pct_vol_slides, 
                                        long_leg = asset.long, 
                                        short_leg = asset.short, 
                                        return_all = return_all, 
                                        pct_spot = pct_spot, 
                                        greeks_to_calc = greeks_to_calc)
        else:
            raise TypeError(f'Invalid Asset Type. Recieved {asset}')

    @staticmethod
    def spot_vol_grid(asset = None, 
                      vol_pct=[-0.05, -0.02, -0.01, 0, +0.01, +0.02, +0.05], 
                      spot_pct=[0.8, 0.9, 0.95, 1, 1.05, 1.1, 1.2],
                      greek_enum = 'pv',
                      **kwargs):
        """
        Calculates a grid of spot and vol shocks
        Returns a dataframe containing both PnL and price post shock
        Parameters:
        vol_pct_spot = An iterable with vol shocks
        spot_pct_spot = An iterable with price shocks
        **kwargs = Keyword arguments as seen in Calculate.PV()
                For asset_type == 'stock' kwargs is S0
                    
        """

        if asset is None:
            k = kwargs['K']
            exp_date = kwargs['exp_date']
            sigma = kwargs['sigma']
            s0 = kwargs['S0']
            put_call = kwargs['put_call']
            r = kwargs['r']
            y = kwargs['y']
            start = kwargs['start']
            modelType = 'option'
        else:
            assert asset.__class__.__name__ == 'Option', f'Invalid asset type, expected Option, recieved {asset.__class__.__name__}'
            k = getattr(asset, OptionModelAttributes.K.value)
            exp_date = getattr(asset, OptionModelAttributes.exp_date.value)
            sigma = getattr(asset, OptionModelAttributes.sigma.value)
            s0 = getattr(asset, OptionModelAttributes.S0.value)
            y = getattr(asset, OptionModelAttributes.y.value)
            put_call = getattr(asset, OptionModelAttributes.put_call.value)
            r = getattr(asset, OptionModelAttributes.r.value)
            start = getattr(asset, OptionModelAttributes.start.value)

        def spot_vol_helper(spot_shock, vol_shock, greek_enum):
            if greek_enum not in ['pv', 'delta', 'gamma', 'vega', 'rho', 'theta', 'vanna', 'volga','pnl']:
                raise TypeError(f'Invalid greek_enum, expected one of ["pv", "delta", "gamma", "vega", "rho", "theta", "vanna", "volga"], recieved {greek_enum}')
            

            if greek_enum == 'pnl':
                fn = Calculate.pv
            else:
                fn = getattr(Calculate, greek_enum)
            shocked_spot = spot_shock * s0
            shocked_vol = sigma + vol_shock

            ## Calculate shocked value
            if greek_enum in ['pv', 'pnl']:
                shocked_value = fn(K = k, exp_date = exp_date, sigma = shocked_vol, S0 = shocked_spot,
                                            put_call = put_call, r =r, y = y, start = start)
            else:
                shocked_value = fn(K = k, exp = exp_date, sigma = shocked_vol, S = shocked_spot,
                                    flag = put_call, r =r,start = start, y = y)
            
            ## Calculate base value for Pnl and set value to shocked - base
            if greek_enum == 'pnl':
                pv = fn(K = k, exp_date = exp_date, sigma = sigma, S0 = s0,
                                        put_call = put_call, r =r, y = y, start = start)
                value = shocked_value - pv

            ## Else just return shocked value
            else: 
                value = shocked_value
            return value
        
        
        ## Create dataframe
        spv = pd.DataFrame(index = sorted(vol_pct), columns = sorted(spot_pct))
        for v in vol_pct:
            for s in spot_pct:
                spv.loc[v, s] = spot_vol_helper(s, v, greek_enum)
        spv.index.name = 'vol_shock'
        spv.columns.name = 'spot_shock'
        return spv
                                
            
    @staticmethod
    def scenario_helper(type_, pct_spot, **kwargs):
        """
        scenario_helper should return a dataframe with shocks as index
        This is a multi-purpose scenario calculator. It calculates the scenario for different shocks
        """
        modelType = kwargs['modelType']
        if modelType == 'option':
            k = kwargs['K']
            exp_date = kwargs['exp_date']
            sigma = kwargs['sigma']
            s0 = kwargs['S0']
            put_call = kwargs['put_call']
            r = kwargs['r']
            y = kwargs['y']
            start = kwargs['start']
            greeks_to_calc = kwargs['greeks_to_calc']
            
        else: 
            s0 = kwargs['S0']
            greeks_to_calc = None
        if type_ == 'spot':
            pct_spot.append(1) if 1 not in pct_spot else None
            scen = pd.DataFrame(index = [x for x in range(len(pct_spot))], data = {'shocks':pct_spot})
            
            if modelType =='option':
                scen['underlier_spot_slides'] = scen.apply(lambda x:(x['shocks'] - 1)*s0, axis = 1)
                scen['shocked_pv'] = scen.apply(lambda x:Calculate.pv(K = k, exp_date = exp_date, sigma = sigma, S0 = x['shocks']*s0,
                                                                        put_call = put_call, r =r, y = y, start = start),axis = 1)
                scen['pv'] = scen.apply(lambda x:Calculate.pv(K = k, exp_date = exp_date, sigma = sigma, S0 = s0,
                put_call = put_call, r =r, y = y, start = start),axis = 1)
                scen['pnl'] = scen['shocked_pv'] - scen['pv']
                if greeks_to_calc:
                    for greek in greeks_to_calc:
                        fn = getattr(Calculate, greek)
                        scen[greek] = scen.apply(lambda x:fn(K = k, exp = exp_date, sigma = sigma, S = s0*x['shocks'],
                            flag = put_call, r =r,start = start, y = y),axis = 1)

                scen.sort_values('shocks', inplace = True)
                return scen
            elif modelType =='stock':
                scen['shocked_pv'] = scen.apply(lambda x: s0 * x['shocks'], axis = 1)
                scen['pv'] = s0
                scen['pnl'] = scen['shocked_pv'] - scen['pv']
                scen.sort_values('shocks', inplace = True)
                return scen
        if type_ == 'vol':
            pct_spot.append(0.0) if 0.0 not in pct_spot else None
            scen = pd.DataFrame(index = [x for x in range(len(pct_spot))], data = {'shocks':pct_spot})
            
            if modelType =='option':
                scen['shocked_pv'] = scen.apply(lambda x:Calculate.pv(K = k, exp_date = exp_date, sigma = sigma + x['shocks'], S0 = s0,
                                                                    put_call = put_call, r =r, y = y, start = start),axis = 1)
                scen['pv'] = scen.apply(lambda x:Calculate.pv(K = k, exp_date = exp_date, sigma = sigma, S0 = s0,
                put_call = put_call, r =r, y = y, start = start),axis = 1)
                scen['pnl'] = scen['shocked_pv'] - scen['pv']
                if greeks_to_calc:
                    for greek in greeks_to_calc:
                        greek = greek.lower()
                        fn = getattr(Calculate, greek)
                        scen[greek] = scen.apply(lambda x:fn(K = k, exp = exp_date, sigma = sigma+ x['shocks'], S = s0,
                            flag = put_call, r =r,start = start, y = y),axis = 1)

                scen.sort_values('shocks', inplace = True)
                return scen

                




    @staticmethod
    def delta(asset = None, S = None, K = None, r = None, sigma = None, start = None, flag = None, exp = None, y = None, model = 'bs'):
        
        """
        Returns the Delta of an option
        """
        from trade.assets.Option import Option
        if isinstance(asset, Option) or asset.__class__.__name__ == 'Option':
            args = [S, K, r, sigma, y, model]
            args_str = [OptionModelAttributes.S0.value, 
                        OptionModelAttributes.K.value, 
                        OptionModelAttributes.r.value, 
                        OptionModelAttributes.sigma.value, 
                        OptionModelAttributes.y.value,
                        'model']
            for i in range(len(args)):
                if args[i] is None:
                    args[i] = getattr(asset, args_str[i])
            
            t = time_distance_helper(asset.exp, asset.end_date)
            flag = getattr(asset, OptionModelAttributes.put_call.value)
            if model == 'bs':
                d = delta(flag = flag.lower(),S = args[0], K = args[1], t = t, r = args[2], sigma = args[3], q = args[4] )
            elif model == 'binomial':
                d = delta(flag = flag.lower(),S = args[0], K = args[1], t = t, r = args[2], sigma = args[3], q = args[4] )
            return d

        elif asset == None:
            assert all(v is not None for v in [S, K, r, sigma, start, flag, exp, y]), f"None of y, S, K, r, sigma, start, flag, exp, can be None"
            if sigma == 0:
                logger.error("Sigma cannot be 0")
                logger.error(f"Kwargs: {locals()}")
                return 0.0
            
            if sigma == 0:
                raise ValueError("Sigma cannot be 0")
            t = time_distance_helper(exp, start)
            if model == 'bs':
                d = delta(flag = flag.lower(), S = S, K = K, t = t, r = r, sigma = sigma, q = y )
            elif model == 'binomial':
                d = delta(flag = flag.lower(), S = S, K = K, t = t, r = r, sigma = sigma, q = y )
            elif model == 'mcs':
                raise NotImplementedError("Monte Carlo Simulation not implemented yet")
            else:
                raise ValueError(f"Invalid Model Type, recieved {model}, expected 'bs', 'binomial' or 'mcs'")
            d = float(d)
            return float(d)
        else:
            raise Exception(f"Delta cannot be Calculated for {asset} type")


    @staticmethod
    def vega(asset = None, S = None, K = None, r = None, sigma = None, start = None, flag = None, exp = None, y = None, model = 'bs'):
        
        """
        Returns the Vega of an option
        """
        from trade.assets.Option import Option
        if isinstance(asset, Option) or asset.__class__.__name__ == 'Option':
            args = [S, K, r, sigma, y, model]
            args_str = [OptionModelAttributes.S0.value, 
                        OptionModelAttributes.K.value, 
                        OptionModelAttributes.r.value, 
                        OptionModelAttributes.sigma.value, 
                        OptionModelAttributes.y.value,
                        'model']
            for i in range(len(args)):
                if args[i] is None:
                    args[i] = getattr(asset, args_str[i])
            

            t = time_distance_helper(asset.exp, asset.end_date)
            flag = getattr(asset, OptionModelAttributes.put_call.value)
            if model == 'bs':
                d = vega(flag = flag.lower(),S = args[0], K = args[1], t = t, r = args[2], sigma = args[3], q = args[4] )
            elif model == 'binomial':
                d = vega(flag = flag.lower(),S = args[0], K = args[1], t = t, r = args[2], sigma = args[3], q = args[4] )
            return d

        elif asset == None:
            assert all(v is not None for v in [S, K, r, sigma, start, flag, exp, y]), f"None of y, S, K, r, sigma, start, flag, exp, can be None"
            t = time_distance_helper(exp, start)
            if model == 'bs':
                d = vega(flag = flag.lower(), S = S, K = K, t = t, r = r, sigma = sigma, q = y )
            elif model == 'binomial':
                d = vega(flag = flag.lower(), S = S, K = K, t = t, r = r, sigma = sigma, q = y )
            elif model == 'mcs':
                raise NotImplementedError("Monte Carlo Simulation not implemented yet")
            else:
                raise ValueError(f"Invalid Model Type, recieved {model}, expected 'bs', 'binomial' or 'mcs'")
            
            d = float(d)
            return float(d)
        else:
            raise Exception(f"Vega cannot be Calculated for {asset} type")



    @staticmethod
    def vanna(asset = None, S = None, K = None, r = None, sigma = None, start = None, flag = None, exp = None, y = None, model = 'bs'):
        
        """
        Returns the vanna of an option
        """
        from trade.assets.Option import Option
        if isinstance(asset, Option) or asset.__class__.__name__ == 'Option':
            args = [S, K, r, sigma, y, model]
            args_str = [OptionModelAttributes.S0.value, 
                        OptionModelAttributes.K.value, 
                        OptionModelAttributes.r.value, 
                        OptionModelAttributes.sigma.value, 
                        OptionModelAttributes.y.value,
                        'model']
            for i in range(len(args)):
                if args[i] is None:
                    args[i] = getattr(asset, args_str[i])
 
            t = time_distance_helper(asset.exp, asset.end_date)
            flag = getattr(asset, OptionModelAttributes.put_call.value)
            if model == 'bs':
                d = vanna(flag = flag.lower(),S = args[0], K = args[1], T = t, r = args[2], sigma = args[3], q = args[4] )
            elif model == 'binomial':
                d = vanna(flag = flag.lower(),S = args[0], K = args[1], T = t, r = args[2], sigma = args[3], q = args[4] )
            elif model == 'mcs':
                raise NotImplementedError("Monte Carlo Simulation not implemented yet")
            else:
                raise ValueError(f"Invalid Model Type, recieved {model}, expected 'bs', 'binomial' or 'mcs'")
            return d

        elif asset == None:
            assert all(v is not None for v in [S, K, r, sigma, start, flag, exp, y]), f"None of y, S, K, r, sigma, start, flag, exp, can be None"
            t = time_distance_helper(exp, start)
            if model == 'bs':
                d = vanna(flag = flag.lower(), S = S, K = K, T = t, r = r, sigma = sigma, q = y )
            elif model == 'binomial':
                d = vanna(flag = flag.lower(), S = S, K = K, T = t, r = r, sigma = sigma, q = y )
            elif model == 'mcs':
                raise NotImplementedError("Monte Carlo Simulation not implemented yet")
            else:
                raise ValueError(f"Invalid Model Type, recieved {model}, expected 'bs', 'binomial' or 'mcs'")
            d = float(d)
            return float(d)
        else:
            raise Exception(f"Vanna cannot be Calculated for {asset} type")

    @staticmethod
    def volga(asset = None, S = None, K = None, r = None, sigma = None, start = None, flag = None, exp = None, y = None, model = 'bs'):
        
        """
        Returns the volga of an option
        """
        from trade.assets.Option import Option
        if isinstance(asset, Option) or asset.__class__.__name__ == 'Option':
            args = [S, K, r, sigma, y, model]
            args_str = [OptionModelAttributes.S0.value, 
                        OptionModelAttributes.K.value, 
                        OptionModelAttributes.r.value, 
                        OptionModelAttributes.sigma.value, 
                        OptionModelAttributes.y.value,
                        'model']
            for i in range(len(args)):
                if args[i] is None:
                    args[i] = getattr(asset, args_str[i])
            
    
            t = time_distance_helper(asset.exp, asset.end_date)
            flag = getattr(asset, OptionModelAttributes.put_call.value)
            if model == 'bs':
                d = volga(flag = flag.lower(),S = args[0], K = args[1], T = t, r = args[2], sigma = args[3], q = args[4] )
            elif model == 'binomial':
                d = volga(flag = flag.lower(),S = args[0], K = args[1], T = t, r = args[2], sigma = args[3], q = args[4] )
            return d

        elif asset == None:
            assert all(v is not None for v in [S, K, r, sigma, start, flag, exp, y]), f"None of y, S, K, r, sigma, start, flag, exp, can be None"
            t = time_distance_helper(exp, start)
            if model == 'bs':
                d = volga(flag = flag.lower(), S = S, K = K, T = t, r = r, sigma = sigma, q = y )
            elif model == 'binomial':
                d = volga(flag = flag.lower(), S = S, K = K, T = t, r = r, sigma = sigma, q = y )
            elif model == 'mcs':
                raise NotImplementedError("Monte Carlo Simulation not implemented yet")
            else:
                raise ValueError(f"Invalid Model Type, recieved {model}, expected 'bs', 'binomial' or 'mcs'")
            d = float(d)
            return float(d)
        else:
            raise Exception(f"Volga cannot be Calculated for {asset} type")

    @staticmethod
    def gamma(asset = None, S = None, K = None, r = None, sigma = None, start = None, flag = None, exp = None, y = None, model = 'bs'):
        
        """
        Returns the gamma of an option
        """
        from trade.assets.Option import Option
        if isinstance(asset, Option) or asset.__class__.__name__ == 'Option':
            args = [S, K, r, sigma, y, model]
            args_str = [OptionModelAttributes.S0.value, 
                        OptionModelAttributes.K.value, 
                        OptionModelAttributes.r.value, 
                        OptionModelAttributes.sigma.value, 
                        OptionModelAttributes.y.value,
                        'model']
            for i in range(len(args)):
                if args[i] is None:
                    args[i] = getattr(asset, args_str[i])
            
            t = time_distance_helper(asset.exp, asset.end_date)
            if model == 'bs':
                d = gamma(flag = flag.lower(),S = args[0], K = args[1], t = t, r = args[2], sigma = args[3], q = args[4] )
            elif model == 'binomial':
                d = gamma(flag = flag.lower(),S = args[0], K = args[1], t = t, r = args[2], sigma = args[3], q = args[4] )
            return d

        elif asset == None:
            assert all(v is not None for v in [S, K, r, sigma, start, flag, exp, y]), f"None of y, S, K, r, sigma, start, flag, exp, can be None"
            if sigma == 0:
                logger.error("Sigma cannot be 0")
                logger.error(f"Kwargs: {locals()}")
                return 0.0
            
            t = time_distance_helper(exp, start)
            if model == 'bs':
                d = gamma(flag = flag.lower(), S = S, K = K, t = t, r = r, sigma = sigma, q = y )
            elif model == 'binomial':
                d = gamma(flag = flag.lower(), S = S, K = K, t = t, r = r, sigma = sigma, q = y )
            elif model == 'mcs':
                raise NotImplementedError("Monte Carlo Simulation not implemented yet")
            else:
                raise ValueError(f"Invalid Model Type, recieved {model}, expected 'bs', 'binomial' or 'mcs'")

            d = float(d)
            return float(d)
        else:
            raise Exception(f"Gamma cannot be Calculated for {asset} type")
    
    @staticmethod
    def theta(asset = None, S = None, K = None, r = None, sigma = None, start = None, flag = None, exp = None, y = None, model = 'bs'):
        
        """
        Returns the theta of an option
        """
        from trade.assets.Option import Option
        if model =='mcs':
            raise NotImplementedError("Monte Carlo Simulation not implemented yet")
        elif model not in ['bs', 'binomial']:
            raise ValueError(f"Invalid Model Type, recieved {model}, expected 'bs', 'binomial' or 'mcs'")
        
        if isinstance(asset, Option) or asset.__class__.__name__ == 'Option':
            args = [S, K, r, sigma, y, model]
            args_str = [OptionModelAttributes.S0.value, 
                        OptionModelAttributes.K.value, 
                        OptionModelAttributes.r.value, 
                        OptionModelAttributes.sigma.value, 
                        OptionModelAttributes.y.value,
                        'model']
            for i in range(len(args)):
                if args[i] is None:
                    args[i] = getattr(asset, args_str[i])
            t = time_distance_helper(asset.exp, asset.end_date)
            if model == 'bs':
                d = theta(flag = flag.lower(),S = args[0], K = args[1], t = t, r = args[2], sigma = args[3], q = args[4] )
            elif model == 'binomial':
                d = theta(flag = flag.lower(),S = args[0], K = args[1], t = t, r = args[2], sigma = args[3], q = args[4] )
            return d

        elif asset == None:
            assert all(v is not None for v in [S, K, r, sigma, start, flag, exp, y]), f"None of y, S, K, r, sigma, start, flag, exp, can be None"
            t = time_distance_helper(exp, start)
            if model == 'bs':
                d = theta(flag = flag.lower(), S = S, K = K, t = t, r = r, sigma = sigma, q = y )
            elif model == 'binomial':
                d = theta(flag = flag.lower(), S = S, K = K, t = t, r = r, sigma = sigma, q = y )
            d = float(d)
            return float(d)
        else:
            raise Exception(f"Theta cannot be Calculated for {asset} type")


    @staticmethod
    def rho(asset = None, S = None, K = None, r = None, sigma = None, start = None, flag = None, exp = None, y = None, model = 'bs'):
        
        """
        Returns the rho of an option
        """
        from trade.assets.Option import Option
        if model =='mcs':
            raise NotImplementedError("Monte Carlo Simulation not implemented yet")
        elif model not in ['bs', 'binomial']:
            raise ValueError(f"Invalid Model Type, recieved {model}, expected 'bs', 'binomial' or 'mcs'")
        
        if isinstance(asset, Option) or asset.__class__.__name__ == 'Option':
            args = [S, K, r, sigma, y, model]
            args_str = [OptionModelAttributes.S0.value, 
                        OptionModelAttributes.K.value, 
                        OptionModelAttributes.r.value, 
                        OptionModelAttributes.sigma.value, 
                        OptionModelAttributes.y.value,
                        'model']
            for i in range(len(args)):
                if args[i] is None:
                    args[i] = getattr(asset, args_str[i])
            t = time_distance_helper(asset.exp, asset.end_date)
            if model == 'bs':
                d = rho(flag = flag.lower(),S = args[0], K = args[1], t = t, r = args[2], sigma = args[3], q = args[4] )
            elif model == 'binomial':
                d = rho(flag = flag.lower(),S = args[0], K = args[1], t = t, r = args[2], sigma = args[3], q = args[4] )
            return d

        elif asset == None:
            assert all(v is not None for v in [S, K, r, sigma, start, flag, exp, y]), f"None of y, S, K, r, sigma, start, flag, exp, can be None"
            t = time_distance_helper(exp, start)
            if model == 'bs':
                d = rho(flag = flag.lower(), S = S, K = K, t = t, r = r, sigma = sigma, q = y )
            elif model == 'binomial':
                d = rho(flag = flag.lower(), S = S, K = K, t = t, r = r, sigma = sigma, q = y )
            d = float(d)
            return float(d)
        else:
            raise Exception(f"Rho cannot be Calculated for {asset} type")

    @staticmethod
    def greeks(asset = None, S = None, K = None, r = None, sigma = None, start = None, flag = None, exp = None, y = None, model = 'bs'):

        """
        Returns all the greeks of an option as dictionary
        """
        from trade.assets.Option import Option
        kwargs = locals()
        if model =='mcs':
            raise NotImplementedError("Monte Carlo Simulation not implemented yet")
        elif model not in ['bs', 'binomial']:
            raise ValueError(f"Invalid Model Type, recieved {model}, expected 'bs', 'binomial' or 'mcs'")
        if isinstance(asset, Option) or asset.__class__.__name__ == 'Option':
            try:
                greeks = {'Delta': Calculate.delta(asset, S =S, K = K, r = r, sigma = sigma, start = start, model = model),
                'Gamma': Calculate.gamma(asset, S =S, K = K, r = r, sigma = sigma, start = start, model = model),
                'Vega': Calculate.vega(asset, S =S, K = K, r = r, sigma = sigma, start = start, model = model),
                'Theta': Calculate.theta(asset, S =S, K = K, r = r, sigma = sigma, start = start, model = model) if Calculate.theta(asset, S =S, K = K, r = r, sigma = sigma, start = start, model = model) is not None else 0,
                'Rho': Calculate.rho(asset, S =S, K = K, r = r, sigma = sigma, start = start, model = model),
                'Vanna':Calculate.vanna(asset, S =S, K = K, r = r, sigma = sigma, start = start, model = model),
                'Volga':Calculate.volga(asset, S =S, K = K, r = r, sigma = sigma, start = start, model = model)
                }
                return greeks
                
            except Exception as e:
                print(e)
                return {'Delta': 0.0, 'Gamma': 0.0, 'Vega': 0.0, 'Theta': 0.0, 'Rho': 0.0, 'Vanna': 0.0, 'Volga': 0.0}
            
        elif asset == None:
            try:
                greeks = {'Delta': Calculate.delta(S =S, K = K, r = r, sigma = sigma, start = start, flag = flag, exp = exp, y = y, model=model),
                'Gamma': Calculate.gamma(S =S, K = K, r = r, sigma = sigma, start = start, flag = flag, exp = exp, y = y, model=model),
                'Vega': Calculate.vega(S =S, K = K, r = r, sigma = sigma, start = start, flag = flag, exp = exp, y = y, model=model),
                'Theta': Calculate.theta(S =S, K = K, r = r, sigma = sigma, start = start, flag = flag, exp = exp, y = y, model=model) if Calculate.theta(S =S, K = K, r = r, sigma = sigma, start = start, flag = flag, exp = exp, y = y, model=model) is not None else 0,
                'Rho': Calculate.rho(S =S, K = K, r = r, sigma = sigma, start = start, flag = flag, exp = exp, y = y, model=model),
                'Vanna':Calculate.vanna(S =S, K = K, r = r, sigma = sigma, start = start, flag = flag, exp = exp, y = y, model=model),
                'Volga':Calculate.volga(S =S, K = K, r = r, sigma = sigma, start = start, flag = flag, exp = exp, y = y, model=model)
                }
                return greeks
            except Exception as e:
                
                logger.info('')
                logger.info('Calculate.greeks raised this error')
                logger.info(e,exc_info=True)
                logger.info(f'Kwargs:{kwargs}')
                if isinstance(e, (AssertionError, NotImplementedError)):
                    raise e
                return {'Delta': 0, 'Gamma': 0, 'Vega': 0, 'Theta': 0, 'Rho': 0, 'Vanna': 0, 'Volga':0}
        else:
            raise Exception(
                f"Greeks cannot be Calculated for {asset} type")
    
    @staticmethod
    def attribution(asset,
                    ts_start = None,
                    ts_end= None,
                    ts_timeframe = 'day',
                    ts_timewidth = '1',
                     method = "GB",
                     replace = 'partial',
                     return_both_data = False,
                     **kwargs):
        
        ## To do, add replace option. Either to fill close with midpoint, use only close or use only midpoint
        """
        Calculate attribution of option asset 

        Parameter:
        ____________
        ts_start (str | Datetime): Start date if timeseries
        ts_end (str | Datetime): End date if timeseries  
        ts_timewidth (int): Examples 1,2,3,4. The span over the timeframe
        ts_timeframe (str): The timeframe for aggregation, eg: Minute, Hour, Day, Month, Week, Year
        method (str): Available methods are 'GB' for Greek Based and 'RV' for Revaluation
        replace (str): Available options are 'partial', 'close', 'default_fill'. Partial replaces only the missing data, Close uses close data to fill, default_fill uses the default fill for all data
        return_both_data (bool): If True. Will return both the PnL Data and Full Data
        return_all: specific to OptionStructure. If True, will return all the data for the long and short leg
        
        """
        from trade.assets.Option import Option
        from trade.assets.OptionStructure import OptionStructure

        #GET OPTION TIMESERIES
        today = datetime.today()
        ## Dates to allow ffill
        ts_start = ts_start if ts_start else asset.start_date
        ts_end = ts_end if ts_end else asset.end_date
        start = pd.to_datetime(ts_start) - BDay(2)
        end = pd.to_datetime(ts_end) + BDay(2)
        if asset.__class__.__name__ == 'Option':
    
            ## Designate the columns to be used
            vol_col = ['Bs_iv' if asset.model == 'bsm' else 'Binomial_iv']
            if asset.default_fill:
                vol_col.append(f"{asset.default_fill.capitalize()}_{vol_col[0].lower()}")
            spot_col = ['Close', asset.default_fill.capitalize()]
            spot_col_2 = ['Option_Close', asset.default_fill.capitalize()]
            greeks_col = ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'Vanna', 'Volga']

            ## GET OPTION TIMESERIES
            spot_ts = asset.spot(ts_start= start, 
                                ts_end = end,
                                ts_timeframe= ts_timeframe,
                                ts_timewidth= ts_timewidth,
                                ts = True)[spot_col]
            
            ## Create mask for missing data
            full_data = spot_ts.copy()
            if replace == 'partial':
                close_fill_mask = full_data['Close'] == 0

            elif replace == 'close':
                close_fill_mask = pd.Series([False]*len(full_data), index = full_data.index)

            elif replace == 'default_fill':
                close_fill_mask = pd.Series([True]*len(full_data), index = full_data.index)
            else:
                raise Exception(f"Invalid replace option. Expected 'partial', 'close', 'default_fill', recieved {replace}")

            ## Fill missing data + Rename columns to Option_Close
            full_data.rename(columns = {'Close': 'Option_Close'}, inplace = True)
            full_data.loc[close_fill_mask, 'Option_Close'] = full_data.loc[close_fill_mask, asset.default_fill.capitalize()]


            ## Get Vol Timeseries
            vol = asset.vol(ts_start= start, 
                                ts_end = end,
                                ts_timeframe= ts_timeframe,
                                ts_timewidth= ts_timewidth,
                                ts = True)[vol_col]
            full_data[vol_col] = vol
            ## Fill missing Vol data
            full_data.loc[close_fill_mask, vol_col[0]] = full_data.loc[close_fill_mask, vol_col[1]]

            ## Fixing the vol values that are 0 with a fill forward

            ## TODO, recalculate the pv with the new vol values
            full_data[vol_col] = full_data[vol_col].replace(0.0, np.nan)
            full_data[vol_col] = full_data[vol_col].ffill()

            # # GET STOCK TIMESERIES
            stock_ts = asset.asset.spot(ts = True,
                                        ts_start = start, ts_end = end, ts_timewidth = ts_timewidth,
                                        ts_timeframe= ts_timeframe,
                                        spot_type = OptionModelAttributes.spot_type.value)
            stock_ts.rename(columns = {x: x.capitalize() for x in stock_ts.columns}, inplace = True)
            full_data['Stock_Close'] = stock_ts[['Close']]
            full_data.ffill(inplace = True)
            full_data.fillna(0, inplace = True)
            

            # GET rates timeseries
            intv = identify_interval(ts_timewidth,ts_timeframe)
            full_data['RF_rate'] = get_risk_free_rate_helper(intv)['annualized']

            ## Add Mapping for the columns that were filled
            full_data['DATA_FILL'] = 'NO'
            full_data.loc[close_fill_mask, 'DATA_FILL'] = 'YES'

            # Calculate the percentage change, mark change and get prev day data
            full_data[[f'prev_day_{col}' for col in full_data.columns]] = full_data.shift(1)
            full_data['Stock_Close_Change_Mark'] = full_data['Stock_Close'] - full_data['Stock_Close'].shift(1)
            full_data['Vol_Change_Mark'] = full_data[vol_col[0]] - full_data[vol_col[0]].shift(1)
            full_data['Option_Close_Change_Mark'] = full_data['Option_Close'] - full_data['Option_Close'].shift(1)
            full_data[[f'{x}_Change_Percent' for x in spot_col_2+['Stock_Close']] ] = full_data[spot_col_2+['Stock_Close']].pct_change()
            full_data[['RF_rate_Change_Mark']] = full_data[['RF_rate']] - full_data[['RF_rate']].shift(periods = 1)
            

            if method == "GB":

                ## ADD GREEKS. Make sure it is previous timestamp greeks
                greeks = asset.greeks(ts_start= start, 
                                    ts_end = end,
                                    ts_timeframe= ts_timeframe,
                                    ts_timewidth= ts_timewidth,
                                    greek_type='greeks')
                ## Fill missing greeks
                greek_replace_col = [f'{asset.default_fill.capitalize()}_{greek.lower()}' for greek in greeks_col]
                greeks = greeks[greeks.index.isin(full_data.index)]
                for index,replace_col in enumerate(greek_replace_col):
                    greeks.loc[close_fill_mask, greeks_col[index]] = greeks.loc[close_fill_mask, replace_col]
                full_data = full_data.join(greeks.shift(1))
                full_data.reset_index(inplace = True)
                ## Convert date/time change to seconds, then to days. This is most useful for intraday theta PnL
                full_data['total_seconds'] = (full_data['Datetime']-full_data['Datetime'].shift(1)).dt.total_seconds()/(24*60*60)



                PnL_Data = pd.DataFrame(index = full_data.index)
                PnL_Data['Delta_PnL'] = (full_data['Delta']*100)*full_data['Stock_Close_Change_Mark']
                PnL_Data['Gamma_PnL'] = (full_data['Gamma']*100)*((full_data['Stock_Close_Change_Mark'])**2)*0.5
                PnL_Data['Vega_PnL'] = (full_data['Vega']*100)*full_data['Vol_Change_Mark'] * 100
                PnL_Data['Theta_PnL'] = (full_data['Theta']*100) * full_data['total_seconds']
                PnL_Data['Rho_PnL'] = (full_data['Rho']*100)*full_data['RF_rate_Change_Mark'] * 100
                PnL_Data['Volga_PnL'] = (full_data['Volga']*100)*((full_data['Vol_Change_Mark'])**2)
                PnL_Data['Vanna_PnL'] = (full_data['Vanna']*100)*full_data['Stock_Close_Change_Mark']*full_data['Vol_Change_Mark']
                PnL_Data['Total_PnL'] = PnL_Data.sum(axis = 1)
                PnL_Data['Datetime'] = full_data['Datetime']
                PnL_Data['Option_Close_Change_percent'] = full_data['Option_Close_Change_Percent']
                PnL_Data['Stock_Close_Change_percent'] = full_data['Stock_Close_Change_Percent']
                PnL_Data['Option_Close_Change_Mark'] = full_data['Option_Close_Change_Mark']*100
                PnL_Data['Vol_Change_Diff'] = full_data['Vol_Change_Mark']   
                PnL_Data['Unexplained_PnL'] = PnL_Data['Option_Close_Change_Mark'] - PnL_Data['Total_PnL']
                PnL_Data['Price'] = full_data['Option_Close']*100
                PnL_Data['DATA_FILL'] = full_data['DATA_FILL']
                PnL_Data.set_index('Datetime', inplace = True)
                PnL_Data = PnL_Data[['Delta_PnL', 'Gamma_PnL', 'Theta_PnL', 'Vega_PnL','Volga_PnL', 'Vanna_PnL', 'Rho_PnL', 'Total_PnL', 'Unexplained_PnL', 'Option_Close_Change_Mark','Price' ]]
                PnL_Data.rename(columns= {'Option_Close_Change_Mark': 'Actual_PnL'}, inplace = True)
                PnL_Data = PnL_Data[(PnL_Data.index >= ts_start) & (PnL_Data.index <= ts_end)]
                full_data = full_data[(full_data['Datetime'] >= ts_start) & (full_data['Datetime'] <= ts_end)]
            
            elif method == "RV":
                full_data.reset_index(inplace = True)
                ## Convert date/time change to seconds, then to days. This is most useful for intraday theta PnL
                full_data['total_seconds'] = (full_data['Datetime']-full_data['Datetime'].shift(1)).dt.total_seconds()/(24*60*60)
                full_data['prev_day_Datetime'] = full_data.Datetime.shift(1)

                full_data.dropna(inplace = True)
                PnL_Data = full_data.apply(lambda x: fullRevalPnL(
                                                                    {'S0': x['prev_day_Stock_Close'],
                                                                      'K' : asset.K, 'rf_rate' : x['prev_day_RF_rate'], 
                                                                      'sigma' : x[f'prev_day_{vol_col[0]}'], 'start' : x['prev_day_Datetime'], 
                                                                      'put_call' : asset.put_call, 'exp' : asset.exp, 
                                                                      'y' : asset.y, 'price' : x['prev_day_Option_Close']   }, 

                                                                      
                                                                      {'S0': x['Stock_Close'], 
                                                                   'K' : asset.K, 'rf_rate' : x['RF_rate'], 
                                                                   'sigma' : x[vol_col[0]], 'start' : x['Datetime'], 
                                                                   'put_call' : asset.put_call, 'exp' : asset.exp,
                                                                     'y' : asset.y, 'price' : x['Option_Close']})
                    , axis = 1, result_type = 'expand')
                PnL_Data.set_index('Datetime', inplace = True)
                PnL_Data.index = pd.to_datetime(PnL_Data.index)
                full_data = full_data[(full_data['Datetime'] >= ts_start) & (full_data['Datetime'] <= ts_end)]
                PnL_Data = PnL_Data[(PnL_Data.index >= ts_start) & (PnL_Data.index <= ts_end)]
        
        elif asset.__class__.__name__ == 'OptionStructure':
            return_all = kwargs.get('return_all', False)
            return PatchedCalculateFunc(Calculate.attribution,long_leg = asset.long, short_leg = asset.short, 
                                        return_all = return_all, ts_start = ts_start, 
                                        ts_end = ts_end, ts_timeframe = ts_timeframe, 
                                        ts_timewidth = ts_timewidth, method = method, 
                                        replace = replace, return_both_data = return_both_data)



        else:
            raise Exception(f"Asset type {type(asset)} not supported")
        
        if return_both_data:
            return full_data, PnL_Data
        else:
            return PnL_Data

def fullRevalPnL(
        start_dict,
        end_dict,

):
    """
    Both dictionaries should have the same keys. As follows
    S0 = Spot Price
    K = Strike Price
    rf_rate = Risk Free Rate
    sigma = Volatility
    start = Start Date
    put_call = Put or Call
    exp = Expiry Date
    y = Dividend Rate
    price = Option Price


    start_dict: Dictionary containing the previous day data
    end_dict: Dictionary containing the current day data

    Rturns a dictionary containing the PnL attribution
    """

    assert start_dict.keys() == end_dict.keys(), f"Keys in both dictionaries must be the same. Expected {start_dict.keys()} recieved {end_dict.keys()}"
    assert start_dict['K'] == end_dict['K'], f"Strike Price must be the same. Expected {start_dict['K']} recieved {end_dict['K']}"

    S0, S1, price0, price1 = start_dict['S0'], end_dict['S0'], start_dict['price'], end_dict['price'] 
    K, r0, r1, sigma0, sigma1, y0, y1 = start_dict['K'], start_dict['rf_rate'], end_dict['rf_rate'], start_dict['sigma'], end_dict['sigma'], start_dict['y'], end_dict['y']
    start, end = start_dict['start'], end_dict['start']
    put_call, exp = start_dict['put_call'], start_dict['exp']
    ## Delta PnL
    
    ## To get the Delta PnL, we start by applying a very minute bump to the spot price, then we calculate the new option price
    ## ie we use T_1 data to calculate the new option price, but with S0 + bump


    S0_bump = S0 + 0.000001
    pv0 = Calculate.pv(S0 = S0, K = K, r = r0, sigma = sigma0, start = start, put_call = put_call, exp_date = exp, y = y0)
    pv1 = Calculate.pv(S0 = S1, K = K, r = r1, sigma = sigma1, start = end, put_call = put_call, exp_date = exp, y = y1)
    S0_pv_bump = Calculate.pv(S0 = S0_bump, K = K, r = r0, sigma = sigma0, start = start, put_call = put_call, exp_date = exp, y = y0)
    spot_change_pv = Calculate.pv(S0 = S1, K = K, r = r0, sigma = sigma0, start = start, put_call = put_call, exp_date = exp, y = y0)
    spot_change_pnl = spot_change_pv - pv0
    S0_bump_pnl = S0_pv_bump - pv0
    delta_pnl = (S1 - S0) * S0_bump_pnl * 1000000
    gamma_pnl = spot_change_pnl - delta_pnl


    ## Vega PnL
    ## Similar to the Delta PnL, Vega PnL starts by applying a very minute bump to vols, then we calculate the new option price
    
    bump = 0.0000001
    sigma_bump = sigma0 + bump
    sigma0_pv_bump = Calculate.pv(S0 = S0_bump, K = K, r = r0, sigma = sigma_bump, start = start, put_call = put_call, exp_date = exp, y = y0)
    sigma_change_pv = Calculate.pv(S0 = S0, K = K, r = r0, sigma = sigma1, start = start, put_call = put_call, exp_date = exp, y = y0)
    sigma_plus_spot_pv = Calculate.pv(S0 = S1, K = K, r = r0, sigma = sigma1, start = start, put_call = put_call, exp_date = exp, y = y0)
    sigma_plus_spot_pnl = sigma_plus_spot_pv - pv0
    sigma_change_pnl = sigma_change_pv - pv0
    sigma_bump_pnl = sigma0_pv_bump - pv0
    vega_pnl = (sigma1 - sigma0) * sigma_bump_pnl * 1/bump
    volga_pnl = sigma_change_pnl - vega_pnl
    vanna_pnl = sigma_plus_spot_pnl - delta_pnl - vega_pnl - gamma_pnl - volga_pnl
  
    ## Theta PnL
    pv0
    pv0_tplus1 = Calculate.pv(S0 = S0, K = K, r = r0, sigma = sigma0, start = end, put_call = put_call, exp_date = exp, y = y0)
    theta_pnl = pv0_tplus1

    ## Rho PnL
    rho_tplus1 = Calculate.pv(S0 = S0, K = K, r = r1, sigma = sigma0, start = start, put_call = put_call, exp_date = exp, y = y0)
    rho_pnl = rho_tplus1

    ## Dividend PnL
    div_tplus1 = Calculate.pv(S0 = S0, K = K, r = r0, sigma = sigma0, start = start, put_call = put_call, exp_date = exp, y = y1)
    div_pnl = div_tplus1

    ## Total PnL
    total_pnl = delta_pnl + gamma_pnl + vega_pnl + volga_pnl + theta_pnl + rho_pnl + vanna_pnl + d
    pnl = pv1-pv0
    pnl = price1 - price0
    return {'Delta_PnL': delta_pnl*100, 
            'Gamma_PnL': gamma_pnl* 100, 
            'Vega_PnL': vega_pnl*100, 
            'Volga_PnL': volga_pnl*100, 
            'Theta_PnL': theta_pnl*100,
            'Rho_PnL': rho_pnl*100, 
            'Vanna_PnL': vanna_pnl*100, 
            'Dividend_PnL': div_pnl*100, 
            'Total_PnL': total_pnl*100, 
            'Unexplained_PnL': ((pnl)*100) - total_pnl*100, 
            'Actual_PnL': (pnl*100), 
            'Datetime': end,
            'Price': price1*100}
