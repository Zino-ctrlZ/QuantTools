## To-Do: Consider using SSVI
## To-Do: Add a way to check if the model is valid, and rerun till valid. Log this
##To-Do: 


import os, sys
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.environ['WORK_DIR'])
sys.path.append(os.environ['DBASE_DIR'])
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
import math
from typing import Union
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import interp1d, CubicSpline
from trade.helpers.helper import optionPV_helper, generate_option_tick_new
from trade.helpers.Logging import setup_logger
from scipy.optimize import minimize 
from py_vollib.black_scholes_merton import black_scholes_merton
from trade.helpers.helper import time_distance_helper
import warnings
from pprint import pprint
import ipywidgets as widgets 
from abc import ABC, abstractmethod
from pathos.multiprocessing import ProcessingPool as Pool
from pathos.multiprocessing import cpu_count
from threading import Thread
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from trade.models.ModelLibrary import ModelLibrary
from dbase.database.SQLHelpers import DatabaseAdapter
from trade.helpers.decorators import log_error, log_error_with_stack
import copy
from copy import deepcopy
import plotly.graph_objects as go
import warnings
from sklearn.linear_model import LinearRegression
from abc import ABC, abstractmethod
from typing import Union, List, Literal
import math
warnings.filterwarnings('ignore')


logger = setup_logger('trade.models.VolSurface')
shutdown_event = False


## To-Do: Make sure SVI never throws nan values
class ModelBuilder(ABC):
    """
    Abstract class for building Vol Surface models. This is only for the actual model, not the surface itself.
    """
    @abstractmethod
    def build_model(self):
        pass


    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def predict(self):
        pass




class SurfaceManager(ABC):
    """
    Abstract class for managing volatility surfaces.
    """

    @abstractmethod
    def predict(self):
        pass

    def _is_dte_available_for_svi(self, dte, right):
        """
        Check if the given days to expiration (DTE) is available for the specified option type.

        Parameters:
        dte (int): The days to expiration to check.
        right (str): The type of option to check for. Must be one of "C" (call), "P" (put), "otm" (out of the money), or "itm" (in the money).

        Returns:
        bool: True if the DTE is available for the specified option type, False otherwise.

        Raises:
        ValueError: If the 'right' parameter is not one of "C", "P", "otm", or "itm".
        """
        if right == 'C':
            return dte in self.call_svi_params_table.index
        elif right == 'P':
            return dte in self.put_svi_params_table.index
        elif right == 'otm':
            return dte in self.call_svi_params_table.index and dte in self.put_svi_params_table.index
        elif right == 'itm':
            return dte in self.call_svi_params_table.index and dte in self.put_svi_params_table.index
        else:
            raise ValueError('Invalid right type. Must be one of "C", "P", "otm", or "itm"')


    def interpolate_svi(self, dte, right, interpolate_variables = True):
        """
        Interpolates the SVI (Stochastic Volatility Inspired) parameters for a given date to expiration (dte) and option type (right).
        
        Parameters:
        -----------
        dte : int or float
            The date to expiration for which the interpolation is to be performed.
        
        right : str
            The type of option, either 'C' for call or 'P' for put.
        
        interpolate_variables : bool, optional
            If True, interpolate the SVI variables; otherwise, interpolate the SVI parameters table. Default is True.
        
        Returns:
        --------
        pandas.DataFrame
            A DataFrame containing the interpolated SVI parameters for the specified dte.
        
        Raises:
        -------
        ValueError
            If the 'right' parameter is not one of 'C' or 'P'.
        """
        if right == 'C':
            if interpolate_variables:
                override_str = 'call_svi_variables'
            else:
                override_str = 'call_svi_params_table'
            schema = getattr(self, override_str).copy()

        elif right == 'P':
            if interpolate_variables:
                override_str = 'put_svi_variables'
            else:
                override_str = 'put_svi_params_table'
            schema = getattr(self, override_str).copy()

        else:
            raise ValueError(f'Invalid right type. Must be one of "C" or "P" Received: {right}')
        

        schema.loc[dte] = np.nan
        schema.sort_index(inplace=True)
        minimized = schema.iloc[max([0,schema.index.get_loc(dte) - 2]): min([schema.index.get_loc(dte) + 3, len(schema)-1])]
        minimized = minimized.interpolate(**self.interpolation_kwargs)
        # schema.loc[dte] = minimized.loc[dte]
        self.interpolated_dtes[right].append(dte)

        setattr(self, override_str, schema)
        return minimized
    
    
    def _sigmoid_func(self, x):
        x = np.log(x/self.spot)
        return 1/(1 + np.exp(4*x))

  



class DumasModelResults:
    """
    Class to store the results of the Dumas model.
    """
    def __init__(self):
        pass


class DumasRollingModel:
    """
    Class to implement the Dumas model for a rolling window of days to expiration (DTE).
    """
    ## to-do: Avoid errors when there are no values in the chain. This is coming from DumasModelBuilder when there are no values in the chain
    def __init__(self, chain, window, date,  right, period, dumas_width = .30,):
        """
        Initializes the VolSurface object with the given parameters.

        Parameters:
        chain (DataFrame): The option chain data.
        window (int): The window size for the volatility surface.
        date (datetime): The date for which the volatility surface is being created.
        right (str): The option type, either 'C' for call or 'P' for put.
        period (str): The period for the volatility surface, must be one of ['weekly', 'short', 'medium', 'long'].
        dumas_width (float, optional): The width parameter for the Dumas model, default is 0.30.

        Raises:
        AssertionError: If the period is not one of ['weekly', 'short', 'medium', 'long'].
        AssertionError: If the right is not one of ['C', 'P'].
        AssertionError: If the window is greater than the maximum window size for the given period.

        """
        terms = {
        'weekly': (0, 14, 5),
        'short': (14, 90, 30),
        'medium': (90, 360, 360),
        'long': (360, 850, 3300)
        }

        chain_terms = {
        'weekly':  chain[chain.DTE <= terms['weekly'][1]],
        'short': chain[(chain.DTE > terms['short'][0]) & (chain.DTE <= terms['short'][1])],
        'medium': chain[(chain.DTE > terms['medium'][0]) & (chain.DTE <= terms['medium'][1])],
        'long': chain[chain.DTE > terms['long'][0]]
        }

        assert period in ['weekly', 'short', 'medium', 'long'], f'Period must be one of {terms.keys()}'
        assert right in ['C', 'P'], f'Period must be one of {terms.keys()}'
        assert window <= terms[period][2], f'Period ({period}) Window must be less than {terms[period][2]}'

        try:
            # # Chain filtering
            chain['IV_filtered'] = chain['vol']
            lower_quantile = chain['vol'].quantile(0.01)
            upper_quantile = chain['vol'].quantile(0.99)
            chain = chain[(chain['vol'] > lower_quantile) & (chain['vol'] < upper_quantile)]
            self.period = period
            self.min_window = terms[period][0]
            self.max_window = terms[period][1]
            self.chain = chain_terms[period]
            self.window = terms[period][2]
            self.date = date
            self.right = right
            self.dumas_width = dumas_width
            self.models = []
            self.window_ranges = []
            self.spot = chain['Spot'].values[0]
            self.__initiate_dumas_params()
        except Exception as e:
            logger.error(f'Error in DumasRollingModel: {e}', exc_info = True)
            logger.error(f'Key Variables: {locals()}')
            raise e


    def __str__(self):
        return f'DumasRollingModel({self.period})'
    def __repr__(self):
        return f'DumasRollingModel({self.period})'
    
    def between(self, value):
        if isinstance(value, (int, float)):
            return self.min_window <= value <= self.max_window
        else:
            raise TypeError(f'"{value}" is not a numeric format. {self.__class__.__name__} returns truth value for DTE comparisons')
        
    def __initiate_dumas_params(self):
        reduced_chain = self.chain
        dumas_width = self.dumas_width 
        right = self.right
        reduced_chain = reduced_chain.sort_values('DTE')
        moneyness_params = {'min': 1-dumas_width, 'max': 1 + dumas_width}
        reduced_chain['Moneyness'] = reduced_chain['strike']/reduced_chain['Spot']
        dumas_chain = reduced_chain[(reduced_chain['vol'] > 0) & (reduced_chain['Moneyness'] > moneyness_params['min']) & (reduced_chain['Moneyness'] < moneyness_params['max']) ]
        dumas_chain['X'] = np.log(dumas_chain['strike'] / dumas_chain['Spot'])
        dumas_chain['T'] = dumas_chain['DTE'] / 365
        dumas_chain['IV'] = dumas_chain['vol']
        dumas_chain['X^2'] = dumas_chain['X'] ** 2
        dumas_chain['X^3'] = dumas_chain['X'] ** 3
        dumas_chain['XT'] = dumas_chain['X'] * dumas_chain['T']
        dumas_chain['T^2'] = dumas_chain['T'] ** 2
        dumas_chain['DTE'] = dumas_chain['DTE'].astype(int)
        dumas_chain = dumas_chain[dumas_chain.right == right]
        self.dumas_chain = dumas_chain 

    def fit_dumas_model_rolling(self):
        """
        Fits a rolling linear regression model to the implied volatility (IV) data using the Dumas model.

        The method fits the model over different rolling windows of the data, depending on the specified period.
        If the period is 'long' or 'medium', a single model is fitted over the entire data range.
        Otherwise, models are fitted over rolling windows of a specified size.

        Attributes:
            dumas_chain (DataFrame): The data containing the DTE (days to expiration), IV (implied volatility), 
                         and other relevant columns for the Dumas model.
            window (int): The size of the rolling window.
            period (str): The period type, which can be 'long', 'medium', or other values.
            min_window (int): The minimum window size for the rolling model.
            max_window (int): The maximum window size for the rolling model.
            models (list): A list to store the fitted linear regression models.
            window_ranges (list): A list to store the ranges of the rolling windows used for each model.

        Returns:
            None
        """
        chain = self.dumas_chain
        window_size = self.window
        T_values = chain.DTE.unique()
        if self.period not in ['long', 'medium']:
            for i in range(self.min_window, max(T_values)+1, 1):
                start = i
                end = i + window_size
                use_chain = chain[(chain['DTE'] >= start) & (chain['DTE'] <= end)]
                if use_chain.empty:
                    continue
                linear_model = LinearRegression()
                X = use_chain[['X', 'X^2', 'XT', 'T', 'T^2']]
                y = use_chain['IV']
                linear_model.fit(X, y)
                self.models.append(linear_model)
                self.window_ranges.append((start, end))


        else:
            linear_model = LinearRegression()
            X = chain[['X', 'X^2', 'XT', 'T', 'T^2']]
            y = chain['IV']
            linear_model.fit(X, y)
            self.models.append(linear_model)
            self.window_ranges.append((self.min_window, self.max_window))

    ## To-do: Predict IV for multiple T & K values, currently works for single T multiple K
    def predict_average(self, dte_target, k_range = None):
        def predict_average(self, dte_target, k_range=None):
            """
            Predicts the average volatility surface for a given target days to expiration (DTE).
            Parameters:
            -----------
            dte_target : int
                The target days to expiration (DTE) for which the prediction is to be made. Must be between self.min_window and self.max_window.
            k_range : int, float, or array-like, optional
                The range of strike prices (k) for which the prediction is to be made. If None, a default range is used. If a single int or float is provided, it is converted to an array.
            Returns:
            --------
            DumasModelResults
                An object containing the predictions and the mean prediction for the given DTE and strike price range.
            Raises:
            -------
            AssertionError
                If dte_target is not between self.min_window and self.max_window.
            """
        assert self.min_window <= dte_target <= self.max_window, f'DTE must be between {self.min_window} and {self.max_window}'
        if isinstance(k_range, (int, float)):
            k_range = np.array([k_range])
        predictions = []
        self.prediction_window = []
        T = dte_target/365
        
        if k_range is None:
            k_range = np.linspace(min(self.dumas_chain['X']), max(self.dumas_chain['X']), 100)
        T2 = np.array([T ** 2 for _ in range(len(k_range))])
        X2 = np.array([k_range ** 2 for k_range in k_range])
        XT = np.array([k_range * T for k_range in k_range])
        X = k_range
        
        for window, model in zip(self.window_ranges, self.models):
            model_func = lambda X, X2, XT, T, T2: model.intercept_ + model.coef_[0]*X + model.coef_[1]*X2 + model.coef_[2]*XT + model.coef_[3]*T + model.coef_[4]*T2
            start, end = window
            if (start <= dte_target) & (end >= dte_target):
                self.prediction_window.append(window)
                y_pred = np.array([model_func(X, X2, XT, T, T2)])
                predictions.append(y_pred)
        

        result_obj = DumasModelResults()
        result_obj.predictions = predictions
        result_obj.mean_prediction = np.mean(predictions, axis = 0)
        
        return result_obj
    
    def predict_distance_weighted_average(self, dte_target, k_range = None):
        if k_range is None:
            k_range = np.linspace(min(self.dumas_chain['X']), max(self.dumas_chain['X']), 100)
        resultObj = self.predict_average(dte_target, k_range)
        distance = [abs(dte_target - ((start + end)/2)) for start, end in self.prediction_window]
        w = [1/d if d != 0 else 1 for d in distance]
        norm_w = np.array(w)/sum(w)
        distance_weighted_prediction = np.average(resultObj.predictions, axis = 0, weights = norm_w)
        resultObj.distance_weighted_prediction = distance_weighted_prediction
        return resultObj


class DumasModelBuilder(ModelBuilder):

    """
    Class to build the Dumas model for a given option chain. Relies on the DumasRollingModel class for the actual model fitting. 
    This is a simplified version to reduce the complexity for end user. It predicts the IV for a given DTE and Strike Price
    """
    
    def __init__(self,
                 date, 
                 right,
                 full_chain = None,
                 dumas_width = .30,):

        full_chain = full_chain.copy()
        if full_chain.empty:
            raise ValueError('Chain is empty!')
        
        full_chain.columns = full_chain.columns.str.lower()
        
        if 'spot' in full_chain.columns:
            full_chain = full_chain.rename(columns = {'spot': 'Spot'})

        if 'dte' in full_chain.columns:
            full_chain = full_chain.rename(columns = {'dte': 'DTE'})

        
        self.spot = full_chain['Spot'].values[0]
        self.full_chain = full_chain
        self.calc_dumas = True if isinstance(full_chain, pd.DataFrame) else False
        self.date = date
        self.right = right
        self.dumas_width = dumas_width
        self.dumas_vols = []
        self.right_name = 'Call' if self.right == 'C' else 'Put'
        self.__init_dumas_params()




    def build_model(self):
        """
        Builds and fits the Dumas models.

        Iterates over the `dumas_models` dictionary, fits each model using the 
        `fit_dumas_model_rolling` method, and updates the dictionary with the fitted models.

        Returns:
            None
        """
        for key, value in self.dumas_models.items():
            value.fit_dumas_model_rolling()
            self.dumas_models[key] = value
            

    def __init_dumas_params(self):
        reduced_chain = self.full_chain
        right = self.right
        reduced_chain['Moneyness'] = reduced_chain['strike']/reduced_chain['Spot']
        dumas_chain = reduced_chain[(reduced_chain['vol'] > 0)]
        reduced_chain = reduced_chain[reduced_chain.right == right]
        reduced_chain['X'] = np.log(reduced_chain['strike']/reduced_chain['Spot'])
        self.dumas_chain = reduced_chain
        wkly_dumas = DumasRollingModel(reduced_chain, 5, self.date, self.right, 'weekly', self.dumas_width)
        sht_dumas = DumasRollingModel(reduced_chain, 25, self.date, self.right, 'short', self.dumas_width)
        medium_dumas = DumasRollingModel(reduced_chain, 100, self.date, self.right, 'medium', self.dumas_width)
        long_dumas = DumasRollingModel(reduced_chain, 200, self.date, self.right, 'long', self.dumas_width)
        self.dumas_models = {'weekly': wkly_dumas, 'short': sht_dumas, 'medium': medium_dumas, 'long': long_dumas}


    
    def plot(self, dte):
        """
        Plots the implied volatility surface for a given days to expiration (DTE).

        Parameters:
        dte (int): The days to expiration for which the volatility surface is to be plotted.

        Raises:
        BaseException: If the provided DTE is larger than the maximum permitted DTE.

        The method performs the following steps:
        1. Iterates through the dumas models to find the appropriate model for the given DTE.
        2. Filters the dumas chain for the given DTE.
        3. Checks if the provided DTE is within the permissible range.
        4. If the dumas chain is empty, generates a default X range.
        5. Predicts the average implied volatility using the dumas model.
        6. Plots the market implied volatility and the predicted implied volatility using Plotly.

        The plot includes:
        - Market Implied Volatility vs Strike Price (if dumas chain is not empty)
        - Dumas Implied Volatility vs Strike Price
        - Title indicating the right name and DTE
        - X-axis labeled as 'Strike Price'
        - Y-axis labeled as 'Implied Volatility'
        - Plot dimensions set to 800x800
        """
        dumas_chain = self.dumas_chain
        for item, value in self.dumas_models.items():
            if value.between(dte):
                dumas_obj = value
                dumas_chain = dumas_obj.dumas_chain
                dumas_chain = dumas_chain[dumas_chain['DTE'] == dte]

        if dte > self.dumas_models['long'].max_window:
            raise BaseException( f'dte cannot be larger than largest permitted - {self.dumas_models["long"].max_window}')

        if dumas_chain.empty:
            X = np.linspace(-1, 2, 100)
        else:
            X = dumas_chain['X']
        self.dumas_obj = dumas_obj.predict_average(dte, X)
        y_pred = self.dumas_obj.mean_prediction[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X, y=dumas_chain['vol'], mode='markers', name='Market IV')) if not dumas_chain.empty else None
        fig.add_trace(go.Scatter(x=X, y=y_pred, mode='markers', name='Dumas IV'))
        fig.update_layout(title=f'{self.right_name}:{dte} DTE Market IV vs Strike Price', xaxis_title='Strike Price', yaxis_title='Implied Volatility', height = 800, width = 800)
        fig.show()

    def predict(self, dte: Union[np.array, int,float], k):
        """
        Predict the Implied Volatility for a given DTE and Strike Price

        params:
        - dte: Days to Expiry
        - k: Strike Price, expecting non-normalized. k should be in the same scale as the chain data

        return:
        - IV: Implied Volatility
        """
        k = np.log(k/self.spot)

        for item, value in self.dumas_models.items():
            if value.between(dte):
                dumas_obj = value
                prediction = dumas_obj.predict_average(dte, k)
        return prediction.mean_prediction[0]
    



class SVIModelBuilder(ModelBuilder):
    """
    SVIModelBuilder is a class that builds a Stochastic Volatility Inspired (SVI) model for a given options chain.

    Attributes:
        cons (tuple): Constraints for the optimization process.
        chain (DataFrame): The options chain data.
        K (ndarray): Array of strike prices.
        spot (float): Spot price of the underlying asset.
        r (float): Risk-free interest rate.
        q (float): Dividend yield.
        T (ndarray): Array of times to expiration in years.
        t (float): Time to expiration in years for the first option in the chain.
        IV (ndarray): Array of implied volatilities.
        right (str): Type of option ('C' for call, 'P' for put).
        S0 (float): Spot price of the underlying asset.
        DTE (int): Days to expiration.
        date (datetime): Date of the options chain.
        chain_price (ndarray): Array of option prices.
        bsm_price (list): List of Black-Scholes-Merton prices for the options.
        Price (list): List of model prices for the options.
        right_name (str): Name of the option type ('Call' or 'Put').
        svi_params1 (list): Initial SVI parameters (unscaled wings).
        svi_params2 (list): Initial SVI parameters (scaled wings).
        new_svi_params1 (ndarray): Optimized SVI parameters (unscaled wings).
        svi_mse1 (float): Mean squared error for the unscaled wings model.
        new_svi_params2 (ndarray): Optimized SVI parameters (scaled wings).
        svi_mse2 (float): Mean squared error for the scaled wings model.
        preferred_svi_params (ndarray): Preferred SVI parameters based on MSE.
        preferred_mse (float): Mean squared error for the preferred model.
        preferred_text (str): Description of the preferred model.
        preferred_svi_variables (dict): SVI variables for the preferred model.
        unpreferred_svi_params (ndarray): Unpreferred SVI parameters.
        unpreferred_mse (float): Mean squared error for the unpreferred model.
        unpreferred_text (str): Description of the unpreferred model.
        unpreferred_svi_variables (dict): SVI variables for the unpreferred model.
        all_guesses1 (list): List of all guesses during optimization (unscaled wings).
        all_guesses2 (list): List of all guesses during optimization (scaled wings).

    Methods:
        __init__(self, chain, date, DTE): Initializes the SVIModelBuilder with the given options chain, date, and days to expiration.
        __str__(self): Returns a string representation of the SVIModelBuilder.
        __repr__(self): Returns a string representation of the SVIModelBuilder.
        __eq__(self, value): Checks if the DTE is equal to the given value.
        __lt__(self, value): Checks if the DTE is less than the given value.
        __gt__(self, value): Checks if the DTE is greater than the given value.
        __le__(self, value): Checks if the DTE is less than or equal to the given value.
        __ge__(self, value): Checks if the DTE is greater than or equal to the given value.
        __filter_chain(self): Filters the options chain based on implied volatility and days to expiration.
        __init_svijw_params(self): Initializes the SVI parameters.
        build_model(self): Builds the SVI model by optimizing the SVI parameters.
        plot(self): Plots the implied volatility surface.
        predict(self, t, k): Predicts the implied volatility for given time to expiration and strike prices.
        save_guesses1(self, x): Saves the guesses during optimization (unscaled wings).
        save_guesses2(self, x): Saves the guesses during optimization (scaled wings).
        save_to_databse(self): Placeholder method for saving the model to a database.
    """


    cons=(
        {'type': 'ineq', 'fun': lambda x: x[0] - 0.00001}, #v > 0 
        {'type': 'ineq', 'fun': lambda x: x[0] - x[4]}, #v >= v_tilde
        {'type': 'ineq', 'fun': lambda x: x[4] - 0.00001}, #v_tilde > 0 
        {'type': 'ineq', 'fun': lambda x: x[2]}, #p >= 0
        {'type': 'ineq', 'fun': lambda x: x[3]}, #c >= 0
        )
    


    def __init__(self,
                 chain, 
                 date, 
                 DTE,
                 ):
        """
        Initializes the VolSurface object.

        Parameters:
        chain (pd.DataFrame): The option chain data.
        date (datetime): The date of the option chain.
        DTE (int): Days to expiration.

        Raises:
        ValueError: If the chain is empty.

        Attributes:
        chain (pd.DataFrame): Filtered option chain data.
        K (np.ndarray): Array of strike prices.
        spot (float): Spot price of the underlying asset.
        r (float): Risk-free interest rate.
        q (float): Dividend yield.
        T (np.ndarray): Array of times to expiration in years.
        t (float): Time to expiration in years for the first option.
        IV (np.ndarray): Array of implied volatilities.
        right (str): Option type ('C' for Call, 'P' for Put).
        S0 (float): Spot price of the underlying asset.
        DTE (int): Days to expiration for the first option.
        date (datetime): The date of the option chain.
        chain_price (np.ndarray): Array of option prices from the chain.
        bsm_price (list): List of Black-Scholes-Merton prices.
        Price (list): List of model prices (Call or Put).
        right_name (str): Name of the option type ('Call' or 'Put').

        """
        if chain.empty:
            raise ValueError('Chain is empty!')


        self.chain = chain
        chain = self.__filter_chain()
        self.K = chain['strike'].values
        self.spot = chain['Spot'].values[0]
        self.r = chain['r'].values[0]
        self.q = chain['q'].values[0]
        self.T = chain.DTE.values/365
        self.t = chain['DTE'].values[0]/365
        self.IV = chain['vol'].values
        self.right = chain['right'].unique()[0]
        self.S0 = chain['Spot'].values[0]
        self.DTE = chain.DTE.values[0]
        self.date = date
        self.chain_price = chain['price'].values
        self.bsm_price = [black_scholes_merton('p', self.S0, self.K[i], self.T[i], self.r, self.IV[i], self.q) for i in range(len(self.T))]

        if self.right == 'C':
            self.Price = [ModelLibrary.CallPrice(self.S0, self.IV[i], self.K[i], self.T[i], self.r, self.q) for i in range(len(self.T))]
        else:
            self.Price = [ModelLibrary.PutPrice(self.S0, self.IV[i], self.K[i], self.T[i], self.r, self.q) for i in range(len(self.T))]
        self.right_name = 'Call' if self.right == 'C' else 'Put'
        
        self.__init_svijw_params()
        

        
    def __str__(self):
        return f'SVIModelBuilder(Right: {self.right_name}, DTE: {self.DTE})'

    def __repr__(self):
        return f'SVIModelBuilder(Right: {self.right_name}, DTE: {self.DTE})'
    
    
    def __eq__(self, value):
        if isinstance(value, (int, float)):
            return self.DTE == value
        else:
            raise TypeError(f'"{value}" is not a numeric format. {self.__class__.__name__} returns truth value for DTE comparisons')

    def __lt__(self, value):
        if isinstance(value, (int, float)):
            return self.DTE < value
        else:
            raise TypeError(f'"{value}" is not a numeric format. {self.__class__.__name__} returns truth value for DTE comparisons')
        
    def __gt__(self, value):
        if isinstance(value, (int, float)):
            return self.DTE > value
        else:
            raise TypeError(f'"{value}" is not a numeric format. {self.__class__.__name__} returns truth value for DTE comparisons')
    
    def __le__(self, value):
        if isinstance(value, (int, float)):
            return self.DTE <= value
        else:
            raise TypeError(f'"{value}" is not a numeric format. {self.__class__.__name__} returns truth value for DTE comparisons')

    def __ge__(self, value):
        if isinstance(value, (int, float)):
            return self.DTE >= value
        else:
            raise TypeError(f'"{value}" is not a numeric format. {self.__class__.__name__} returns truth value for DTE comparisons')

    def __filter_chain(self):

        chain_ = self.chain
        chain = chain_.copy()

        ## Temp fix for chains filtering out strikes > ATM
        ## Ensure max available strike is greater than 5% of spot and min available strike is less than 5% of spot
        ## Wing_filters checks if the max and min strikes are within 5% of the spot price, then if statement uses the opposite to allow ones with strikes outside the range


        chain['IV_filtered'] = chain['vol']
        lower_quantile = chain['vol'].quantile(0.01)
        upper_quantile = chain['vol'].quantile(0.99)
        chain = chain[(chain['vol'] > lower_quantile) & (chain['vol'] < upper_quantile)]
        chain = chain[(chain['vol'] > 0)]
        chain = chain[(chain['DTE'] > 2)]

        iv = interp1d(chain['strike'], chain['vol'],)
        try:
            iv(chain['Spot'].values[0]*1.01)
            iv(chain['Spot'].values[0]*0.99)
        except Exception as e:
            str_e = str(e)
            if 'in x_new is below' in str_e:
                self.chain = chain_
                return chain_
            elif 'in x_new is above' in str_e:
                self.chain = chain_
                return chain_
            
            else:
                raise e
    
        self.chain = chain
        return chain

    
    def __init_svijw_params(self):
        params = [0,0,0,0,0]
        IV = interp1d(self.K, self.IV)
        t = self.t
        ## Generating left wing & right wing slopes using extreme (change at extreme/0.02) * 1/(ATM Vol - scaled to t)
        adj_k = self.K
        atm_iv = IV(self.spot)
        slope_min_wt = abs(-((IV(min(adj_k)*np.exp(0.01)) - IV(min(adj_k)))/0.02) * 1/((atm_iv**2*t)**.5))
        slope_max_wt = abs((IV(max(adj_k)) - IV(max(adj_k)*np.exp(-0.01)))/0.02) * 1/((atm_iv**2*t)**.5)

        ## Optionally adding a scale factor
        slope_scaling_min = ((IV(min(self.K))*100*2*t)/ (IV(self.spot)*100*2*t))
        slope_scaling_max = ((IV(max(self.K))*100*2*t)/ (IV(self.spot)*100*2*t))

        params[2], params[3] = slope_min_wt, slope_max_wt
        
        ## Params 4 is min variance
        minIV = min(self.IV)
        params[4] = minIV**2
        
        atm_scale_dict = {
            'weekly': .05,
            'medium': .025,
            'long': .01
        }

        atm_scale = .01
        
        try:
            if self.right == 'C':
                params[1]  = ((IV(self.spot* (1+atm_scale)) - IV(self.spot* (1-atm_scale)))/2*100*t**0.5) # PSI is skew of the ATM. (101-99)/2 * 100 * sqrt(t)
                if params[1] > 0.000000:
                    params[1] *= -1   
            else:
                params[1]  = ((IV(self.spot* (1-atm_scale)) - IV(self.spot* (1+atm_scale)))/2*100*t**0.5)
                if params[1] < 0.000000: ## Note: This is a hack to avoid negative psi, maybe shouldnt be doing this
                    params[1] *= -1
        except Exception as e:
            atm_scale = .01 ## To-do: Add a better way to handle this
            ## To-do: Find a way to return "Do not build" if error is too persistent
            ## Reducing the atm scale to avoid errors
            if self.right == 'C':
                params[1]  = ((IV(self.spot* (1+atm_scale)) - IV(self.spot* (1-atm_scale)))/2*100*t**0.5) # PSI is skew of the ATM. (101-99)/2 * 100 * sqrt(t)
            else:
                params[1]  = ((IV(self.spot* (1-atm_scale)) - IV(self.spot* (1+atm_scale)))/2*100*t**0.5)


        ## Params 0 is ATM VOl
        params[0] = IV(self.spot)**2 #v from ATM vol
        
        self.svi_params1 = copy.deepcopy(params)
        self.svi_params2 = copy.deepcopy(params)

        ## Optionally scaling the slope
        params[2], params[3] = slope_min_wt * slope_scaling_min, slope_max_wt * slope_scaling_max
        self.svi_params2 = copy.deepcopy(params)


    def build_model(self):
        """
        Builds the volatility surface model using the SVI-JW (Stochastic Volatility Inspired - Jim Gatheral and Antoine Jacquier) method.

        The method minimizes the mean squared error (MSE) between the implied volatility (IV) and the SVI-JW model using the specified optimization method.
        It produces results for both scaled and unscaled wings and selects the preferred parameters based on the MSE.

        Steps:
        1. Initializes the optimization method and function.
        2. Minimizes the MSE for unscaled wings, stores the results, and checks for NaN values.
        3. Minimizes the MSE for scaled wings, stores the results, and checks for NaN values.
        4. Selects the preferred parameters based on the MSE values and ensures no NaN values are present.
        5. Stores both the preferred and unpreferred parameters and their corresponding MSE values and SVI variables.

        Note:
            - The method assumes that self.svi_params1, self.svi_params2, self.cons, self.T, self.S0, self.K, and self.IV are already defined.
            - If the MSE for both sets of parameters is NaN, the preferred parameters will be set to NaN.
        """
        min_func = ModelLibrary.MSE_IV_SVIJW
        method = 'SLSQP'
        # method = 'L-BFGS-B' 
        # method = 'BFGS'
        # method = 'Nelder-Mead'

        ## Producing results for scaled and unscaled wings
        self.all_guesses1 = []
        result = minimize(min_func, self.svi_params1, constraints=self.cons, method = method, tol = 1e-10, args=(deepcopy(self.T), deepcopy(self.S0), deepcopy(self.K), deepcopy(self.IV)), callback=self.save_guesses1) 
        self.new_svi_params1 = result.x
        self.svi_mse1 = (min_func(self.new_svi_params1, self.T, self.S0, self.K, self.IV))
        
        ## To-do: Find a way to ensure no nan, for now will just use interpolation at surface manager
        if math.isnan(self.svi_mse1):
            self.new_svi_params1 = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])

        self.all_guesses2 = []
        result = minimize(min_func, self.svi_params2, constraints=self.cons, method = method, tol = 1e-10, args=(deepcopy(self.T), deepcopy(self.S0), deepcopy(self.K), deepcopy(self.IV)), callback=self.save_guesses2) 
        self.new_svi_params2 = result.x
        self.svi_mse2 = (min_func(self.new_svi_params2, self.T, self.S0, self.K, self.IV))
        if math.isnan(self.svi_mse2):
            self.new_svi_params1 = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])


        # This sets the preferred to first params if the mse is not nan, else uses scaled params
        if not math.isnan(self.svi_mse1):
            self.preferred_svi_params = self.new_svi_params1
            self.preferred_mse = self.svi_mse1
            self.preferred_text = 'wings_unscaled'
            self.preferred_svi_variables = dict(zip(
                ['a', 'b', 'rho', 'sigma', 'm'], 
                ModelLibrary.SVIVariables(self.S0, self.K[0], self.T[0], *self.preferred_svi_params)
            ))

            self.unpreferred_svi_params = self.new_svi_params2
            self.unpreferred_mse = self.svi_mse2
            self.unpreferred_text = 'wings_scaled'
            self.unpreferred_svi_variables = dict(zip(
                ['a', 'b', 'rho', 'sigma', 'm'],
                ModelLibrary.SVIVariables(self.S0, self.K[0], self.T[0], *self.unpreferred_svi_params)
            ))
        else:
            self.preferred_svi_params = self.new_svi_params2
            self.preferred_mse = self.svi_mse2
            self.preferred_text = 'wings_scaled'
            self.preferred_svi_variables = dict(zip(
                ['a', 'b', 'rho', 'sigma', 'm'],
                ModelLibrary.SVIVariables(self.S0, self.K[0], self.T[0], *self.preferred_svi_params)
            ))

            self.unpreferred_svi_params = self.new_svi_params1
            self.unpreferred_mse = self.svi_mse1
            self.unpreferred_text = 'wings_unscaled'
            self.unpreferred_svi_variables = dict(zip(
                ['a', 'b', 'rho', 'sigma', 'm'],
                ModelLibrary.SVIVariables(self.S0, self.K[0], self.T[0], *self.unpreferred_svi_params)
            ))


        ## This chooses the preferred params based on the MSE


        ## Naming convention for preferred and unpreferred params
        ## To-do: Add a check for which is preferred
        ## To-do: Naming is convoluted, refactor, maybe?
  

        # Assuming self.svi_mse1 and self.svi_mse2 are floats or can be treated as such
        # if math.isnan(self.svi_mse2) or (not math.isnan(self.svi_mse1) and self.svi_mse1 <= self.svi_mse2):
        #     # Case: mse2 is NaN or mse1 is less than or equal to mse2
        #     self.preferred_svi_params = self.new_svi_params1
        #     self.preferred_mse = self.svi_mse1
        #     self.preferred_text = 'wings_unscaled'
        #     self.preferred_svi_variables = dict(zip(
        #         ['a', 'b', 'rho', 'sigma', 'm'], 
        #         ModelLibrary.SVIVariables(self.S0, self.K[0], self.T[0], *self.preferred_svi_params)
        #     ))

        #     self.unpreferred_svi_params = self.new_svi_params2
        #     self.unpreferred_mse = self.svi_mse2
        #     self.unpreferred_text = 'wings_scaled'
        #     self.unpreferred_svi_variables = dict(zip(
        #         ['a', 'b', 'rho', 'sigma', 'm'], 
        #         ModelLibrary.SVIVariables(self.S0, self.K[0], self.T[0], *self.unpreferred_svi_params)
        #     ))

        # else:
        #     # Case: mse1 is NaN or mse2 is less than mse1
        #     self.preferred_svi_params = self.new_svi_params2
        #     self.preferred_mse = self.svi_mse2
        #     self.preferred_text = 'wings_scaled'
        #     self.preferred_svi_variables = dict(zip(
        #         ['a', 'b', 'rho', 'sigma', 'm'], 
        #         ModelLibrary.SVIVariables(self.S0, self.K[0], self.T[0], *self.preferred_svi_params)
        #     ))

        #     self.unpreferred_svi_params = self.new_svi_params1
        #     self.unpreferred_mse = self.svi_mse1
        #     self.unpreferred_text = 'wings_unscaled'
        #     self.unpreferred_svi_variables = dict(zip(
        #         ['a', 'b', 'rho', 'sigma', 'm'], 
        #         ModelLibrary.SVIVariables(self.S0, self.K[0], self.T[0], *self.unpreferred_svi_params)
        #     ))

    
    def plot(self):
        """
        Plots the implied volatility (IV) against the strike price for the given volatility surface.

        This method generates a plot that includes:
        - Market implied volatility (IV) as markers.
        - Preferred SVI model implied volatility as a line.
        - Unpreferred SVI model implied volatility as a line.

        The plot is displayed using Plotly and includes the following features:
        - Title indicating the right name and days to expiration (DTE).
        - X-axis labeled as 'Strike Price'.
        - Y-axis labeled as 'Implied Volatility'.
        - Customizable height and width of the plot.

        Parameters:
        None

        Returns:
        None
        """
        range_ = np.linspace(min(self.K), max(self.K), 100)
        unpre_params = self.unpreferred_svi_params
        unpre_text = self.unpreferred_text
        new_vols = (ModelLibrary.TotalVarSVIJW(self.S0, range_, self.T[0], *self.preferred_svi_params)/self.T[0])**0.5
        unpref_vols = (ModelLibrary.TotalVarSVIJW(self.S0, range_, self.T[0], *unpre_params)/self.T[0])**0.5
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.K, y=self.IV, mode='markers', name='Market IV'))
        fig.add_trace(go.Scatter(x=range_, y=new_vols, mode='lines', name=f'{self.preferred_text} - Preferred'))
        fig.add_trace(go.Scatter(x=range_, y=unpref_vols, mode='lines', name=unpre_text))
        fig.update_layout(title=f'{self.right_name}:{self.DTE} DTE Market IV vs Strike Price', xaxis_title='Strike Price', yaxis_title='Implied Volatility', height = 800, width = 800)
        fig.show()

    def predict(self, t, k):
        if isinstance(k, (int, float)):
            k = np.array([k])

        elif isinstance(k, list):
            k = np.array(k)

        elif isinstance(k, (np.ndarray, np.array)):
            pass

        else:
            raise TypeError(f'Strike price must be a numeric value or a list of numeric values, recieved {type(k)}')
            

        ## Note: Sticking to using TotalVarSVIJW for now, will refactor to use VolSVI. Until we can ensure no nan values
        # return ModelLibrary.VolSVI(self.spot, k, self.t, **self.preferred_svi_variables)
        return np.sqrt(ModelLibrary.TotalVarSVIJW(self.spot, k, self.t, *self.preferred_svi_params)/self.t)


    def save_guesses1(self, x):
        self.all_guesses1.append(x)

    def save_guesses2(self, x):
        self.all_guesses2.append(x)

    def save_to_databse(self):
        pass







def build_model(obj, builder):
    try:
        builder.build_model()
        obj.svi_models.append(builder)
        return builder
        # return f"{builder.__class__.__name__} model built successfully"

    except Exception as e:
        return e

def shutdown(pool):
    global shutdown_event
    shutdown_event
    shutdown_event = True
    pool.terminate()


def pool_builder(obj, builders):
    try:
        pool = Pool(len(builders), )
        pool.timeout = 30
        pool.restart()
        partial_build = partial(build_model, obj)
        results = pool.map(partial_build, builders)

    except KeyboardInterrupt as e:

        shutdown_event = True
        shutdown(pool)
        raise

    except Exception as e:
        shutdown(pool)


    finally:
        pool.close()
        pool.join()

    return results




class SurfaceBuilder(ABC):
    """
    Abstract class for building Vol Surface models. This is the surface itself. The Put & Call surfaces with this
    """

    def __init__(self,
                 date, 
                 full_chain,
                 right,
                 dumas_width=0.50):
        """
        Initialize the SurfaceBuilder.

        Args:
            chain (pd.DataFrame): The subset of the options chain.
            date (datetime): The date of the surface.
            full_chain (pd.DataFrame): The full options chain.
            right (str): Option type ('C' for call, 'P' for put).
            dumas_width (float): Width parameter for Dumas model.
        """
        full_chain = full_chain.copy()
        self.svi_models = []  # List to store SVI models
        self.date = date
        self.full_chain = full_chain[full_chain['right'] == right]  # Keep only options of the same type
        self.right = right
        self.dumas_model = DumasModelBuilder( date, right, full_chain, dumas_width)  # Initialize Dumas model

    @abstractmethod
    def build_surface(self):
        """Abstract method to build the surface."""
        pass

    def build_model(self, focus_dtes: [list, int] = None):
        """
        Fit models to the chain data.
        """

        # Build Dumas model
        dumas_worker = self.dumas_model
        dumas_worker.build_model()
        dtes = self.full_chain.DTE.unique()  # Unique days to expiration
        right = self.right # Option types
        

        svis = []
        threads = []

        for dte in dtes:
            if dte <= 2: ## Not interested in DTE less than 2
                continue

            spot = self.full_chain['Spot'].values[0]
 
            chain_subset = self.full_chain[(self.full_chain['DTE'] == dte) & 
                                        (self.full_chain['right'] == right) & 
                                        (self.full_chain['vol'] > 0)]
            

            ## Adding range filter for DTE
            ## Max Strike & Min Strike should be within 5% of spot price
            ## To-do: Refactor this to be more dynamic. it's hard coded for now
            if chain_subset.strike.max() < spot*1.01 or chain_subset.strike.min() > spot*0.99:
                continue

            ## No Empty Chains
            if chain_subset.empty:
                continue
            
            worker = SVIModelBuilder(chain_subset, self.date, dte)
            svis.append(worker)
            
        results = pool_builder(self, svis)
        self.svi_models = results




class CallVolSurfaceBuilder(SurfaceBuilder):
    """
    Class to build the call volatility surface.
    """
    def __init__(self, date, full_chain, dumas_width):
        super().__init__( date, full_chain, 'C', dumas_width)
        # self.build_model()

    def build_surface(self):
        return super().build_surface()



    def __eq__(self, value):
        pass


class PutVolSurfaceBuilder(SurfaceBuilder):
    """
    Class to build the put volatility surface.
    """
    def __init__(self, date, full_chain, dumas_width):
        super().__init__( date, full_chain, 'P', dumas_width)


    def build_surface(self):
        return super().build_surface()
    




class SurfaceManagerModelBuild(SurfaceManager):
    """SurfaceManager is a class designed to manage and predict implied volatilities for options using both Dumas and SVI models. 
    Attributes:
    interpolation_kwargs : dict
        Keyword arguments for the interpolation method used in the class.
    ticker : str
        The ticker symbol of the underlying asset.
    date : str
        The date for which the surface is being managed.
    full_chain : pd.DataFrame
        The full option chain data.
    dumas_width : float
        The width parameter for the Dumas model.
    spot : float
        The spot price of the underlying asset.
    call_builder : CallVolSurfaceBuilder
        An instance of CallVolSurfaceBuilder for building call option surfaces.
    put_builder : PutVolSurfaceBuilder
        An instance of PutVolSurfaceBuilder for building put option surfaces.
    call_svi_variables : pd.DataFrame
        DataFrame containing the preferred SVI variables for call options.
    call_svi_params_table : pd.DataFrame
        DataFrame containing the preferred SVI parameters for call options.
    put_svi_params_table : pd.DataFrame
        DataFrame containing the preferred SVI parameters for put options.
    put_svi_variables : pd.DataFrame
        DataFrame containing the preferred SVI variables for put options.
    interpolated_dtes : dict
        Dictionary to keep track of interpolated DTEs for call and put options.
    Methods:
    __init__(self, ticker, date, full_chain, dumas_width)
        Initializes the SurfaceManager with the given parameters.
    predict(self, dte, k, right, interpolate_variables=True)
    interpolate_svi(self, dte, right, interpolate_variables=True)
        Interpolates the SVI parameters for a given date to expiration (dte) and option type (right).
    _sigmoid_func(self, x)
        A sigmoid function used for weighting in the interpolation process.
    _is_dte_available_for_svi(self, dte, right)
        Checks if the given days to expiration (DTE) is available for the specified option type.
    __has_dte_been_interpolated(self, dte)
        Checks if the given days to expiration (DTE) has been interpolated."""
    
    ## To-do: Might be best to not switch preferred based on MSE, this is because the MSE is not a good indicator of the quality of the fit, and it's unstable when interpolating 
    interpolation_kwargs = {'method': 'spline', 'axis': 0, 'order': 1}

    def __init__(self, ticker, date, full_chain, dumas_width):
        ## Only filer chain based on Expiries, not on whole chain

        super().__init__()

        full_chain.columns = full_chain.columns.str.lower()
        
        if 'spot' in full_chain.columns:
            full_chain = full_chain.rename(columns = {'spot': 'Spot'})

        if 'dte' in full_chain.columns:
            full_chain = full_chain.rename(columns = {'dte': 'DTE'})

        self.ticker = ticker
        self.date = date
        self.full_chain = full_chain
        self.dumas_width = dumas_width
        self.spot = self.full_chain['Spot'].values[0]
        self.dbAdapter = DatabaseAdapter()
        self.call_builder = CallVolSurfaceBuilder( date, full_chain.copy(), dumas_width)
        self.put_builder = PutVolSurfaceBuilder( date, full_chain.copy(), dumas_width)
        
        self.call_builder.build_model()
        self.put_builder.build_model()
        
        self.call_svi_variables = pd.DataFrame({model.DTE: model.preferred_svi_variables for model in self.call_builder.svi_models}).T
        self.call_svi_params_table = pd.DataFrame({model.DTE: model.preferred_svi_params for model in self.call_builder.svi_models}).T
        self.put_svi_params_table = pd.DataFrame({model.DTE: model.preferred_svi_params for model in self.put_builder.svi_models}).T
        self.put_svi_variables = pd.DataFrame({model.DTE: model.preferred_svi_variables for model in self.put_builder.svi_models}).T
        self.call_svi_variables.sort_index(inplace=True)
        self.put_svi_variables.sort_index(inplace=True)
        
        self.call_svi_variables = self.call_svi_variables.interpolate(**self.interpolation_kwargs)
        self.call_svi_params_table = self.call_svi_params_table.interpolate(**self.interpolation_kwargs)
        self.put_svi_variables = self.put_svi_variables.interpolate(**self.interpolation_kwargs)
        self.put_svi_params_table = self.put_svi_params_table.interpolate(**self.interpolation_kwargs)
        self.interpolated_dtes = {'C': [], 'P': []}
        save_thread = Thread(target=self.organize_data_to_save)
        save_thread.start()

    def __str__(self):
        return f'SurfaceManagerModelBuild for {self.ticker} on {self.date}'
    
    def __repr__(self):
        return f'SurfaceManagerModelBuild({self.ticker}, {self.date})'

    def organize_data_to_save(self):
        ## To-do: Add a method to organize data to save to database
        
        data = self.call_svi_params_table.copy()
        data.index.name = 'DTE'
        data.columns = ['v', 'psi', 'p', 'c', 'v_tilde']
        data['ticker'] = self.ticker
        data['build_date'] = self.date
        data['right'] = 'C'
        data  = data.reset_index().merge(self.full_chain[['DTE', 'expiration']], how = 'left', on = 'DTE').drop_duplicates()
        self.dbAdapter.save_to_database(data, 'vol_surface', 'svi_jw_params')


        data = self.put_svi_params_table.copy()
        data.index.name = 'DTE'
        data.columns = ['v', 'psi', 'p', 'c', 'v_tilde']
        data['ticker'] = self.ticker
        data['build_date'] = self.date
        data['right'] = 'P'
        data = data.reset_index().merge(self.full_chain[['DTE', 'expiration']], how = 'left', on = 'DTE').drop_duplicates()
        self.dbAdapter.save_to_database(data, 'vol_surface', 'svi_jw_params')




    def predict(self, dte, k, right, interpolate_variables = False):
        """
        Predicts the implied volatility for given days to expiration (DTE) and strike price(s).
        
        Parameters:
        -----------
        dte : int
            Days to expiration. Must be less than or equal to the maximum DTE in the full chain.
        k : int, float, list, or np.ndarray
            Strike price(s). Can be a single numeric value, a list of numeric values, or a numpy array.
        right : str
            Option type. Must be one of "C" (call), "P" (put), "otm" (out of the money), or "itm" (in the money).
        interpolate_variables : bool, optional
            Whether to interpolate variables for SVI model if DTE is not directly available. Default is True.
        
        Returns:
        --------
        predictions : dict
            A dictionary containing the predicted implied volatilities. Keys include:
            - 'k': The input strike prices.
            - 'dumas': Predicted volatilities using the Dumas model.
            - 'svi': Predicted volatilities using the SVI model.
        
        Raises:
        -------
        ValueError
            If DTE is larger than the maximum permitted DTE or if an invalid option type is provided.
        TypeError
            If the strike price is not a numeric value or a list/array of numeric values.
        """

        T = dte/365
        if dte > max(self.full_chain.DTE):
            raise ValueError(f'DTE cannot be larger than the largest permitted - {max(self.full_chain.DTE)}')
        predictions = {}
        if isinstance(k, (int, float)):
            k = np.array([k])
        elif isinstance(k, list):
            k = np.array(k)
        elif isinstance(k, (np.ndarray, np.array)):
            pass
        else:
            raise TypeError(f'Strike price must be a numeric value or a list of numeric values. Received: {type(k)}')
        
        predictions['k'] = k
        if right == 'C':
            builder = self.call_builder
        elif right == 'P':
            builder = self.put_builder
        elif right == 'otm':
            builder = {'C': self.call_builder, 'P': self.put_builder}
        elif right == 'itm':
            builder = {'C': self.call_builder, 'P': self.put_builder}
        else:
            raise ValueError(f'Invalid right type. Must be one of "C", "P", "otm", or "itm" Received: {right}')
        
        ## Predict both SVI & Dumas model with builders

        ## If not dict, then use the builder directly, no joining wings
        if not isinstance(builder, dict):
            dumas_prediction = builder.dumas_model.predict(dte, k)
            predictions['dumas'] = dumas_prediction
            if self._is_dte_available_for_svi(dte, right) and dte not in self.interpolated_dtes[right]:
                for model in builder.svi_models:
                    if model == dte:
                        svi_prediction = model.predict(dte, k)
                        predictions['svi'] = svi_prediction
            else:
                schema = self.interpolate_svi(dte, right, interpolate_variables)

                if interpolate_variables:
                    schema_var = schema.loc[dte].to_dict() 
                    svi_prediction = ModelLibrary.VolSVI(self.full_chain['Spot'].values[0], k, dte/365, **(schema_var))
                else:
                    schema_var = schema.loc[dte].tolist()
                    svi_prediction = np.sqrt((ModelLibrary.TotalVarSVIJW(self.full_chain['Spot'].values[0], k, dte/365, *schema_var))/T)
                predictions['svi'] = svi_prediction

        
        else:
            call_dumas_prediction = builder['C'].dumas_model.predict(dte, k)
            put_dumas_prediction = builder['P'].dumas_model.predict(dte, k)
            svi_predictions = {}
            for _right in ['C', 'P']:
                if self._is_dte_available_for_svi(dte, _right) and dte not in self.interpolated_dtes[_right]:
                    builder_obj = builder[_right]
                    for model in builder_obj.svi_models:
                        if model == dte:
                            svi_predictions[_right] = model.predict(dte, k)

                else:
                    schema = self.interpolate_svi(dte, _right, interpolate_variables)

                    if interpolate_variables:
                        schema_var = schema.loc[dte].to_dict() 
                        svi_prediction = ModelLibrary.VolSVI(self.full_chain['Spot'].values[0], k, dte/365, **(schema_var))
                    else:
                        schema_var = schema.loc[dte].tolist()
                        svi_prediction = np.sqrt((ModelLibrary.TotalVarSVIJW(self.full_chain['Spot'].values[0], k, dte/365, *schema_var))/T)
                    svi_predictions[_right] = svi_prediction
            
            if right == 'otm':
                w = self._sigmoid_func(k)
                predictions['dumas'] = w*put_dumas_prediction + (1-w)*call_dumas_prediction
                predictions['svi'] = w*svi_predictions['P'] + (1-w)*svi_predictions['C']

            elif right == 'itm':
                w = self._sigmoid_func(k)
                predictions['dumas'] = w * call_dumas_prediction + (1-w) * put_dumas_prediction
                predictions['svi'] = w * svi_predictions['C'] + (1-w) * svi_predictions['P']

        return predictions
        
    def __has_dte_been_interpolated(self, dte):
        return dte in self.interpolated_dtes



class SurfaceManagerDatabase(SurfaceManager):
    
    """ 
    SurfaceManagerDatabase is a class that manages the volatility surface modeling for options.
    This class initiates the svi parameters from database.
        Attributes:
        date : datetime
            The build date of the full option chain.
        full_chain : DataFrame
            The full option chain data.
        call_svi_params : dict
            Parameters for the call SVI model.
        put_svi_params : dict
            Parameters for the put SVI model.
        CallDumasBuilder : DumasModelBuilder
            Builder for the call Dumas model.
        PutDumasBuilder : DumasModelBuilder
            Builder for the put Dumas model.
        interpolation_kwargs : dict
            Keyword arguments for the interpolation method used in the class.
        Methods:
        __init__(tick, full_chain, call_svi_params, put_svi_params):
            Initializes the SurfaceManagerDatabase with the provided parameters.
        predict(dte, k, right, interpolate_variables=True):
    """
    interpolation_kwargs = {'method': 'spline', 'axis': 0, 'order': 1}
    def __init__(self, tick, full_chain, call_svi_params, put_svi_params):
        super().__init__()
        full_chain.columns = full_chain.columns.str.lower()
        
        if 'spot' in full_chain.columns:
            full_chain = full_chain.rename(columns = {'spot': 'Spot'})

        if 'dte' in full_chain.columns:
            full_chain = full_chain.rename(columns = {'dte': 'DTE'})

        self.tick = tick
        self.spot = full_chain['Spot'].values[0]
        self.full_chain = full_chain.rename(columns = {'spot': 'Spot'})
        self.date = full_chain.build_date.values[0]
        self.call_svi_params_table = call_svi_params.set_index('dte')[['v', 'psi', 'p', 'c', 'v_tilde']]
        self.put_svi_params_table = put_svi_params.set_index('dte')[['v', 'psi', 'p', 'c', 'v_tilde']]
        self.CallDumasBuilder = DumasModelBuilder(self.date, 'C', full_chain)
        self.PutDumasBuilder = DumasModelBuilder(self.date, 'P', full_chain)
        self.CallDumasBuilder.build_model()
        self.PutDumasBuilder.build_model()
        self.interpolated_dtes = {'C': [], 'P': []}


    def __str__(self):
        return f"SurfaceManagerDatabase({self.tick} on {pd.to_datetime(self.date).strftime('%Y%m%d')})"
    
    def __repr__(self):
        return f"SurfaceManagerDatabase({self.tick} on {pd.to_datetime(self.date).strftime('%Y%m%d')})"
    
    def __getattribute__(self, name):
        model_build_args = ['call_builder',
                        'call_svi_params_table',
                        'call_svi_variables',
                        'date',
                        'dbAdapter',
                        'dumas_width',
                        'full_chain',
                        'interpolate_svi',
                        'interpolated_dtes',
                        'interpolation_kwargs',
                        'organize_data_to_save',
                        'predict',
                        'put_builder',
                        'put_svi_params_table',
                        'put_svi_variables',
                        'spot',
                        'ticker']
        try:
            return super().__getattribute__(name)
        except:
            if name in model_build_args:
                raise AttributeError(f"'SurfaceManagerDatabase' object has no attribute '{name}', this attribute is only available in 'SurfaceManagerModelBuild' object. If you want to use this attribute, please use 'SurfaceManagerModelBuild' object or 'force_build = True' in SurfaceLab.")
            else:
                raise AttributeError(f"'SurfaceManagerDatabase' object has no attribute '{name}'")
            

    def predict(self, dte, k, right, interpolate_variables = True):
        """
        Predicts the implied volatility for given days to expiration (DTE) and strike price(s).
        
        Parameters:
        -----------
        dte : int
            Days to expiration. Must be less than or equal to the maximum DTE in the full chain.
        k : int, float, list, or np.ndarray
            Strike price(s). Can be a single numeric value, a list of numeric values, or a numpy array.
        right : str
            Option type. Must be one of "C" (call), "P" (put), "otm" (out of the money), or "itm" (in the money).
        interpolate_variables : bool, optional
            Whether to interpolate variables for SVI model if DTE is not directly available. Default is True.
        
        Returns:
        --------
        predictions : dict
            A dictionary containing the predicted implied volatilities. Keys include:
            - 'k': The input strike prices.
            - 'dumas': Predicted volatilities using the Dumas model.
            - 'svi': Predicted volatilities using the SVI model.
        
        Raises:
        -------
        ValueError
            If DTE is larger than the maximum permitted DTE or if an invalid option type is provided.
        TypeError
            If the strike price is not a numeric value or a list/array of numeric values.
        """
        interpolate_variables = False  ## Note: Consider adding this back to the func call above
        T = dte/365
        if dte > max(self.full_chain.DTE):
            raise ValueError(f'DTE cannot be larger than the largest permitted - {max(self.full_chain.DTE)}')
        predictions = {}
        if isinstance(k, (int, float)):
            k = np.array([k])
        elif isinstance(k, list):
            k = np.array(k)
        elif isinstance(k, (np.ndarray, np.array)):
            pass
        else:
            raise TypeError(f'Strike price must be a numeric value or a list of numeric values. Received: {type(k)}')
        
        predictions['k'] = k
        if right == 'C':
            builder = self.CallDumasBuilder
        elif right == 'P':
            builder = self.PutDumasBuilder
        elif right == 'otm':
            builder = {'C': self.CallDumasBuilder, 'P': self.PutDumasBuilder}
        elif right == 'itm':
            builder = {'C': self.CallDumasBuilder, 'P': self.PutDumasBuilder}
        else:
            raise ValueError(f'Invalid right type. Must be one of "C", "P", "otm", or "itm" Received: {right}')
        
        ## Predict both SVI & Dumas model with builders

        ## If not dict, then use model params loaded from database
        if not isinstance(builder, dict):
            dumas_prediction = builder.predict(dte, k)
            predictions['dumas'] = dumas_prediction
            if self._is_dte_available_for_svi(dte, right) and dte not in self.interpolated_dtes[right]:
                schema_table = self.call_svi_params_table if right == 'C' else self.put_svi_params_table
                schema_var = schema_table.loc[dte]
                svi_prediction = np.sqrt((ModelLibrary.TotalVarSVIJW(self.full_chain['Spot'].values[0], k, dte/365, *schema_var))/T)
                predictions['svi'] = svi_prediction
            
            else:
                schema = self.interpolate_svi(dte, right, interpolate_variables)

                if interpolate_variables:
                    schema_var = schema.loc[dte].to_dict() 
                    svi_prediction = ModelLibrary.VolSVI(self.full_chain['Spot'].values[0], k, dte/365, **(schema_var))
                else:
                    schema_var = schema.loc[dte].tolist()
                    svi_prediction = np.sqrt((ModelLibrary.TotalVarSVIJW(self.full_chain['Spot'].values[0], k, dte/365, *schema_var))/T)
                predictions['svi'] = svi_prediction

        
        else:
            call_dumas_prediction = builder['C'].predict(dte, k)
            put_dumas_prediction = builder['P'].predict(dte, k)
            svi_predictions = {}
            for _right in ['C', 'P']:
                if self._is_dte_available_for_svi(dte, _right) and dte not in self.interpolated_dtes[_right]:
                    schema_table = self.call_svi_params_table if _right == 'C' else self.put_svi_params_table
                    schema_var = schema_table.loc[dte]
                    svi_prediction = np.sqrt((ModelLibrary.TotalVarSVIJW(self.full_chain['Spot'].values[0], k, dte/365, *schema_var))/T)
                    svi_predictions[_right] = svi_prediction

                else:
                    schema = self.interpolate_svi(dte, _right, interpolate_variables)

                    if interpolate_variables:
                        schema_var = schema.loc[dte].to_dict() 
                        svi_prediction = ModelLibrary.VolSVI(self.full_chain['Spot'].values[0], k, dte/365, **(schema_var))
                    else:
                        schema_var = schema.loc[dte].tolist()
                        svi_prediction = np.sqrt((ModelLibrary.TotalVarSVIJW(self.full_chain['Spot'].values[0], k, dte/365, *schema_var))/T)
                    svi_predictions[_right] = svi_prediction
            
            if right == 'otm':
                w = self._sigmoid_func(k)
                predictions['dumas'] = w*put_dumas_prediction + (1-w)*call_dumas_prediction
                predictions['svi'] = w*svi_predictions['P'] + (1-w)*svi_predictions['C']

            elif right == 'itm':
                w = self._sigmoid_func(k)
                predictions['dumas'] = w * call_dumas_prediction + (1-w) * put_dumas_prediction
                predictions['svi'] = w * svi_predictions['C'] + (1-w) * svi_predictions['P']

        return predictions



class SurfaceLab:
    """
    SurfaceLab is a class that manages the volatility surface modeling for options.
    This class initiates the SurfaceManager from either the database or the model build.
    """
    def __init__(self, tick, date, full_chain, dumas_width, force_build = False):

        full_chain.columns = full_chain.columns.str.lower()
        self.ticker = tick
        self.build_date = date
        self.dumas_width = dumas_width
        self.spot = full_chain['spot'].values[0]
        self.force_build = force_build

        
        if 'spot' in full_chain.columns:
            full_chain = full_chain.rename(columns = {'spot': 'Spot'})

        if 'dte' in full_chain.columns:
            full_chain = full_chain.rename(columns = {'dte': 'DTE'})

        ## Try getting params from database
        self.dbAdapter = DatabaseAdapter()
        
        self.full_chain = full_chain
        query = f"""
        SELECT * FROM svi_jw_params WHERE ticker = '{tick}' AND build_date = '{date}' """
        svi_data = self.dbAdapter.query_database('vol_surface', 'svi_jw_params', query)

        if not force_build:

            if svi_data.empty:
                self.manager = SurfaceManagerModelBuild(tick, date, full_chain, dumas_width)

            else:
                calls_svi_params = svi_data[svi_data['right'] == 'C']
                puts_svi_params = svi_data[svi_data['right'] == 'P']
                self.manager = SurfaceManagerDatabase(tick, full_chain, calls_svi_params, puts_svi_params)
        else:
            self.manager = SurfaceManagerModelBuild(tick, date, full_chain, dumas_width)


    def __str__(self):
        return f"SurfaceLab({self.ticker} on {pd.to_datetime(self.build_date).strftime('%Y%m%d')})"
    
    def __repr__(self):
        return f"SurfaceLab({self.ticker} on {pd.to_datetime(self.build_date).strftime('%Y%m%d')})"
    
    def predict(self, dte, k, right, interpolate_variables = True):
        return self.manager.predict(dte, k, right, interpolate_variables)

