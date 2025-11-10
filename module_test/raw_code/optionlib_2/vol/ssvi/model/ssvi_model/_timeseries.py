# pylint: disable=global-statement
# pylint: disable=broad-exception-caught
# pylint: disable=arguments-differ
from __future__ import annotations
from typing import ClassVar, Optional, Literal
import time
from datetime import datetime
import pandas as pd
from pydantic import Field, PrivateAttr, ConfigDict, BaseModel
from dateutil.relativedelta import relativedelta
import numpy as np
import traceback
from trade.helpers.Logging import setup_logger
from trade.helpers.pydantic import loud_post_init
from trade.helpers.types import SingletonMixin
from trade.helpers.helper import (
    is_weekend, 
    not_trading_day, 
    check_missing_dates, 
    assert_member_of_enum
)
from module_test.raw_code.optionlib_2.vol.ssvi.model.ssvi_model._ssvi import _SSVIModel
from module_test.raw_code.optionlib_2.vol.ssvi.model.ssvi_model._eod_ssvi import EODMarketSSVIModel
from module_test.raw_code.optionlib_2.vol.ssvi.model.chain import MarketChainLoader
from module_test.raw_code.optionlib_2.vol.ssvi.types import VolType, DivType, VolSide
from module_test.raw_code.optionlib_2.vol.ssvi.controller import (
    get_global_config,
)
logger = setup_logger('optionlib.ssvi.model.ssvi_model')
class SsviTimeseriesEOD(BaseModel, SingletonMixin):
    """
    End-of-day SSVI timeseries model for managing SSVI models over time.
    This class extends the SSVIParentModel to handle time series data for SSVI models.
    """

    ## Class variable to cache instances
    _instances: ClassVar[dict[str, "EODMarketSSVIModel"]] = {}
    _initialized: bool = PrivateAttr(default=False)
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)
    symbol: str = Field(..., description="Symbol of the underlying asset")
    _model_set: dict = PrivateAttr(default_factory=dict)

    ## Optional Inputs/Derived inputs
    ## In other model classes, div_type & model is populated from chain
    ## But since this will be loading other models, it is populated from global
    _model: VolType = PrivateAttr(default=None)
    _div_type: DivType = PrivateAttr(default=None)
    models: Optional[dict[str, _SSVIModel]] = Field(default=None, description="Dictionary of SSVI models for different option sides")

    @property
    def iterations(self) -> int:
        """
        Returns the number of iterations for model fitting.
        """
        return get_global_config().model_iterations
    
    @iterations.setter
    def iterations(self, value: int):
        get_global_config().model_iterations = value

    @property
    def chunk_size(self) -> int:
        """
        Returns the chunk size for processing.
        """
        return get_global_config().chunk_size

    @chunk_size.setter
    def chunk_size(self, value: int):
        get_global_config().chunk_size = value

    @loud_post_init
    def model_post_init(self, context):
        global_config = get_global_config()
        self._model = global_config.vol_type
        self._div_type = global_config.div_type

    def __new__(cls, symbol: str, *args, **kwargs):
        key = symbol
        if key not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[key] = instance
        else:
            logger.info("Using cached instance for %s", symbol)
        return cls._instances[key]
    
    def __init__(self, *args, **data):
        # First-time init for this cached instance:
        # If __pydantic_private__ isn't set yet, it's the first real init.
        if getattr(self, "__pydantic_private__", None) is None:
            super().__init__(*args, **data)     # sets fields and creates private store
            self._initialized = True            # safe now

    @classmethod
    def clear_instances(cls, clear_tree: bool = False):
        """
        Clear all cached instances of SsviTimeseriesEOD and optionally clear instances of EODMarketSSVIModel and MarketChainLoader.
        Args:
            clear_tree (bool): If True, also clear instances of EODMarketSSVIModel and MarketChainLoader.
        """
        cls._instances.clear()
        if clear_tree:
            EODMarketSSVIModel.clear_instances()
            MarketChainLoader.clear_instances()
            SsviTimeseriesEOD.clear_all_instances()

        

    @classmethod
    def instances(cls) -> dict[str, "EODMarketSSVIModel"]:
        return cls._instances
    
    @property
    def model_set(self) -> dict[str, EODMarketSSVIModel]:
        return self._model_set
    
    @property
    def model(self) -> VolType:
        return self._model
    
    @property
    def div_type(self) -> DivType:
        return self._div_type
    
    @model.setter
    def model(self, value: VolType):
        enum_v = assert_member_of_enum(value, VolType)
        self._model = enum_v

    @div_type.setter
    def div_type(self, value: DivType):
        enum_v = assert_member_of_enum(value, DivType)
        self._div_type = enum_v

    def __repr__(self):
        return f"<SsviTimeseriesEOD(symbol={self.symbol})>"
    
    def _get_model_for_date(self, valuation_date: str|datetime) -> EODMarketSSVIModel:
        """
        Retrieve or create an EODMarketSSVIModel for the given valuation date.
        
        Args:
            valuation_date (str | datetime): The valuation date for the model.

        Returns:
            EODMarketSSVIModel: The EODMarketSSVIModel for the given date.
        """
        if isinstance(valuation_date, str):
            valuation_date = pd.to_datetime(valuation_date)
        valuation_date = valuation_date.strftime('%Y-%m-%d')

        # Check if the model already exists for the given date
        if valuation_date in self._model_set:
            return self._model_set[valuation_date]

        # Create a new model if it doesn't exist
        new_model = EODMarketSSVIModel(valuation_date=valuation_date, symbol=self.symbol)
        new_model.fit() ## Load and save
        logger.critical("Loaded model for Tick %s on %s", self.symbol, valuation_date)
        self._model_set[valuation_date] = new_model
        return new_model
    
    def _prepare_inputs(self,
                        k: float| np.ndarray,
                        exp: str| datetime| np.ndarray,
                        strike_type: Literal['p', 'k', 'pf', 'f'],
                        right: VolSide) -> tuple[np.ndarray, np.ndarray, VolSide]:
        """
        Normalize k, exp and right inputs to arrays/enums and validate strike_type requirements.
        Returns: (k_array, exp_array, right_enum)
        """
        # Normalize strike_type default
        if strike_type is None:
            strike_type = 'p'

        # Prepare k: use defaults when k is not provided, otherwise ensure an array
        if k is None:
            if strike_type in ('k', 'pf'):
                raise ValueError(f"When strike_type is '{strike_type}', k must be provided.")
            k_arr = np.array([-0.1, 0.0, 0.1]) if strike_type == 'f' else np.array([0.9, 1.0, 1.1])
        else:
            k_arr = np.atleast_1d(k)

        # Prepare exp: default to '3m' when not provided, otherwise ensure an array
        exp_arr = np.atleast_1d(exp) if exp is not None else np.array(['3m'])

        # Prepare right: default from global config, validate if provided as string
        if right is None:
            right_enum = get_global_config().vol_side
        elif isinstance(right, str):
            right_enum = assert_member_of_enum(right, VolSide)
        else:
            right_enum = right

        return k_arr, exp_arr, right_enum

    def _prepare_date_range(self, start_date: str|datetime, end_date: str|datetime) -> tuple[pd.DatetimeIndex, pd.Timestamp, pd.Timestamp]:
        """
        Normalize and validate the date range, returning all business days in the range.
        """
        if start_date is None:
            start_date = datetime.today() - relativedelta(weeks=1)
        if end_date is None:
            end_date = datetime.today()

        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)

        if start_ts > end_ts:
            raise ValueError("start_date must be earlier than or equal to end_date")

        all_dates = pd.date_range(start=start_ts, end=end_ts, freq='B')
        return all_dates, start_ts, end_ts

    def _ensure_missing_models_loaded(self, all_dates: pd.DatetimeIndex, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> None:
        """
        Check for missing trading dates and ensure models are loaded for required dates.
        """
        missing = check_missing_dates(x=pd.DataFrame({'Datetime': all_dates}), _start=start_ts, _end=end_ts)
        if not missing:
            return

        logger.warning("Missing dates in the range: %s. Will load this missing data", missing)
        for dt in missing:
            if is_weekend(dt) or not_trading_day(dt):
                logger.info("Skipping weekend or non-trading date: %s", dt)
                continue
            try:
                self._get_model_for_date(dt)
            except Exception as e:
                logger.error("Error loading model for %s: %s", dt, e)

    def _collect_predictions(self,
                             all_dates: pd.DatetimeIndex,
                             k: np.ndarray,
                             exp: np.ndarray,
                             strike_type: Literal['p', 'k', 'pf', 'f'],
                             right: VolSide) -> list[pd.DataFrame]:
        """
        Iterate business dates, call per-date model.predict and collect formatted dataframes.
        """
        results: list[pd.DataFrame] = []
        for current_date in all_dates:
            if not_trading_day(current_date):
                logger.info("Skipping non-trading day: %s", current_date)
                continue
            try:
                model = self._get_model_for_date(current_date)
                df_vols = model.predict(k=k, exp=exp, strike_type=strike_type, right=right)
                df_vols = df_vols.reset_index()
                df_vols['Datetime'] = current_date.strftime('%Y-%m-%d')
                df_vols.set_index(['Datetime', 'strike', 'exp'], inplace=True)
                results.append(df_vols)
            except Exception as e:
                logger.error("Error predicting for %s: %s", current_date, e)
                logger.error("Traceback:\n%s", traceback.format_exc())
        return results

    def predict(self,
                k: float| np.ndarray = None,
                exp: str| datetime| np.ndarray = None,
                strike_type: Literal['p', 'k', 'pf', 'f'] = 'f',
                right: VolSide = None,
                start_date: str|datetime = None,
                end_date: str|datetime = None) -> pd.DataFrame:
        """
        Predict the implied volatility for a given strike and expiration over a date range.
        This method orchestrates parsing inputs, ensuring models are loaded for missing dates,
        and collecting per-date predictions.
        """
        # Normalize inputs
        start = time.time()
        k_arr, exp_arr, right_enum = self._prepare_inputs(k, exp, strike_type, right)
        logger.info("Prepared inputs in %.2f seconds", time.time() - start)

        # Prepare date range
        all_dates, start_ts, end_ts = self._prepare_date_range(start_date, end_date)
        logger.info("Prepared date range from %s to %s with %d business days", start_ts.strftime('%Y-%m-%d'), end_ts.strftime('%Y-%m-%d'), len(all_dates))

        # Ensure missing models are loaded
        self._ensure_missing_models_loaded(all_dates, start_ts, end_ts)
        logger.info("Prepared missing models in %.2f seconds", time.time() - start)

        # Collect per-date predictions
        results = self._collect_predictions(all_dates, k_arr, exp_arr, strike_type, right_enum)
        logger.info("Collected predictions in %.2f seconds", time.time() - start)

        if not results:
            raise ValueError("No results were generated. Check if the date range includes valid trading days.")
        return pd.concat(results).sort_index()

