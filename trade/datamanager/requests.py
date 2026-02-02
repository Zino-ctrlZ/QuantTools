from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Union
import pandas as pd
from trade.datamanager._enums import ModelPrice, OptionSpotEndpointSource, SeriesId, VolatilityModel, OptionPricingModel, RealTimeFallbackOption
from trade.optionlib.config.types import DivType


@dataclass
class LoadRequest:
    
    ## Required parameters
    symbol: str

    ## Timeseries parameters
    start_date: Optional[Union[str, pd.Timestamp]] = None
    end_date: Optional[Union[str, pd.Timestamp]] = None
    as_of: Optional[Union[str, pd.Timestamp]] = None
    rt: Optional[bool] = False
    on_date: Optional[bool] = False

    ## Option specific parameters
    expiration: Optional[Union[str, pd.Timestamp]] = None
    strike: Optional[float] = None
    right: Optional[str] = None

    ## Data type
    series_id: Optional[SeriesId] = None

    ## Enum types
    dividend_type: Optional[DivType] = None
    endpoint_source: Optional[OptionSpotEndpointSource] = None
    vol_model: Optional[VolatilityModel] = None
    market_model: Optional[OptionPricingModel] = None
    model_price: Optional[ModelPrice] = None
    fall_back_option: Optional[RealTimeFallbackOption] = None

    ## What to load
    load_spot: bool = False
    load_forward: bool = False
    load_dividend: bool = False
    load_rates: bool = False
    load_option_spot: bool = False
    load_vol: bool = False
    load_greek: bool = False
    undo_adjust: bool = True

    def __post_init__(self):
        ## Validation

        ## Dates:
        if self.rt:
            self.on_date = True
            self.as_of = datetime.now().date()

        if all(date is not None for date in [self.start_date, self.end_date, self.as_of]):
            raise ValueError("Only pass start_date and end_date or as_of, not both.")
        
        if all(date is None for date in [self.start_date, self.end_date, self.as_of]):
            raise ValueError("Either start_date and end_date or as_of must be provided.")
        
        if self.start_date is not None and self.end_date is not None:
            if pd.to_datetime(self.start_date) > pd.to_datetime(self.end_date):
                raise ValueError("start_date must be earlier than or equal to end_date.")
            
        if self.as_of is not None:
            if self.start_date is not None or self.end_date is not None:
                raise ValueError("If as_of is provided, start_date and end_date must be None.")
            self.as_of = pd.to_datetime(self.as_of)
            self.start_date = self.as_of
            self.end_date = self.as_of
            self.on_date = True

        ## Option parameters
        option_params = [self.expiration, self.strike, self.right]
        option_params_str = ["expiration", "strike", "right"]
        if self.load_greek or self.load_vol or self.load_option_spot:
            for i, param in enumerate(option_params):
                if param is None:
                    raise ValueError(f"{option_params_str[i]} must be provided when loading option data.")
            

        if self.load_option_spot:
            if self.strike is None or self.right is None:
                raise ValueError("Strike and right must be provided when loading option spot data.")
        
        if self.model_price is None:
            self.model_price = ModelPrice.MIDPOINT

