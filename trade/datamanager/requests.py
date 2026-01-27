from dataclasses import dataclass
from typing import Optional, Union
import pandas as pd
from trade.datamanager._enums import OptionSpotEndpointSource, SeriesId, VolatilityModel, OptionPricingModel
from trade.optionlib.config.types import DivType


@dataclass
class LoadRequest:
    
    ## Required parameters
    symbol: str

    ## Timeseries parameters
    start_date: Union[str, pd.Timestamp]
    end_date: Union[str, pd.Timestamp]

    ## Option specific parameters
    expiration: Union[str, pd.Timestamp]
    strike: Optional[float] = None
    right: Optional[str] = None

    ## Data type
    series_id: Optional[SeriesId] = None

    ## Enum types
    dividend_type: Optional[DivType] = None
    endpoint_source: Optional[OptionSpotEndpointSource] = None
    vol_model: Optional[VolatilityModel] = None
    market_model: Optional[OptionPricingModel] = None

    ## What to load
    load_spot: bool = True
    load_forward: bool = True
    load_dividend: bool = True
    load_rates: bool = True
    load_option_spot: bool = False
    load_vol: bool = True
    undo_adjust: bool = True

    def __post_init__(self):
        if self.load_option_spot:
            if self.strike is None or self.right is None:
                raise ValueError("Strike and right must be provided when loading option spot data.")
