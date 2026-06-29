"""LoadRequest dataclass for model data loading orchestration.

Validates date windows and optional pre-loaded timeseries inputs before
``_load_model_data`` / ``_load_model_data_timeseries`` fetch missing factors; rt/as_of
window expansion is handled in the orchestrator, not on ``LoadRequest`` pegged dates.

Comment density: domain policy

Core Classes:
    LoadRequest: Flags and payloads for vol/greeks/theo data loading.
"""

from __future__ import annotations

from datetime import date, datetime
from dataclasses import dataclass
from typing import  Optional, Set, Union

import pandas as pd

from trade.datamanager._enums import (
    ModelPrice,
    OptionPricingModel,
    OptionSpotEndpointSource,
    RealTimeFallbackOption,
    SeriesId,
    VolatilityModel,
)
from trade.datamanager.result import (
    DividendsResult,
    ForwardResult,
    GreekResultSet,
    OptionSpotResult,
    RatesResult,
    SpotResult,
    VolatilityResult,
)
from trade.datamanager.utils.classification import resolve_checked_missing_dates_for_option_contract
from trade.helpers.helper import get_missing_dates, to_datetime
from trade.helpers.Logging import setup_logger
from trade.optionlib.config.types import DivType
from trade.datamanager.utils.logging import get_logging_level, register_to_factor_list

logger = setup_logger("trade.datamanager.requests", stream_log_level=get_logging_level())
register_to_factor_list("trade.datamanager.requests")

PandasData = Union[pd.Series, pd.DataFrame]

## Provided inputs that share option-spot vendor calendar exemptions.
_OPTION_PROVIDED_INPUT_NAMES = frozenset(
    {"option_spot_timeseries", "vol_timeseries", "greek_timeseries"}
)


@dataclass
class LoadRequest:
    """Request specifying which market data to load for model pricing."""

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

    ## Provided inputs
    spot_timeseries: Optional[SpotResult] = None
    forward_timeseries: Optional[ForwardResult] = None
    dividend_timeseries: Optional[DividendsResult] = None
    rates_timeseries: Optional[RatesResult] = None
    option_spot_timeseries: Optional[OptionSpotResult] = None
    vol_timeseries: Optional[VolatilityResult] = None
    greek_timeseries: Optional[GreekResultSet] = None

    def __post_init__(self) -> None:
        """Validate dates, option params, and optional pre-loaded timeseries."""
        ## If rt is True, set on_date to True and as_of to the current date.
        if self.rt:
            self.on_date = True
            self.as_of = datetime.now().date()

        ## If all dates are provided, raise an error. Either start_date and end_date or as_of must be provided.
        if all(date is not None for date in [self.start_date, self.end_date, self.as_of]):
            raise ValueError("Only pass start_date and end_date or as_of, not both.")

        ## If no dates are provided, raise an error. Either start_date and end_date or as_of must be provided.
        if all(date is None for date in [self.start_date, self.end_date, self.as_of]):
            raise ValueError("Either start_date and end_date or as_of must be provided.")

        ## If start_date and end_date are provided, check if start_date is earlier than end_date.
        if self.start_date is not None and self.end_date is not None:
            if pd.to_datetime(self.start_date) > pd.to_datetime(self.end_date):
                raise ValueError("start_date must be earlier than or equal to end_date.")

            ## Single-day window → point-in-time mode.
            if to_datetime(self.start_date).date() == to_datetime(self.end_date).date():
                self.as_of = to_datetime(self.start_date)
                self.start_date = None
                self.end_date = None
                self.on_date = True
                logger.info(
                    "Setting as_of to %s because start_date and end_date are the same date.",
                    self.as_of,
                )

        ## If as_of is provided, set start_date and end_date to the as_of date and set on_date to True.
        if self.as_of is not None:
            if self.start_date is not None or self.end_date is not None:
                raise ValueError("If as_of is provided, start_date and end_date must be None.")

            self.as_of = pd.to_datetime(self.as_of)
            self.start_date = self.as_of
            self.end_date = self.as_of
            self.on_date = True

            ## If rt is not True and the as_of date is the current date, set rt to True.
            if not self.rt and self.as_of.date() == datetime.now().date():
                self.rt = True
                logger.info("Setting rt to True because as_of date is the current date.")

        ## Option parameters
        option_params = [self.expiration, self.strike, self.right]
        option_params_str = ["expiration", "strike", "right"]
        if self.load_greek or self.load_vol or self.load_option_spot:
            for i, param in enumerate(option_params):
                if param is None:
                    raise ValueError(
                        f"{option_params_str[i]} must be provided when loading option data."
                    )

        if self.load_option_spot:
            if self.strike is None or self.right is None:
                raise ValueError("Strike and right must be provided when loading option spot data.")

        if self.model_price is None:
            self.model_price = ModelPrice.MIDPOINT
        
        ## If rt is True, set endpoint_source to QUOTE.
        if self.rt:
            self.endpoint_source = OptionSpotEndpointSource.QUOTE

        self._validate_provided_inputs()

    def _validate_provided_inputs(self) -> None:
        """Accept or discard caller-provided timeseries based on window completeness."""
        validatees = [
            (self.load_spot, self.spot_timeseries, "load_spot", "spot_timeseries"),
            (self.load_forward, self.forward_timeseries, "load_forward", "forward_timeseries"),
            (self.load_dividend, self.dividend_timeseries, "load_dividend", "dividend_timeseries"),
            (self.load_rates, self.rates_timeseries, "load_rates", "rates_timeseries"),
            (self.load_option_spot, self.option_spot_timeseries, "load_option_spot", "option_spot_timeseries"),
            (self.load_vol, self.vol_timeseries, "load_volatility", "vol_timeseries"),
            (self.load_greek, self.greek_timeseries, "load_greek", "greek_timeseries"),
        ]

        for load_flag, timeseries, load_name, timeseries_name in validatees:
            if load_flag and timeseries is not None:
                if self._is_missing_dates(
                    self.start_date,
                    self.end_date,
                    timeseries.timeseries,
                    timeseries_name=timeseries_name,
                ):
                    logger.info(
                        "Provided %s timeseries has missing dates. Consider reloading without "
                        "providing timeseries to fetch complete data.",
                        timeseries_name,
                    )
                    logger.info(
                        "Setting %s to None and %s to %s. We will attempt to load from source, "
                        "but if provided data is complete we will use it and skip loading from source.",
                        timeseries_name,
                        load_name,
                        load_flag,
                    )
                    setattr(self, timeseries_name, None)
                    setattr(self, load_name, load_flag)
                else:
                    logger.info("Using provided %s timeseries for loading.", timeseries_name)
                    setattr(self, load_name, False)

    def _vendor_checked_missing_set(self) -> Set[date]:
        """Return vendor-confirmed absent dates for the request's option contract."""
        if self.strike is None or self.right is None or self.expiration is None:
            return set()
        checked = resolve_checked_missing_dates_for_option_contract(
            symbol=self.symbol,
            strike=self.strike,
            right=self.right,
            expiration=self.expiration,
            valid_start=self.start_date,
            valid_end=self.end_date,
        )
        return {to_datetime(d).date() for d in checked}

    def _is_missing_dates(
        self,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        series: Optional[PandasData],
        *,
        timeseries_name: str,
    ) -> bool:
        """Return True when unexplained B-day gaps remain in the provided window.

        For option spot / vol / greeks, vendor-confirmed absent dates are subtracted
        before deciding incompleteness — same exemption cert uses at L2/L3.

        Args:
            start_date: Requested window start.
            end_date: Requested window end.
            series: Pre-loaded timeseries payload.
            timeseries_name: Attribute name for logging and option-artifact detection.

        Returns:
            True when gaps remain after option vendor exemptions.
        """
        if series is None or (hasattr(series, "empty") and series.empty):
            return True

        missing_dates = get_missing_dates(_start=start_date, _end=end_date, x=series)
        if not missing_dates:
            return False

        if timeseries_name in _OPTION_PROVIDED_INPUT_NAMES:
            checked_set = self._vendor_checked_missing_set()
            if checked_set:
                unexplained = [
                    d for d in missing_dates if to_datetime(d).date() not in checked_set
                ]
                if not unexplained:
                    logger.info(
                        "Provided %s gaps are vendor-confirmed absent dates only: %s",
                        timeseries_name,
                        sorted(checked_set),
                    )
                    return False
                missing_dates = unexplained

        logger.warning("Missing dates in provided data: %s", missing_dates)
        return True
