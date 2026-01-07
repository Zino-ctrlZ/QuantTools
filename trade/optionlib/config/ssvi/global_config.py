from typing import ClassVar, Optional
from dataclasses import dataclass as stdlib_dataclass, field as stdlib_field
from trade.optionlib.config.types import VolSide, DivType, VolType


@stdlib_dataclass
class SSVIGlobalConfig:
    """
    Singleton class for global configuration of the SSVI model.
    There will only be one instance of this class. Whether you create a new instance or use the instance() method,
    you will always get the same object.

    Intention is to provide a centralized configuration for the SSVI model that can be easily accessed and modified.
    """

    __SINGLETON__: ClassVar[bool] = True
    _CREATED: ClassVar[Optional["SSVIGlobalConfig"]] = None
    _initialized: ClassVar[bool] = False
    """
    Global configuration for SSVI model.
    Attributes:
        vol_side (VolSide): Which side of the volatility surface to model ('call', 'put', 'otm').
        div_type (DivType): Type of dividends to consider ('discrete', 'continuous').
        vol_type (VolType): Type of volatility to use for calibration ('bs', 'binomial').
        N (int): Number of steps for binomial model.
        iteration (int): Number of iterations for refining European equivalent volatilities.
    """
    vol_side: VolSide = stdlib_field(default=VolSide.OTM)
    div_type: DivType = stdlib_field(default=DivType.DISCRETE)
    vol_type: VolType = stdlib_field(default=VolType.BINOMIAL)
    N: int = stdlib_field(default=250)
    iteration: int = stdlib_field(default=2)
    chunk_size: int = stdlib_field(default=5000)
    model_iterations: int = stdlib_field(default=50_000)
    save_cache: bool = stdlib_field(default=True)
    force_calc: bool = stdlib_field(default=False)
    overwrite_existing: bool = stdlib_field(default=False)
    fit_all_sides: bool = stdlib_field(default=False)

    def __new__(cls, *args, **kwargs):
        if cls.__SINGLETON__ and cls._CREATED is not None:
            return cls._CREATED
        instance = super().__new__(cls)
        cls._CREATED = instance
        return instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

    @classmethod
    def instance(cls):
        """Get the singleton instance of the SSVIGlobalConfig."""
        if cls._CREATED is None:
            cls._CREATED = cls()
        return cls._CREATED

    @classmethod
    def reset(cls):
        """Reset the singleton instance (for testing or reconfiguration)."""
        cls._CREATED = None

    def __setattr__(self, name, value):
        ## Ensure enum values are valid
        enum_names = {
            "vol_side": VolSide,
            "div_type": DivType,
            "vol_type": VolType,
        }
        if name in enum_names:
            if isinstance(value, str):
                try:
                    value = enum_names[name](value)
                except ValueError as e:
                    raise ValueError(
                        f"Invalid value '{value}' for {name}. Allowed values are: {[e.value for e in enum_names[name]]}"
                    ) from e
            elif not isinstance(value, enum_names[name]):
                raise ValueError(f"{name} must be an instance of {enum_names[name]}")
        super().__setattr__(name, value)
