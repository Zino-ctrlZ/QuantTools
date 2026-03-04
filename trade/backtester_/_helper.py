
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING
import pandas as pd
from .data import PTDataset

if TYPE_CHECKING:
    from ._strategy import StrategyBase

REQUIRED = object()


def make_bt_wrapper(
    brain_cls: type["StrategyBase"],
    *,
    name: Optional[str] = None,
    param_overrides: Optional[Dict[str, Any]] = None,
    start_date: Optional[str] = None,
    dataset_factory: Optional[Callable[[Any], Any]] = None,
    plot_indicators: bool = True,
    verbose: bool = False,
):
    """
    Generates a Backtesting.py Strategy wrapper class for a given brain strategy.

    - brain_cls.bt_params declares which params are exposed to wrapper/optimize
    - wrapper exposes those params as class attributes (so Backtesting.optimize can mutate them)
    - wrapper builds brain instance in init() and delegates decisions in next()

    dataset_factory:
      - function that takes the backtesting df (self.data.df) and returns whatever your brain expects.
      - if None, passes the df directly.
      - Example: dataset_factory=lambda df: PTDataset(data=df, name=None)
    """

    # Late import keeps this file importable even when backtesting isn't installed in some contexts
    from backtesting import Strategy  # type: ignore

    param_overrides = param_overrides or {}
    dataset_factory = dataset_factory or (lambda df: PTDataset(data=df, name=None))

    # Build class attributes for exposed params (defaults + overrides)
    class_attrs: Dict[str, Any] = {}
    for k, v in brain_cls.bt_params.items():
        class_attrs[k] = param_overrides.get(k, v)

    # Always define start_date on the wrapper class
    class_attrs["start_date"] = pd.Timestamp(start_date) if start_date is not None else None

    wrapper_name = name or f"BT_{brain_cls.__name__}"

    def _init(self):
        # Build kwargs for brain init
        brain_kwargs = {}
        for k in brain_cls.bt_params.keys():
            val = getattr(self, k)
            if val is REQUIRED:
                raise ValueError(f"{wrapper_name}: parameter '{k}' is REQUIRED but was not set.")
            brain_kwargs[k] = val

        # Build dataset for the brain
        ds = dataset_factory(self.data.df)

        # Pass start_date directly (Timestamp or None)
        ticker_name = getattr(ds, "name", None) or getattr(ds, "ticker", None) or "NA"
        self.brain = brain_cls(
            data=ds,
            start_trading_date=self.start_date,
            ticker=ticker_name,
            **brain_kwargs,
        )

        # Optional plotting
        if plot_indicators:
            for ind in getattr(self.brain, "indicators", {}).values():
                try:
                    self.__setattr__(
                        ind.name,
                        self.I(lambda s=ind.values: s, name=ind.name, overlay=ind.overlay),
                    )
                except Exception:
                    pass

    def _next(self):
        date = self.data.index[-1]
        open_decision = self.brain.should_open(date=date)
        if open_decision.ok:
            if verbose:
                print(f"Opening position on {date} at price {self.data.Close[-1]}")
                print(f"Info: {self.brain.info_on_date(date=date)}")
            if open_decision.side == 1:
                if verbose:
                    print("Going LONG")
                self.buy()
            elif open_decision.side == -1:
                if verbose:
                    print("Going SHORT")
                self.sell()
            else:
                raise ValueError(f"Invalid side in open_decision: {open_decision.side}")
            self.brain.open_action(
                date=date,
                signal_id=open_decision.signal_id,
                side=open_decision.side,
                entry_price=self.data.Close[-1],
            )

        else:
            close_decision = self.brain.should_close(date=date)
            if not close_decision.ok:
                return
            if verbose:
                print(f"Closing position on {date} at price {self.data.Close[-1]}")
                print(f"Info: {self.brain.info_on_date(date=date)}")
            self.position.close()
            self.brain.close_action(date=date)

    # Create the Strategy subclass dynamically
    Wrapper = type(
        wrapper_name,  ## New class name
        (Strategy,),  ## Base classes
        {  ## Class attributes
            **class_attrs,
            "brain_cls": brain_cls,
            "init": _init,
            "next": _next,
        },
    )

    return Wrapper
