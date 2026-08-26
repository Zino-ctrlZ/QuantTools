"""Backtesting.py Strategy wrappers for StrategyBase brains.

Builds a thin Strategy subclass that constructs a brain in ``init`` and
delegates open/close decisions in ``next``. Indicators marked
``plot_in_backtester=False`` are skipped when registering ``Strategy.I`` so
sparse / late-starting feature series cannot delay the first ``next`` call via
backtesting.py indicator warmup.

Entry features that never go through ``Strategy.I`` are instead packed into
the ``buy``/``sell`` ``tag`` dict and exploded onto ``_trades`` after run.

Core Functions:
    make_bt_wrapper: Build the backtesting.py Strategy subclass for a brain.
    build_entry_tag: Pack ``signal_id`` plus additional entry info into ``tag``.
    explode_trade_tags: Restore ``Tag`` to ``signal_id`` and emit ``Entry_*`` columns.

Processing Flow:
    1. Wrapper ``init`` constructs the brain and registers plottable ``I()`` series.
    2. Wrapper ``next`` calls ``open_action``, then ``buy``/``sell`` with a tag dict.
    3. ``PTBacktester.run`` explodes each dataset's ``_trades.Tag`` after ``Backtest.run``.

Comment density: orchestration
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Set, TYPE_CHECKING

import pandas as pd

from .data import PTDataset

if TYPE_CHECKING:
    from ._strategy import StrategyBase

REQUIRED = object()
TAG_SIGNAL_ID_KEY = "signal_id"


def build_entry_tag(
    signal_id: Any,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the ``buy``/``sell`` tag payload for one entry.

    ``signal_id`` is always written last so it cannot be overwritten by
    ``additional_on_entry_info``. After ``explode_trade_tags``, this value is
    restored onto the ``Tag`` column for downstream consumers.

    Args:
        signal_id: Canonical signal identifier for the opening order.
        extra_info: Optional mapping from ``additional_on_entry_info``.

    Returns:
        Tag dict containing ``signal_id`` plus any extra entry fields.

    Raises:
        TypeError: If ``extra_info`` is not ``None`` and not a dict.

    Examples:
        >>> build_entry_tag("AAPL20200102SHORT", {"zscore": 2.1})
        {'zscore': 2.1, 'signal_id': 'AAPL20200102SHORT'}
    """
    if extra_info is None:
        payload: Dict[str, Any] = {}
    elif isinstance(extra_info, dict):
        payload = dict(extra_info)
    else:
        raise TypeError(
            f"extra_info must be a dict or None, got {type(extra_info).__name__}"
        )
    payload[TAG_SIGNAL_ID_KEY] = signal_id
    return payload


def _entry_feature_column(name: Any) -> str:
    """Return an ``Entry_`` column name without double-prefixing.

    Args:
        name: Feature key from the tag dict.

    Returns:
        Column name matching backtesting.py ``Entry_*`` convention.
    """
    text = str(name)
    if text.startswith("Entry_"):
        return text
    return f"Entry_{text}"


def _tag_extra_keys(tags: Iterable[Any]) -> List[str]:
    """Collect extra tag keys in first-seen order, excluding ``signal_id``.

    Args:
        tags: Raw ``Tag`` values, some of which may be dict payloads.

    Returns:
        Extra feature keys to explode into ``Entry_*`` columns.
    """
    keys: List[str] = []
    seen: Set[str] = set()
    for tag in tags:
        if not isinstance(tag, dict):
            continue
        for key in tag.keys():
            if key == TAG_SIGNAL_ID_KEY or key in seen:
                continue
            seen.add(key)
            keys.append(key)
    return keys


def explode_trade_tags(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Explode dict ``Tag`` payloads into ``Entry_*`` columns.

    Backtesting.py stores ``buy(tag=...)`` as a single ``Tag`` column. This
    helper restores ``Tag`` to ``signal_id`` (backward compatible with
    ``"SHORT" in row["Tag"]`` consumers) and writes remaining keys as
    ``Entry_<feature>``. Existing ``Entry_*`` columns from ``Strategy.I`` are
    left unchanged.

    Idempotent: frames whose ``Tag`` values are not dicts are returned as-is.

    Args:
        trades_df: ``stats['_trades']`` frame from ``Backtest.run``.

    Returns:
        Copy with ``Tag`` as ``signal_id`` and extra ``Entry_*`` columns.

    Examples:
        >>> df = pd.DataFrame({"Tag": [{"signal_id": "x", "zscore": 1.2}]})
        >>> exploded = explode_trade_tags(df)
        >>> exploded.loc[0, "Tag"], exploded.loc[0, "Entry_zscore"]
        ('x', 1.2)
    """
    if trades_df.empty or "Tag" not in trades_df.columns:
        return trades_df

    tags = trades_df["Tag"]
    if not any(isinstance(tag, dict) for tag in tags):
        return trades_df

    out = trades_df.copy()
    extra_keys = _tag_extra_keys(tags)

    ## Restore Tag to the opening signal_id so options/WFA code that does
    ## `"SHORT" in trade["Tag"]` or `signal_id = trade["Tag"]` keeps working.
    out["Tag"] = [tag.get(TAG_SIGNAL_ID_KEY) if isinstance(tag, dict) else tag for tag in tags]

    for key in extra_keys:
        column = _entry_feature_column(key)
        ## Keep Strategy.I snapshots when both sources expose the same name.
        if column in out.columns:
            continue
        out[column] = [tag.get(key) if isinstance(tag, dict) else None for tag in tags]
    return out


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
    - next() packs ``signal_id`` plus ``additional_on_entry_info`` into ``tag``

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
        """Construct the brain and optionally register plottable indicators."""
        # Build kwargs for brain init
        brain_kwargs = {}
        for k in brain_cls.bt_params.keys():
            val = getattr(self, k)
            if val is REQUIRED:
                raise ValueError(f"{wrapper_name}: parameter '{k}' is REQUIRED but was not set.")
            brain_kwargs[k] = val
        brain_kwargs["ticker"] = getattr(self, "_name", None)
        if verbose:
            print("Brain Kwargs: ", brain_kwargs)
            print("Saved Name: ", getattr(self, "_name", None))

        # Build dataset for the brain
        ds = dataset_factory(self.data.df)

        # Pass start_date directly (Timestamp or None)
        self.brain = brain_cls(
            data=ds,
            start_trading_date=self.start_date,
            ticker=brain_kwargs.pop("ticker"),
            **brain_kwargs,
        )

        ## Only register indicators flagged for backtester plotting. Sparse /
        ## late-starting series (e.g. setup features with long leading NaNs)
        ## must stay out of Strategy.I — backtesting.py delays first next()
        ## until every non-scatter I() indicator has a non-NaN value.
        if plot_indicators:
            for ind in getattr(self.brain, "indicators", {}).values():
                if not getattr(ind, "plot_in_backtester", True):
                    continue
                try:
                    self.__setattr__(
                        ind.name,
                        self.I(
                            lambda s=ind.values: s,
                            name=ind.name,
                            overlay=ind.overlay,
                        ),
                    )
                except Exception:
                    pass

    def _next(self):
        """Evaluate open/close decisions for the latest bar."""
        date = self.data.index[-1]
        open_decision = self.brain.should_open(date=date)
        if verbose:
            print(f"Open Decision: {open_decision.ok}, Date: {date}")
        if open_decision.ok:
            if verbose:
                print(f"Opening position on {date} at price {self.data.Close[-1]}")
                print(f"Info: {self.brain.info_on_date(date=date)}")
            if open_decision.side not in (1, -1):
                raise ValueError(f"Invalid side in open_decision: {open_decision.side}")
            ## open_action first so additional_on_entry_info sees post-open
            ## fields (e.g. atr_stop). Tag dict carries signal_id plus those
            ## extras; explode_trade_tags restores Tag to signal_id after run.
            self.brain.open_action(
                date=date,
                signal_id=open_decision.signal_id,
                side=open_decision.side,
                entry_price=self.data.Close[-1],
            )
            tag = build_entry_tag(
                open_decision.signal_id,
                self.brain.additional_on_entry_info(date=date),
            )
            if open_decision.side == 1:
                if verbose:
                    print("Going LONG")
                self.buy(tag=tag)
            else:
                if verbose:
                    print("Going SHORT")
                self.sell(tag=tag)

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
