"""Enforce the ``bt_params`` contract shared by strategies and portfolio books.

Lifted verbatim from ``StrategyBase.__init_subclass__`` and ``get_bt_params``
so ``StrategyBase`` and ``SystematicMultiAsset`` validate params the same way.
Defaults reproduce today's StrategyBase behavior exactly; a class opts into
anything looser through a class attribute.

Core Classes:
    ParamContractMixin: Class-definition check of ``bt_params`` against ``__init__``.

Risk/Assumptions:
    Deliberately defines no ``bt_params`` default. A subclass without one must
    keep raising, and an inherited ``{}`` would silently let it through.
"""

from __future__ import annotations

import inspect
from typing import Any, ClassVar, Dict, List, Tuple


class ParamContractMixin:
    """Validate ``bt_params`` against ``__init__`` when a subclass is defined.

    Class attributes:
        _required_init_args: Args every concrete ``__init__`` must accept, e.g.
            ``("data", "start_trading_date", "ticker", "tplusn")`` for strategies.
        _contract_root: Set True in a base class's own body to skip checking that
            base itself (StrategyBase used ``cls is StrategyBase`` for this).
        _kwargs_satisfies_contract: When True, an ``__init__`` with ``**kwargs``
            counts as accepting every required arg and ``bt_params`` key, so a
            subclass only spells out what it adds. False keeps today's strict rule.
    """

    _required_init_args: ClassVar[Tuple[str, ...]] = ()
    _kwargs_satisfies_contract: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate that subclasses properly define bt_params and __init__ signature.

        Raises:
            TypeError: If bt_params is missing, not a dict, or contains keys not in __init__.
        """
        super().__init_subclass__(**kwargs)
        must_have_in_init = list(cls._required_init_args)

        ## Bases mark themselves; abstract classes are checked once made concrete.
        if cls.__dict__.get("_contract_root", False) or inspect.isabstract(cls):
            return

        # 1) bt_params must exist and be a dict
        if not hasattr(cls, "bt_params"):
            raise TypeError(f"{cls.__name__} must define class attribute bt_params (dict).")
        if not isinstance(cls.bt_params, dict):
            raise TypeError(f"{cls.__name__}.bt_params must be a dict of param_name -> default/REQUIRED.")

        # 2) every key in bt_params must be accepted by __init__
        params = inspect.signature(cls.__init__).parameters
        has_var_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        ## Only opted-in classes (books) let forwarded **kwargs reach a parent that
        ## already passed this check. Strategies keep the strict rule.
        if cls._kwargs_satisfies_contract and has_var_kwargs:
            accepted = set(params) | set(must_have_in_init) | set(cls.bt_params)
        else:
            accepted = set(params)

        # Enforce that required args are in __init__
        for req in must_have_in_init:
            if req not in accepted:
                raise TypeError(f"{cls.__name__}.__init__ must accept parameter '{req}'.")

        # Enforce params are in bt_params
        for p in params.values():
            if p.name in must_have_in_init + ["self", "kwargs", "args"]:
                continue
            if p.name not in cls.bt_params:
                raise TypeError(
                    f"{cls.__name__}.__init__ has parameter '{p.name}' which is not listed in {cls.__name__}.bt_params."
                )

        for k in cls.bt_params.keys():
            if k not in accepted and k not in must_have_in_init:
                raise TypeError(
                    f"{cls.__name__}.bt_params includes '{k}', but {cls.__name__}.__init__ "
                    f"does not accept '{k}' (and has no **kwargs)."
                )

    def get_bt_params(self) -> Dict[str, Any]:
        """Return the current backtest parameter values for this instance.

        Returns:
            Current values keyed by ``bt_params``.

        Raises:
            TypeError: If ``bt_params`` is not a dictionary.
            AttributeError: If any declared backtest parameter is missing on the instance.
        """
        bt_params = getattr(type(self), "bt_params", None)
        if not isinstance(bt_params, dict):
            raise TypeError(f"{type(self).__name__}.bt_params must be a dict.")

        current_params: Dict[str, Any] = {}
        missing_keys: List[str] = []
        for key in bt_params.keys():
            if not hasattr(self, key):
                missing_keys.append(key)
                continue
            current_params[key] = getattr(self, key)

        if missing_keys:
            raise AttributeError(f"{type(self).__name__} is missing backtest parameter attribute(s): {missing_keys}.")

        return current_params
