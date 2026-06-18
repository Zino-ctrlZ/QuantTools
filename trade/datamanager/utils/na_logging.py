"""NA inspection logging for datamanager retrieval results.

Scans timeseries payloads returned by datamanager APIs for NaN/None values and
logs pretty-printed snapshots with dataclass parameters, cross-component context,
and environment metadata (including ``THETADATA_USE_V3``).

Comment density: orchestration

Processing Flow:
    1. ``@log_na_after_retrieval`` (or an explicit call) invokes ``log_retrieval_na``.
    2. Dispatch selects a type-specific logger (Result, ModelResultPack, TimeseriesData, etc.).
    3. Type-specific code collects NA datetime indices per component/field.
    4. ``_log_na_snapshots`` emits one pretty-printed WARNING per NA index to
       ``trade.datamanager.utils.model_na``.

Core Functions:
    get_na_log_context: Build shared environment context for NA snapshots.
    log_retrieval_na: Dispatch NA logging for any supported retrieval return type.
    log_model_result_pack_na: Log NA values across a loaded ModelResultPack.
    log_na_after_retrieval: Decorator that logs NA values after API calls.
"""

from __future__ import annotations

import functools
import os
import pprint
from dataclasses import fields
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import pandas as pd

from trade.datamanager.result import ModelResultPack, Result
from trade.helpers.Logging import setup_logger
from trade.datamanager.utils.logging import MODEL_NA_LOGGER_NAME, get_logging_level # noqa

na_logger = setup_logger(MODEL_NA_LOGGER_NAME, stream_log_level="CRITICAL")

## Field exclusion sets for param snapshots (timeseries values are logged separately).
##
## _RESULT_SKIP_FIELDS: timeseries and nested payloads are too large or redundant here.
## _PACKET_NESTED_RESULT_FIELDS: ModelResultPack already nests factor results; we only
##     want guiding fields (dividend_type, undo_adjust, etc.) at the packet level.
## _AT_INDEX_NA_SKIP_FIELDS: get_at_index hardcodes rates=np.nan and never loads IRX.
##     Logging that field would flood the file with false positives.
_RESULT_SKIP_FIELDS = frozenset({"timeseries", "model_input_keys", "dividend_result", "grid"})
_PACKET_NESTED_RESULT_FIELDS = frozenset(
    {"spot", "forward", "dividend", "rates", "option_spot", "vol", "greek", "time_to_load"}
)
_AT_INDEX_NA_SKIP_FIELDS = frozenset({"rates"})

F = TypeVar("F", bound=Callable[..., Any])


def get_na_log_context() -> Dict[str, Any]:
    """Return environment metadata attached to every NA log snapshot."""
    return {
        "thetadata_use_v3": os.environ.get("THETADATA_USE_V3"),
    }


## --- Serialization helpers ---


def _serialize_value(value: Any) -> Any:
    """Convert a scalar or enum to a JSON-friendly logging representation."""
    if isinstance(value, Enum):
        return {"name": value.name, "value": value.value}
    if value is None:
        return None
    if isinstance(value, (pd.Series, pd.DataFrame, dict)):
        return None
    if isinstance(value, Result):
        return None
    return value


def _serialize_dataclass_params(
    obj: Any,
    *,
    skip_fields: frozenset = _RESULT_SKIP_FIELDS,
) -> Dict[str, Any]:
    """Extract scalar dataclass parameters, serializing enums as name/value pairs."""
    if obj is None:
        return {}
    params: Dict[str, Any] = {}
    for field in fields(obj):
        if field.name in skip_fields:
            continue
        value = getattr(obj, field.name)
        if isinstance(value, Result):
            continue
        if isinstance(value, (pd.Series, pd.DataFrame)):
            continue
        serialized = _serialize_value(value)
        ## Enums serialize to {name, value}; plain scalars pass through as-is.
        ## Explicit None is kept so "field was unset" shows up in the log.
        if serialized is not None or value is None:
            params[field.name] = serialized if serialized is not None else None
        else:
            params[field.name] = value
    return params


## --- NA detection helpers ---


def _na_indices_in_timeseries(
    timeseries: Optional[Union[pd.Series, pd.DataFrame]],
) -> pd.DatetimeIndex:
    """Return datetime indices where any value in the timeseries is NA."""
    if timeseries is None or timeseries.empty:
        return pd.DatetimeIndex([])
    if isinstance(timeseries, pd.Series):
        return pd.DatetimeIndex(timeseries.index[timeseries.isna()])
    ## DataFrame: flag the row if any column is NA (e.g. one bad greek column).
    return pd.DatetimeIndex(timeseries.index[timeseries.isna().any(axis=1)])


def _normalize_log_value(value: Any) -> Any:
    """Convert pandas/numpy scalars to plain Python types for readable log output."""
    if value is None or (not isinstance(value, (str, bytes)) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Enum):
        return {"name": value.name, "value": value.value}
    return value


def _format_na_snapshot(snapshot: Dict[str, Any]) -> str:
    """Pretty-print an NA snapshot dict as multi-line log text."""
    return pprint.pformat(snapshot, indent=2, width=120, sort_dicts=False)


def _timeseries_values_at_index(
    timeseries: Optional[Union[pd.Series, pd.DataFrame]],
    index: pd.Timestamp,
) -> Optional[Dict[str, Any]]:
    """Return column/value mapping for a timeseries at a single datetime index."""
    if timeseries is None or index not in timeseries.index:
        return None
    row = timeseries.loc[index]
    if isinstance(timeseries, pd.Series):
        label = timeseries.name or "value"
        return {label: _normalize_log_value(row)}
    return {col: _normalize_log_value(row[col]) for col in row.index}


def _value_has_na(value: Any) -> bool:
    """Return True when a scalar, Series, or DataFrame contains NA."""
    if value is None:
        return True
    if isinstance(value, pd.Series):
        return bool(value.isna().any())
    if isinstance(value, pd.DataFrame):
        return bool(value.isna().any().any())
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(pd.isna(value))
    return False


## --- Core snapshot emitter ---


def _log_na_snapshots(
    *,
    log_label: str,
    params: Dict[str, Any],
    components: List[Tuple[str, Optional[Union[pd.Series, pd.DataFrame]], Dict[str, Any]]],
) -> None:
    """Log NA snapshots for one or more named timeseries components."""
    na_by_component: Dict[str, pd.DatetimeIndex] = {}
    all_na_indices: List[pd.Timestamp] = []

    for name, timeseries, _ in components:
        indices = _na_indices_in_timeseries(timeseries)
        if len(indices) == 0:
            continue
        na_by_component[name] = indices
        all_na_indices.extend(indices.tolist())

    if not all_na_indices:
        return

    unique_indices = sorted(set(all_na_indices))
    for index in unique_indices:
        na_sources = [name for name, indices in na_by_component.items() if index in indices]
        snapshot: Dict[str, Any] = {
            "environment": get_na_log_context(),
            "log_label": log_label,
            "params": params,
            "index": index.isoformat(),
            "na_sources": na_sources,
            "components": {},
        }
        ## Cross-factor debug: when spot is NA on a date, we also log rates, vol, etc.
        ## at that same index so you can see what else was loaded — not just the bad field.
        for name, timeseries, component_params in components:
            snapshot["components"][name] = {
                "params": component_params,
                "values_at_index": _timeseries_values_at_index(timeseries, index),
            }
        na_logger.warning("%s:\n%s", log_label, _format_na_snapshot(snapshot))


def _model_result_components(
    packet: ModelResultPack,
) -> List[Tuple[str, Optional[Result]]]:
    """Return named result components currently held in a ModelResultPack."""
    return [
        ("spot", packet.spot),
        ("dividend", packet.dividend),
        ("rates", packet.rates),
        ("forward", packet.forward),
        ("option_spot", packet.option_spot),
        ("vol", packet.vol),
        ("greek", packet.greek),
    ]


## --- Public loggers (one entry point per retrieval return shape) ---


def log_model_result_pack_na(packet: ModelResultPack) -> None:
    """Log NA values found in a ModelResultPack with full cross-factor context."""
    result_components = _model_result_components(packet)
    components: List[Tuple[str, Optional[Union[pd.Series, pd.DataFrame]], Dict[str, Any]]] = []
    for name, result in result_components:
        if result is None:
            continue
        components.append((name, result.timeseries, _serialize_dataclass_params(result)))

    packet_params = _serialize_dataclass_params(packet, skip_fields=_RESULT_SKIP_FIELDS | _PACKET_NESTED_RESULT_FIELDS)
    _log_na_snapshots(
        log_label="Model timeseries NA detected",
        params=packet_params,
        components=components,
    )


def log_result_na(
    result: Result,
    *,
    manager: str,
    method: str,
    **context: Any,
) -> None:
    """Log NA values for a single Result dataclass returned by a datamanager API."""
    params = {
        **get_na_log_context(),
        "manager": manager,
        "method": method,
        **context,
        **_serialize_dataclass_params(result),
    }
    components = [
        (
            manager,
            result.timeseries,
            _serialize_dataclass_params(result),
        )
    ]
    _log_na_snapshots(
        log_label=f"{manager}.{method} NA detected",
        params=params,
        components=components,
    )


def log_timeseries_data_na(
    data: Any,
    *,
    manager: str,
    method: str,
    symbol: Optional[str] = None,
    **context: Any,
) -> None:
    """Log NA values across populated fields in a MarketTimeseries TimeseriesData result."""
    params = {
        **get_na_log_context(),
        "manager": manager,
        "method": method,
        "symbol": symbol,
        **context,
    }
    components: List[Tuple[str, Optional[Union[pd.Series, pd.DataFrame]], Dict[str, Any]]] = []
    for field_name in ("spot", "chain_spot", "dividends", "dividend_yield", "split_factor", "rates"):
        value = getattr(data, field_name, None)
        if value is None or (hasattr(value, "empty") and value.empty):
            continue
        components.append((field_name, value, {"symbol": symbol, "field": field_name}))

    for key, value in (getattr(data, "additional_data", None) or {}).items():
        if value is None or (hasattr(value, "empty") and value.empty):
            continue
        components.append((f"additional_data.{key}", value, {"symbol": symbol, "field": key}))

    _log_na_snapshots(
        log_label=f"{manager}.{method} NA detected",
        params=params,
        components=components,
    )


def log_at_index_result_na(
    result: Any,
    *,
    manager: str,
    method: str,
    **context: Any,
) -> None:
    """Log NA values in a MarketTimeseries AtIndexResult point-in-time snapshot."""
    params = {
        **get_na_log_context(),
        "manager": manager,
        "method": method,
        "symbol": getattr(result, "sym", None),
        "date": getattr(result, "date", None),
        **context,
    }
    na_fields: List[str] = []
    field_values: Dict[str, Any] = {}
    for field in fields(result):
        if field.name in _AT_INDEX_NA_SKIP_FIELDS:
            continue
        value = getattr(result, field.name)
        if isinstance(value, pd.DataFrame):
            serialized = {col: _normalize_log_value(value[col].iloc[0]) for col in value.columns} if not value.empty else None
            if _value_has_na(value):
                na_fields.append(field.name)
            field_values[field.name] = serialized
        elif isinstance(value, pd.Series):
            serialized = {key: _normalize_log_value(val) for key, val in value.items()}
            if _value_has_na(value):
                na_fields.append(field.name)
            field_values[field.name] = serialized
        else:
            if _value_has_na(value):
                na_fields.append(field.name)
            field_values[field.name] = _normalize_log_value(value)

    if not na_fields:
        return

    ## AtIndexResult is a single-date snapshot (not a timeseries), so we log a flat
    ## field map instead of the per-index assembly used in _log_na_snapshots.
    snapshot = {
        "environment": get_na_log_context(),
        "log_label": f"{manager}.{method} NA detected",
        "params": params,
        "na_sources": na_fields,
        "values": field_values,
    }
    na_logger.warning("%s:\n%s", snapshot["log_label"], _format_na_snapshot(snapshot))


def log_pandas_series_na(
    series: pd.Series,
    *,
    manager: str,
    method: str,
    **context: Any,
) -> None:
    """Log NA values in a raw pandas Series returned by a datamanager API."""
    params = {
        **get_na_log_context(),
        "manager": manager,
        "method": method,
        **context,
    }
    _log_na_snapshots(
        log_label=f"{manager}.{method} NA detected",
        params=params,
        components=[(manager, series, {"series_name": series.name})],
    )


def log_scalar_na(
    value: Any,
    *,
    manager: str,
    method: str,
    **context: Any,
) -> None:
    """Log when a scalar retrieval result is NA."""
    if not _value_has_na(value):
        return
    snapshot = {
        "environment": get_na_log_context(),
        "log_label": f"{manager}.{method} NA detected",
        "params": {
            "manager": manager,
            "method": method,
            **context,
        },
        "value": _normalize_log_value(value),
    }
    na_logger.warning("%s:\n%s", snapshot["log_label"], _format_na_snapshot(snapshot))


## --- Dispatch and decorator entry points ---


def log_retrieval_na(
    result: Any,
    *,
    manager: str,
    method: str,
    **context: Any,
) -> None:
    """Dispatch NA logging for any supported datamanager retrieval return type."""
    if isinstance(result, ModelResultPack):
        log_model_result_pack_na(result)
        return
    if isinstance(result, Result):
        log_result_na(result, manager=manager, method=method, **context)
        return

    ## ponytail: lazy import; avoids circular import with market_data at module load
    from trade.datamanager.market_data import AtIndexResult, TimeseriesData

    if isinstance(result, TimeseriesData):
        log_timeseries_data_na(result, manager=manager, method=method, **context)
        return
    if isinstance(result, AtIndexResult):
        log_at_index_result_na(result, manager=manager, method=method, **context)
        return
    if isinstance(result, pd.Series):
        log_pandas_series_na(result, manager=manager, method=method, **context)
        return
    if isinstance(result, (int, float)) and not isinstance(result, bool):
        log_scalar_na(result, manager=manager, method=method, **context)


def log_na_after_retrieval(manager: str) -> Callable[[F], F]:
    """Decorator that inspects retrieval results for NA values before returning them."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)
            context: Dict[str, Any] = {}
            ## Most managers are per-symbol (args[0].symbol on SpotDataManager, etc.).
            ## MarketTimeseries passes sym as the first arg after self, so fall back to args[1].
            if args and hasattr(args[0], "symbol"):
                context["symbol"] = getattr(args[0], "symbol")
            if len(args) > 1 and isinstance(args[1], str):
                context.setdefault("symbol", args[1])
            log_retrieval_na(result, manager=manager, method=func.__name__, **context)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator
