"""Live-only pipeline checkpoints for CREATE / limits diagnostics.

Comment density: orchestration.

Emits compact scalar snapshots along the market-data → limits path so NaN /
missing current-date rows are searchable without dumping DataFrames. Disabled
unless the process-global live flag is on (or an explicit ``enabled=True``
override is passed).

Core Functions:
    set_live_diag_enabled: Flip the process-global live-diag gate.
    is_live_diag_enabled: Read the live-diag gate.
    get_row_at_date: Safe current-date (or explicit date) row lookup.
    fields_at_current_date: Extract named columns from that row.
    log_live_checkpoint: Emit one ``live_chk`` log line.

Processing Flow:
    1. Live entry calls ``set_live_diag_enabled(True)``.
    2. Call sites pass a stage id + scalar ``data`` (or a frame via
       ``fields_at_current_date``).
    3. Helper no-ops when the gate is off; otherwise logs stage, ids, flags, data.

Usage:
    >>> from EventDriven.riskmanager.live_diag import set_live_diag_enabled, log_live_checkpoint
    >>> set_live_diag_enabled(True)
    >>> log_live_checkpoint("limits_sized", trade_id="&L:A..", date="2026-07-14", data={"delta": float("nan")})
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime
from typing import Any, Dict, Iterable, Mapping, Optional, Union

import pandas as pd

from trade.helpers.Logging import setup_logger

logger = setup_logger("EventDriven.riskmanager.live_diag", stream_log_level="INFO")

## Process-global gate: False for backtests; flipped True when a live session starts.
_LIVE_DIAG_ENABLED: bool = False

## Columns whose values get an ok/nan/inf/missing flag in the log line.
_FLAG_KEYS = frozenset(
    {
        "delta",
        "gamma",
        "vega",
        "theta",
        "rho",
        "vol",
        "iv",
        "midpoint",
        "price",
        "option_price",
        "closebid",
        "closeask",
        "delta_lmt",
        "quantity",
        "s",
        "r",
        "y",
    }
)

## Greeks / quotes at leg_loaded (extended) vs earlier pipeline stages.
LEG_LOADED_FIELD_COLUMNS = (
    "Delta",
    "Gamma",
    "Vega",
    "Theta",
    "Vol",
    "Midpoint",
    "Closebid",
    "Closeask",
)

## Quote + delta columns for net/load/trade checkpoints (unchanged from original leg_loaded set).
OPTION_TRADE_FIELD_COLUMNS = (
    "Delta",
    "Vol",
    "Midpoint",
    "Closebid",
    "Closeask",
)

## Raw datamanager artifact columns (lowercase) before the joined frame.
GREEK_ARTIFACT_COLUMNS = ("delta", "gamma", "vega", "theta", "rho")
OPTION_SPOT_ARTIFACT_COLUMNS = ("midpoint", "closebid", "closeask")
VOL_ARTIFACT_COLUMNS = ("iv",)

DateLike = Union[datetime, date, str, pd.Timestamp]


def set_live_diag_enabled(enabled: bool) -> None:
    """Enable or disable live diagnostic checkpoints process-wide.

    Args:
        enabled: When ``True``, ``log_live_checkpoint`` emits; when ``False``, no-op.
    """
    global _LIVE_DIAG_ENABLED
    _LIVE_DIAG_ENABLED = bool(enabled)
    logger.info("live_diag enabled=%s", _LIVE_DIAG_ENABLED)


def enable_live_diag() -> None:
    """Turn on live diagnostic checkpoints."""
    set_live_diag_enabled(True)


def disable_live_diag() -> None:
    """Turn off live diagnostic checkpoints."""
    set_live_diag_enabled(False)


def is_live_diag_enabled() -> bool:
    """Return whether live diagnostic checkpoints are enabled.

    Returns:
        ``True`` when the process-global live-diag gate is on.
    """
    return _LIVE_DIAG_ENABLED


def _resolve_current_date(date_value: Optional[DateLike] = None) -> pd.Timestamp:
    """Normalize to a tz-naive midnight Timestamp (defaults to NY calendar today).

    Args:
        date_value: Explicit as-of date, or ``None`` for ``ny_now().date()``.

    Returns:
        Normalized pandas Timestamp at midnight.
    """
    ## Lazy import: helper pulls heavy deps (QuantLib/openbb); only need it when logging.
    from trade.helpers.helper import ny_now, to_datetime

    ## Live always reasons about "today" in NY; callers may still pass req.date.
    if date_value is None:
        date_value = ny_now().date()
    return pd.Timestamp(to_datetime(date_value)).tz_localize(None).normalize()


def get_row_at_date(
    data: Optional[pd.DataFrame],
    date_value: Optional[DateLike] = None,
) -> Optional[pd.Series]:
    """Safely fetch one row for the as-of date without raising on missing index.

    Tries Timestamp, date, and date-equality masks. Duplicate index hits take the
    last row. Returns ``None`` when the frame is empty or the date is absent —
    never uses bare ``df.loc[date]`` that can KeyError.

    Args:
        data: Option/position timeseries DataFrame.
        date_value: As-of date; defaults to current NY calendar date.

    Returns:
        Single-row ``Series``, or ``None`` if not found.
    """
    if data is None or getattr(data, "empty", True):
        return None

    from trade.helpers.helper import to_datetime

    target = _resolve_current_date(date_value)
    target_date = target.date()
    index = data.index

    row: Optional[Union[pd.Series, pd.DataFrame]] = None
    try:
        if target in index:
            row = data.loc[target]
        elif target_date in index:
            row = data.loc[target_date]
        else:
            ## String / mixed indexes: compare on calendar date without KeyError
            try:
                mask = pd.DatetimeIndex(to_datetime(index)).normalize() == target
            except (TypeError, ValueError):
                try:
                    mask = pd.Index([to_datetime(x).date() for x in index]) == target_date
                except (TypeError, ValueError):
                    return None
            if not bool(mask.any()):
                return None
            row = data.loc[mask]
    except KeyError:
        return None

    if row is None:
        return None
    ## Duplicate labels → DataFrame; keep last observation for that calendar day
    if isinstance(row, pd.DataFrame):
        if row.empty:
            return None
        return row.iloc[-1]
    return row


def fields_at_current_date(
    data: Optional[pd.DataFrame],
    columns: Iterable[str],
    date_value: Optional[DateLike] = None,
) -> Dict[str, Any]:
    """Extract named columns from the current-date row, with a found flag.

    Missing columns or a missing row yield ``None`` values. Column names are
    matched case-insensitively against the frame.

    Args:
        data: Option/position timeseries DataFrame.
        columns: Column names to pull (e.g. ``Delta``, ``Midpoint``).
        date_value: As-of date; defaults to current NY calendar date.

    Returns:
        Dict of column → value, plus ``row_found`` (bool).
    """
    cols = list(columns)
    if isinstance(data, pd.Series):
        series = data
        row = get_row_at_date(series.to_frame(name=series.name or "value"), date_value)
        if row is None:
            out = {c: None for c in cols}
            out["row_found"] = False
            return out
        name = series.name or "value"
        out = {}
        for col in cols:
            if str(col).lower() == str(name).lower():
                out[col] = row.get(name, row.iloc[0] if len(row) else None)
            else:
                out[col] = None
        out["row_found"] = True
        return out

    row = get_row_at_date(data, date_value)
    if row is None:
        out: Dict[str, Any] = {c: None for c in cols}
        out["row_found"] = False
        return out

    ## Case-insensitive map so Delta/delta both resolve after capitalize passes
    lower_map = {str(name).lower(): name for name in row.index}
    out = {}
    for col in cols:
        actual = lower_map.get(str(col).lower())
        out[col] = None if actual is None else row.get(actual)
    out["row_found"] = True
    return out


def log_option_data_checkpoint(
    stage: str,
    *,
    opttick: str,
    data: Optional[pd.DataFrame] = None,
    date: Optional[DateLike] = None,
    columns: Iterable[str] = OPTION_TRADE_FIELD_COLUMNS,
    note: str = "",
    **extra: Any,
) -> None:
    """Emit one option-data checkpoint with standard quote/delta columns.

    Args:
        stage: Stable stage id (e.g. ``option_data_windowed``).
        opttick: Option tick being processed.
        data: Joined option timeseries frame.
        date: As-of date for row lookup.
        columns: Columns to pull from the current-date row.
        note: Optional free-text qualifier.
        **extra: Additional scalar payload keys merged into ``data``.
    """
    payload: Dict[str, Any] = {
        "opttick": opttick,
        **fields_at_current_date(data, columns, date_value=date),
        **extra,
    }
    log_live_checkpoint(stage, trade_id=opttick, date=date, data=payload, note=note)


def _value_flag(value: Any) -> str:
    """Classify a scalar for compact health flags in the log line.

    Args:
        value: Candidate numeric or missing value.

    Returns:
        One of ``ok``, ``nan``, ``inf``, ``missing``, ``nonnum``.
    """
    if value is None:
        return "missing"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nonnum"
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf"
    return "ok"


def _compact_payload(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop frame-like values so checkpoint logs stay scalar-only.

    Args:
        data: Caller-supplied payload.

    Returns:
        Copy with DataFrame/Series values replaced by shape markers.
    """
    out: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, (pd.DataFrame, pd.Series)):
            out[key] = f"<{type(value).__name__} shape={getattr(value, 'shape', '?')}>"
        else:
            out[key] = value
    return out


def log_live_checkpoint(
    stage: str,
    *,
    trade_id: Optional[str] = None,
    date: Optional[DateLike] = None,
    data: Optional[Mapping[str, Any]] = None,
    note: str = "",
    level: int = logging.INFO,
    enabled: Optional[bool] = None,
) -> None:
    """Log one live pipeline checkpoint, or no-op when diagnostics are off.

    Args:
        stage: Stable stage id (e.g. ``leg_loaded``, ``limits_sized``).
        trade_id: Position / trade id when known.
        date: As-of date shown in the log (defaults to current NY date).
        data: Scalar payload (deltas, prices, etc.). Frames are compacted.
        note: Optional free-text qualifier (e.g. ``post_size``).
        level: Logging level for the line.
        enabled: Override the global gate; ``None`` uses ``is_live_diag_enabled``.
    """
    if enabled is None:
        enabled = is_live_diag_enabled()
    if not enabled:
        return

    payload = _compact_payload(dict(data or {}))
    as_of = _resolve_current_date(date)
    flags = {
        key: _value_flag(value)
        for key, value in payload.items()
        if str(key).lower() in _FLAG_KEYS
    }
    logger.log(
        level,
        "live_chk stage=%s trade_id=%s date=%s %s flags=%s data=%s",
        stage,
        trade_id,
        as_of.strftime("%Y-%m-%d"),
        note,
        flags,
        payload,
    )
