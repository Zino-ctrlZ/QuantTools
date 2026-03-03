from datetime import date, datetime, time
from enum import Enum
from typing import Dict, Optional, Any, Union
from .._enums import Interval, ArtifactType, SeriesId

DATE_HINT = Union[datetime, str]
def _norm_str(x: str) -> str:
    return x.strip().upper()


def _safe_part(x: Optional[str]) -> str:
    return x if x not in (None, "", "None") else "-"

def _format_value(v: Any) -> str:
    """
    Keep it simple + deterministic.
    """
    if v is None:
        return "-"
    if isinstance(v, Enum):
        return str(v.value)
    if isinstance(v, str):
        return _norm_str(v)
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int,)):
        return str(v)
    if isinstance(v, float):
        # avoid 0.30000000000004 style keys
        return f"{v:.12g}"
    if isinstance(v, datetime):
        # stable, compact. (no tz handling by design here)
        return v.strftime("%Y%m%dT%H%M%S")
    if isinstance(v, date):
        return v.strftime("%Y%m%d")

    if isinstance(v, time):
        return v.strftime("%H%M%S")
    return str(v)


def construct_cache_key(
    symbol: str,
    interval: Optional[Interval],
    artifact_type: ArtifactType,
    series_id: SeriesId,
    **extra_parts: Any,
) -> str:
    """Constructs deterministic cache key from symbol, interval, artifact type, series ID, and extra parts."""

    if series_id in (SeriesId.AT_TIME, SeriesId.SNAPSHOT):
        assert "time" in extra_parts, "time must be provided for at_time or snapshot series_id"
        assert "date" in extra_parts, "date must be provided for at_time or snapshot series_id"
        assert isinstance(extra_parts["time"], time), "time must be a time object"
        assert isinstance(extra_parts["date"], date), "date must be a date object"

    parts = [
        f"symbol:{_norm_str(symbol)}",
        f"interval:{_format_value(interval)}",
        f"artifact_type:{artifact_type.value}",
        f"series_id:{series_id.value}",
    ]

    for k in sorted(extra_parts.keys()):
        parts.append(f"{k}:{_format_value(extra_parts[k])}")

    return "|".join(parts)

def _parse_cache_key(key: str) -> Dict[str, str]:
    """Parses a pipe-delimited cache key into a dictionary of key-value pairs."""
    parts = key.split("|")
    result = {}
    for part in parts:
        k, v = part.split(":", 1)
        result[k] = v
    return result