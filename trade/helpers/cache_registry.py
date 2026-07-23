"""Atomic read/write helpers for the CustomCache clear_dirs registry."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path


def registry_tmp_path(registry: Path) -> Path:
    """Unique tmp per writer so concurrent processes do not share clear_dirs.tmp."""
    return registry.parent / f"{registry.stem}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"


def write_cache_registry(registry: Path, data: dict, *, indent: int | None = None) -> None:
    """Atomically update a cache-dir registry via pid/uuid-scoped tmp + os.replace."""
    tmp = registry_tmp_path(registry)
    with tmp.open("w") as f:
        if indent is not None:
            json.dump(data, f, default=str, indent=indent)
        else:
            json.dump(data, f, default=str)
    tmp.replace(registry)
