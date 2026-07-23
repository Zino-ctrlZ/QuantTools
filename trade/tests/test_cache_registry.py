"""Tests for atomic cache-dir registry writes (clear_dirs.json)."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pytest

from trade.helpers.cache_registry import write_cache_registry


def _registry_writer(registry_path: str, cache_key: str) -> str:
    registry = Path(registry_path)
    raw = registry.read_text()
    data = json.loads(raw) if raw.strip() else {}
    data[cache_key] = "2099-01-01"
    write_cache_registry(registry, data)
    return cache_key


def test_write_cache_registry_uses_unique_tmp_files(tmp_path: Path) -> None:
    registry = tmp_path / "clear_dirs.json"
    registry.write_text("{}")

    write_cache_registry(registry, {"a": "2099-01-01"})
    write_cache_registry(registry, {"a": "2099-01-01", "b": "2099-01-01"})

    assert json.loads(registry.read_text()) == {
        "a": "2099-01-01",
        "b": "2099-01-01",
    }
    assert list(tmp_path.glob("clear_dirs.*.tmp")) == []


def test_concurrent_registry_writes_do_not_raise(tmp_path: Path) -> None:
    registry = tmp_path / "clear_dirs.json"
    registry.write_text("{}")

    keys = [f"/cache/dir-{i}" for i in range(4)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(_registry_writer, str(registry), key) for key in keys
        ]
        for future in as_completed(futures):
            future.result()

    data = json.loads(registry.read_text())
    assert data
    assert not list(tmp_path.glob("clear_dirs.*.tmp"))
