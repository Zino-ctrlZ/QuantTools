"""Tests for quote 472 CSV logging under GEN_CACHE_PATH."""

from __future__ import annotations

import csv
from pathlib import Path

from trade.datamanager.utils.quote_472_log import (
    QUOTE_472_CSV_COLUMNS,
    append_quote_472_row,
    quote_472_csv_path,
)


def test_append_quote_472_row_writes_metadata_and_url(tmp_path: Path, monkeypatch) -> None:
    """CSV rows include option metadata, missing date, and request URL."""
    monkeypatch.setattr("trade.GEN_CACHE_PATH", tmp_path)
    path = append_quote_472_row(
        symbol="AMD",
        expiration="20251017",
        strike="220.00",
        right="C",
        missing_date="2024-02-21",
        url="http://localhost:25503/v3/option/history/quote?symbol=AMD&date=20240221",
    )
    assert path == tmp_path / "thetadata" / "quote_472.csv"
    assert path is not None
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0].keys()) == QUOTE_472_CSV_COLUMNS
    assert rows[0]["symbol"] == "AMD"
    assert rows[0]["expiration"] == "20251017"
    assert rows[0]["strike"] == "220.00"
    assert rows[0]["right"] == "C"
    assert rows[0]["missing_date"] == "2024-02-21"
    assert "history/quote" in rows[0]["url"]
    assert quote_472_csv_path() == path
