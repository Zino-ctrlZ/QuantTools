"""Sanitize option-spot frames whose columns are not string labels."""

from __future__ import annotations

import pandas as pd
import pytest

from trade.datamanager.exceptions import EmptyDataException
from trade.datamanager.utils.data_structure import _data_structure_sanitize


def test_sanitize_lowercases_string_columns() -> None:
    """Capitalized OHLC labels become lowercase."""
    idx = pd.DatetimeIndex(["2021-04-20 16:00:00", "2021-04-21 16:00:00"])
    df = pd.DataFrame({"Open": [1.0, 2.0], "Close": [1.5, 2.5]}, index=idx)
    out = _data_structure_sanitize(df, start="2021-04-20", end="2021-04-21")
    assert list(out.columns) == ["open", "close"]


def test_sanitize_empty_datetimeindex_rangeindex_raises_empty() -> None:
    """Empty option-spot fetch shape must not hit Index.str AttributeError."""
    df = pd.DataFrame()
    df.index = pd.DatetimeIndex([])
    with pytest.raises(EmptyDataException):
        _data_structure_sanitize(df, start="2021-04-20", end="2021-04-21", source_name="probe")


def test_sanitize_integer_columns_become_string_labels() -> None:
    """Mixed concat leftovers with integer names lowercase as strings."""
    idx = pd.DatetimeIndex(["2021-04-20 16:00:00", "2021-04-21 16:00:00"])
    df = pd.DataFrame({0: [1.0, 2.0], "Open": [3.0, 4.0]}, index=idx)
    out = _data_structure_sanitize(df, start="2021-04-20", end="2021-04-21")
    assert list(out.columns) == ["0", "open"]


def test_sanitize_multiindex_columns_flattened() -> None:
    """MultiIndex column tuples flatten to lowercase joined names."""
    idx = pd.DatetimeIndex(["2021-04-20 16:00:00"])
    cols = pd.MultiIndex.from_tuples([("OHLC", "Open"), ("OHLC", "Close")])
    df = pd.DataFrame([[1.0, 2.0]], index=idx, columns=cols)
    out = _data_structure_sanitize(df, start="2021-04-20", end="2021-04-21")
    assert list(out.columns) == ["ohlc_open", "ohlc_close"]
