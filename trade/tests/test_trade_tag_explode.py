"""Tests for backtesting.py trade-tag packing and Entry_* explosion.

These tests cover the dict ``tag`` payload used by ``make_bt_wrapper`` and the
post-run explode that restores ``Tag`` to ``signal_id`` while emitting
``Entry_*`` columns for additional entry features.

Usage:
    Run with ``pytest trade/tests/test_trade_tag_explode.py``.
"""

import pandas as pd
import pytest

from trade.backtester_._helper import build_entry_tag, explode_trade_tags


def test_build_entry_tag_includes_signal_id_and_extra_fields() -> None:
    """Pack extra entry info and always keep signal_id on the tag dict."""
    tag = build_entry_tag("USO20200102SHORT", {"zscore": 2.1, "ret_60d": 0.05})

    assert tag["signal_id"] == "USO20200102SHORT"
    assert tag["zscore"] == 2.1
    assert tag["ret_60d"] == 0.05


def test_build_entry_tag_overwrites_extra_signal_id() -> None:
    """Canonical open-decision signal_id wins over extra_info['signal_id']."""
    tag = build_entry_tag("keep-me", {"signal_id": "replace-me", "zscore": 1.0})

    assert tag["signal_id"] == "keep-me"
    assert tag["zscore"] == 1.0


def test_build_entry_tag_none_extra_info() -> None:
    """None extra_info still produces a signal_id-only tag dict."""
    assert build_entry_tag("abc") == {"signal_id": "abc"}


def test_build_entry_tag_rejects_non_dict_extra_info() -> None:
    """Non-dict extra_info is a contract error, not silently ignored."""
    with pytest.raises(TypeError, match="extra_info must be a dict or None"):
        build_entry_tag("abc", extra_info=["zscore"])  # type: ignore[arg-type]


def test_explode_trade_tags_restores_tag_and_adds_entry_columns() -> None:
    """Dict tags become Tag=signal_id plus Entry_<feature> columns."""
    trades = pd.DataFrame(
        {
            "Size": [1, -1],
            "Tag": [
                {"signal_id": "AAPL20200102SHORT", "zscore": 2.1, "atr_stop": 12.5},
                {"signal_id": "MSFT20200103SHORT", "zscore": 1.8, "atr_stop": 11.0},
            ],
        }
    )

    exploded = explode_trade_tags(trades)

    assert list(exploded["Tag"]) == ["AAPL20200102SHORT", "MSFT20200103SHORT"]
    assert "SHORT" in exploded.loc[0, "Tag"]
    assert list(exploded["Entry_zscore"]) == [2.1, 1.8]
    assert list(exploded["Entry_atr_stop"]) == [12.5, 11.0]
    assert "Entry_signal_id" not in exploded.columns
    assert "signal_id" not in exploded.columns


def test_explode_trade_tags_skips_existing_entry_columns() -> None:
    """Leave Strategy.I Entry_* snapshots in place on name collision."""
    trades = pd.DataFrame(
        {
            "Tag": [{"signal_id": "x", "zscore": 9.9, "ret_60d": 0.04}],
            "Entry_zscore": [1.23],
        }
    )

    exploded = explode_trade_tags(trades)

    assert exploded.loc[0, "Tag"] == "x"
    assert exploded.loc[0, "Entry_zscore"] == 1.23
    assert exploded.loc[0, "Entry_ret_60d"] == 0.04


def test_explode_trade_tags_does_not_double_prefix_entry_keys() -> None:
    """Keys already named Entry_* keep a single prefix."""
    trades = pd.DataFrame({"Tag": [{"signal_id": "x", "Entry_spread": 0.2}]})

    exploded = explode_trade_tags(trades)

    assert exploded.loc[0, "Entry_spread"] == 0.2
    assert "Entry_Entry_spread" not in exploded.columns


def test_explode_trade_tags_is_idempotent_for_plain_signal_ids() -> None:
    """Already-exploded or legacy Tag=signal_id frames are unchanged."""
    trades = pd.DataFrame({"Tag": ["AAPL20200102SHORT"], "Entry_zscore": [2.1]})

    exploded = explode_trade_tags(trades)

    pd.testing.assert_frame_equal(exploded, trades)


def test_explode_trade_tags_empty_or_missing_tag() -> None:
    """Empty frames and frames without Tag pass through."""
    empty = pd.DataFrame()
    no_tag = pd.DataFrame({"Size": [1]})

    assert explode_trade_tags(empty) is empty
    pd.testing.assert_frame_equal(explode_trade_tags(no_tag), no_tag)


def test_explode_then_join_matches_option_table_tag_contract() -> None:
    """Downstream Tag consumers can still copy Tag onto signal_id."""
    trades = pd.DataFrame(
        {
            "Tag": [{"signal_id": "USO20200102SHORT", "ret_60d": 0.1}],
            "ReturnPct": [0.05],
        }
    )
    exploded_df = explode_trade_tags(trades)
    exploded_df["signal_id"] = exploded_df["Tag"]

    assert exploded_df.loc[0, "signal_id"] == "USO20200102SHORT"
    assert exploded_df.loc[0, "Entry_ret_60d"] == 0.1
