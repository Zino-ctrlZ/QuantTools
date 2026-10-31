"""Tests for parse_signal_id and its fast YYYYMMDD date path."""

import unittest

import pandas as pd

from EventDriven.helpers import _parse_signal_date, parse_signal_id


class TestParseSignalDate(unittest.TestCase):
    """The fast path must match pd.to_datetime exactly, including failures."""

    def test_matches_pd_to_datetime_on_every_day_of_two_years(self) -> None:
        """Valid YYYYMMDD strings, leap day included, parse to the same Timestamp."""
        for s in pd.date_range("2019-01-01", "2020-12-31", freq="D").strftime("%Y%m%d"):
            self.assertEqual(_parse_signal_date(s), pd.to_datetime(s))

    def test_invalid_dates_still_raise(self) -> None:
        """Impossible dates fall back to pd.to_datetime and raise like before."""
        for s in ("20201301", "20210229", "00000000", "abcdefgh"):
            with self.assertRaises(ValueError):
                _parse_signal_date(s)

    def test_non_digit_input_uses_original_parser(self) -> None:
        """Inputs outside the fast path keep pd.to_datetime behavior."""
        self.assertEqual(_parse_signal_date("2020-1-2"), pd.to_datetime("2020-1-2"))


class TestParseSignalId(unittest.TestCase):
    """parse_signal_id output is unchanged."""

    def test_long_with_slug(self) -> None:
        """Slug, ticker, date, and direction are split out."""
        self.assertEqual(
            parse_signal_id("mr::AAPL20200102LONG"),
            dict(direction="LONG", date=pd.Timestamp("2020-01-02"), ticker="AAPL", strategy_slug="mr"),
        )

    def test_short_without_slug(self) -> None:
        """Short ids slice the date one character earlier."""
        self.assertEqual(
            parse_signal_id("BA20251030SHORT"),
            dict(direction="SHORT", date=pd.Timestamp("2025-10-30"), ticker="BA", strategy_slug=None),
        )

    def test_close_id_raises(self) -> None:
        """Ids without LONG or SHORT are rejected before the date is parsed."""
        with self.assertRaises(ValueError):
            parse_signal_id("AMZN20251118CLOSE")


if __name__ == "__main__":
    unittest.main()
