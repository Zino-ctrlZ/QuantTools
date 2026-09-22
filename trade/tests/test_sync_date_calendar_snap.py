"""Unit tests for ``_sync_date`` vendor-calendar snapping.

Comment density: domain policy

Core Functions:
    test_sync_date_keeps_listed_session_with_market_close_time: afternoon start on a
        listed day must not snap to the next midnight.
    test_sync_date_snaps_unlisted_holiday_to_nearest_session: true calendar gaps still
        snap to the nearest listed date.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

from trade.datamanager._enums import OptionSpotEndpointSource
from trade.datamanager.utils.date import _sync_date


def test_sync_date_keeps_listed_session_with_market_close_time() -> None:
    """Keep a listed calendar day when the request carries market-close time.

    ``list_dates`` returns midnight timestamps. Callers such as
    ``load_option_pnl_data`` pass ``change_to_last_busday`` results at 16:00.
    Membership must use calendar dates so 2026-09-17 16:00 stays on 2026-09-17.
    """
    listed = [
        "2026-09-16",
        "2026-09-17",
        "2026-09-18",
        "2026-09-21",
    ]
    with patch(
        "trade.datamanager.utils.date.get_listed_option_dates",
        return_value=listed,
    ):
        start, end = _sync_date(
            symbol="META",
            start_date=datetime(2026, 9, 17, 16, 0, 0),
            end_date=datetime(2026, 9, 21, 0, 0, 0),
            strike=750.0,
            expiration="2027-03-19",
            right="C",
            endpoint_source=OptionSpotEndpointSource.EOD,
        )

    assert start.date().isoformat() == "2026-09-17"
    assert end.date().isoformat() == "2026-09-21"


def test_sync_date_snaps_unlisted_holiday_to_nearest_session() -> None:
    """Snap an unlisted holiday start to the nearest listed session date."""
    listed = [
        "2026-09-04",
        "2026-09-08",
        "2026-09-09",
    ]
    with patch(
        "trade.datamanager.utils.date.get_listed_option_dates",
        return_value=listed,
    ):
        start, end = _sync_date(
            symbol="META",
            start_date=datetime(2026, 9, 7, 16, 0, 0),
            end_date=datetime(2026, 9, 9, 0, 0, 0),
            strike=750.0,
            expiration="2027-03-19",
            right="C",
            endpoint_source=OptionSpotEndpointSource.EOD,
        )

    assert start.date().isoformat() == "2026-09-08"
    assert end.date().isoformat() == "2026-09-09"
