"""Tests for environment-aware log rotation retention."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from trade.helpers import Logging as logging_mod


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ("live", logging_mod._LIVE_LOG_BACKUP_COUNT),
        ("long_bbands_v2", logging_mod._DEFAULT_LOG_BACKUP_COUNT),
        ("prod", logging_mod._DEFAULT_LOG_BACKUP_COUNT),
        (None, logging_mod._DEFAULT_LOG_BACKUP_COUNT),
    ],
)
def test_log_backup_count_by_environment(env: str | None, expected: int) -> None:
    with patch.object(logging_mod, "_get_current_environment", return_value=env):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LOG_BACKUP_DAYS", None)
            assert logging_mod._get_log_backup_count() == expected


def test_log_backup_days_env_overrides_environment() -> None:
    with patch.object(logging_mod, "_get_current_environment", return_value="live"):
        with patch.dict(os.environ, {"LOG_BACKUP_DAYS": "7"}):
            assert logging_mod._get_log_backup_count() == 7
