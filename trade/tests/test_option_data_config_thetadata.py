"""Tests that OptionDataConfig.thetadata is the live ThetaData SETTINGS object."""

from __future__ import annotations

from dbase.DataAPI.ThetaData.v3.vars import SETTINGS
from trade.datamanager.config import (
    ListedSessionNotFoundPolicy,
    OptionDataConfig,
    ThetaDataV3Controls,
)


def test_option_data_config_exports_thetadata_types() -> None:
    """ThetaData policy enum and controls class are importable from config."""
    assert ListedSessionNotFoundPolicy.OMIT.value == "omit"
    assert ListedSessionNotFoundPolicy.RAISE.value == "raise"
    assert ThetaDataV3Controls is type(SETTINGS)


def test_option_data_config_thetadata_is_settings_singleton() -> None:
    """Editing OptionDataConfig().thetadata mutates ThetaData SETTINGS in place."""
    cfg = OptionDataConfig()
    assert cfg.thetadata is SETTINGS
    previous = cfg.thetadata.listed_session_not_found
    try:
        cfg.thetadata.listed_session_not_found = ListedSessionNotFoundPolicy.OMIT
        assert SETTINGS.listed_session_not_found is ListedSessionNotFoundPolicy.OMIT
        cfg.thetadata.listed_session_not_found = ListedSessionNotFoundPolicy.RAISE
        assert SETTINGS.listed_session_not_found is ListedSessionNotFoundPolicy.RAISE
        cfg.thetadata.use_old_formatting = False
        assert SETTINGS.use_old_formatting is False
        cfg.thetadata.use_old_formatting = True
        assert SETTINGS.use_old_formatting is True
    finally:
        cfg.thetadata.listed_session_not_found = previous
