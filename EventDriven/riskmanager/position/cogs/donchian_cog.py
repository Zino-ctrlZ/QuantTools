"""Donchian momentum position sizing and roll opinion cog.

Applies Donchian-specific sizing logic for new positions and emits DTE-based
ROLL opinions during analysis for Donchian strategy positions. Persistence of
limits and metadata goes through ``PositionStore`` (in-memory by default,
database when ``live=True``).

Core Classes:
    DonchianMomentumCogConfig: Runtime knobs for scaling and DTE checks.
    DonchianMomentumCog: Position sizing and analysis implementation.

Core Dataclasses:
    _DonchianLimitsMetaData: Per-trade sizing metadata capture.

Usage:
    >>> cog = DonchianMomentumCog(eq_strategy=strategy, live=False)
    >>> cog.on_new_position(state)
"""

from typing import Dict, Optional, TYPE_CHECKING

import pandas as pd
from dataclasses import dataclass

from EventDriven.configs.core import DonchianMomentumCogConfig
from EventDriven.dataclasses.limits import PositionLimits
from EventDriven.dataclasses.states import CogActions, NewPositionState, PositionAnalysisContext
from EventDriven.riskmanager.actions import ROLL, Changes
from EventDriven.riskmanager.position.base import BaseCog
from EventDriven.riskmanager.position.cogs.analyze_utils import get_dte_and_moneyness_from_trade_id
from EventDriven.riskmanager.position.cogs.limits import _LimitsMetaData
from EventDriven.riskmanager.live_diag import log_live_checkpoint
from EventDriven.riskmanager.sizer._utils import default_delta_limit, delta_position_sizing
from EventDriven.types import SignalID
from trade.backtester_._multi_asset_strategy import MultiAssetStrategy
from trade.helpers.Logging import setup_logger
from trade.helpers.helper import change_to_last_busday

if TYPE_CHECKING:
    from EventDriven.riskmanager.position.stores.limits_store import PositionStore

logger = setup_logger("EventDriven.riskmanager.position.cogs.donchian_cog")


def _entry_delay_days_from_info(info: Dict[str, object]) -> int:
    """Return entry delay from composite ``momentum_*`` or standalone strategy info."""
    return int(info.get("momentum_entry_delay_days", info.get("entry_delay_days", 0)))


@dataclass
class _DonchianLimitsMetaData(_LimitsMetaData):
    """Metadata payload recorded for Donchian-sized positions."""

    breakout_score: float = None
    rvol: float = None


class DonchianMomentumCog(BaseCog):
    """Donchian momentum sizing cog with optional database-backed persistence."""

    default_config: DonchianMomentumCogConfig = DonchianMomentumCogConfig()

    def __init__(
        self,
        eq_strategy: MultiAssetStrategy,
        config: Optional[DonchianMomentumCogConfig] = None,
        *,
        live: bool = False,
        position_store: Optional["PositionStore"] = None,
        verify_after_save: bool = True,
    ) -> None:
        """Initialize the Donchian momentum cog.

        Args:
            eq_strategy: Strategy interface used to fetch indicator data.
            config: Optional runtime configuration for sizing and analysis.
            live: When ``True``, persist limits/metadata via ``DatabasePositionStore``.
            position_store: Optional store override for testing.
            verify_after_save: Re-read the database after each write when ``live``.
        """
        ## Lazy import avoids circular RiskManager -> cogs -> stores -> backtest load
        from EventDriven.riskmanager.position.stores.limits_store import build_position_store

        if config is None:
            config = DonchianMomentumCogConfig()

        super().__init__(config)
        self._config: DonchianMomentumCogConfig = config
        self.eq_strategy = eq_strategy
        self.live = live
        self.position_limits: Dict[str, PositionLimits] = {}
        self.position_metadata: Dict[str, _DonchianLimitsMetaData] = {}
        self.config: DonchianMomentumCogConfig = config
        strategy_name = self.config.run_name or "donchian_momentum_cog"
        self._position_store = build_position_store(
            live=live,
            strategy_name=strategy_name,
            default_dte=self.config.dte_threshold,
            position_store=position_store,
            verify_after_save=verify_after_save,
        )

    def is_donchian_strategy(self, signal_id: str) -> bool:
        """Return True when signal_id belongs to a Donchian strategy."""
        try:
            return SignalID(signal_id).strategy_slug.startswith("donchian")
        except Exception:
            logger.warning(f"Unable to parse signal id {signal_id} for Donchian check.", exc_info=True)
            return False

    def on_new_position(self, state: NewPositionState) -> None:
        """
        Analyze the momentum of a new position based on its entry price and the Donchian channel.

        Args:
            state: The new position state after creation.
        Returns:
            None (this cog is for analysis and opinion generation, not direct action).
        """

        order = state.order
        request = state.request
        ticker = state.symbol
        undl_data = state.undl_at_time_data
        chain_spot = undl_data.chain_spot["close"]
        option_chain = state.at_time_data

        cash_available = (
            request.tick_cash if request.is_tick_cash_scaled else request.tick_cash * 100
        )  # Scale cash to match option notional if not already scaled
        signal_id = SignalID(order.signal_id)
        if not self.is_donchian_strategy(order.signal_id):
            logger.debug(
                f"Skipping Donchian momentum sizing for non-Donchian signal {signal_id}. "
                f"Trade ID: {order['data']['trade_id']}"
            )
            return

        opt_price = option_chain.get_price()
        date = order["date"]
        ## Strategy bars are calendar-indexed; order dates often carry EOD 16:00 and miss _dates_map.
        lookup_date = pd.Timestamp(date).strftime("%Y-%m-%d")
        logger.info(f"On new position: {state} for ticker {ticker} and date {date}")
        ## Get info on the date of the entry
        info = self.eq_strategy.info_on_date(ticker=ticker, current_date=lookup_date)
        entry_delay_days = _entry_delay_days_from_info(info)

        ## Back up by entry_delay business days, then normalize to YYYY-MM-DD for index lookup
        date_minus_entry_delay = (
            change_to_last_busday(
                pd.Timestamp(date) - pd.offsets.BusinessDay(entry_delay_days),
                time_of_day_aware=False,
                eod_time=False,
            )
            .strftime("%Y-%m-%d")
        )
        info_on_entry_delay = self.eq_strategy.info_on_date(
            ticker=ticker, current_date=date_minus_entry_delay
        )
        logger.info(f"Info on entry delay {date_minus_entry_delay}: {info_on_entry_delay}")
        logger.info(f"Date: {lookup_date}, Date minus entry delay: {date_minus_entry_delay}")

        ## Get the breakout score on the date of the entry delay
        breakout_score = info_on_entry_delay.get("momentum_breakout_score", 0)
        rvol = info_on_entry_delay.get("momentum_rvol", 0)

        delta = option_chain.delta
        scaler = self.get_momentum_scaler(
            breakout_score=breakout_score, rvol=rvol, min_mult=self.config.min_scale, max_mult=self.config.max_scale
        )
        rvol_str = f"{rvol:.4f}" if rvol is not None and not pd.isna(rvol) else "None"
        logger.info(
            f"Calculated momentum scaler: {scaler:.2f} for trade ID {order['data']['trade_id']} with breakout score {breakout_score} and RVOL {rvol_str}"
        )

        base_limit = default_delta_limit(
            cash_available=cash_available,
            underlier_price_at_time=chain_spot,
            sizing_lev=self.config.sizing_lev,
        )

        scaled_limit = base_limit * scaler
        logger.info(
            f"Base delta limit: {base_limit:.4f}, Scaled delta limit: {scaled_limit:.4f} for trade ID {order['data']['trade_id']} with momentum scaler {scaler:.2f}"
        )

        ## Scale the delta down to 90% to allow room for natural delta movement and reduce overtrading risk
        tgt_delta = scaled_limit * 0.9

        q = delta_position_sizing(
            cash_available=cash_available,
            option_price_at_time=opt_price,
            delta=delta,
            delta_limit=tgt_delta,
        )

        if q == 0:
            logger.warning(
                f"Position {order['data']['trade_id']} has zero quantity after momentum scaling. "
                f"Breakout Score: {breakout_score}, RVOL: {rvol}, Scaler: {scaler:.2f}, "
                f"Base Delta Limit: {base_limit:.4f}, Scaled Delta Limit: {scaled_limit:.4f}, "
                f"Target Delta (90% of Scaled Limit): {tgt_delta:.4f}, Option Delta: {delta:.4f}"
            )
            if cash_available >= opt_price * 100:  # Check if cash can afford at least 1 contract
                logger.info(
                    f"Cash available (${cash_available:.2f}) is greater than option price (${opt_price * 100:.2f}), but quantity is zero."
                    f" Setting quantity to 1 to allow position opening and future momentum adjustments."
                )
                q = 1

            else:
                logger.warning(
                    f"Cash available (${cash_available:.2f}) is insufficient to afford even 1 contract at option price (${opt_price * 100:.2f}). Quantity remains zero."
                )

        logger.info(
            f"Calculated delta limit: {scaled_limit:.4f}, resulting quantity: {q} lev: {self.config.sizing_lev}. Breakout Score: {breakout_score}, RVOL: {rvol}, Scaler: {scaler:.2f}. Trade ID: {order['data']['trade_id']} with option delta {delta:.4f}"
        )
        ## Live diag: Donchian consume-site (delta here is what sizing just used)
        log_live_checkpoint(
            "donchian_sized",
            trade_id=order["data"]["trade_id"],
            date=request.date,
            data={
                "delta": delta,
                "delta_lmt": scaled_limit,
                "tgt_delta": tgt_delta,
                "price": opt_price,
                "cash": cash_available,
                "quantity": q,
                "scaler": scaler,
                "breakout_score": breakout_score,
                "rvol": rvol,
            },
            note="post_size",
        )
        order["data"]["quantity"] = q
        metadata = _DonchianLimitsMetaData(
            trade_id=order["data"]["trade_id"],
            date=order["date"],
            signal_id=order["signal_id"],
            scalar=scaler,
            sizing_lev=self.config.sizing_lev,
            delta_per_contract=delta,
            option_price=opt_price,
            undl_price=undl_data.chain_spot["close"],
            delta_lmt=scaled_limit,
            new_quantity=q,
            breakout_score=breakout_score,
            rvol=rvol,
        )

        pos_lmts = PositionLimits(
            delta=scaled_limit,
            dte=self.config.dte_threshold,
            creation_date=state.request.date,
        )
        state.limits = pos_lmts

        self._save_position_limits(order["data"]["trade_id"], order["signal_id"], pos_lmts)
        self._store_metadata(metadata)
        logger.debug(f"Stored momentum metadata for trade ID {order['data']['trade_id']}: {metadata}")

    def _save_position_limits(self, trade_id: str, signal_id: str, limits: PositionLimits) -> None:
        """Persist position limits through the configured store.

        Args:
            trade_id: The unique identifier for the trade.
            signal_id: The identifier for the signal that generated the trade.
            limits: The PositionLimits object containing the limits to be saved.
        """
        self.position_limits[trade_id] = limits
        self._position_store.save_limits(trade_id, signal_id, limits)
        logger.debug(f"Saved position limits for trade ID {trade_id}: {limits}")

    def _get_position_limits(self, trade_id: str, signal_id: str) -> Optional[PositionLimits]:
        """Load position limits from the in-process cache or store.

        Args:
            trade_id: Unique trade identifier.
            signal_id: Originating signal identifier.

        Returns:
            Stored limits, or ``None`` when absent.
        """
        if trade_id in self.position_limits:
            return self.position_limits[trade_id]
        limits = self._position_store.get_limits(trade_id, signal_id)
        if limits is not None:
            self.position_limits[trade_id] = limits
        return limits

    def _store_metadata(self, metadata: _DonchianLimitsMetaData) -> None:
        """Persist metadata in-process and through the configured store.

        Args:
            metadata: The _DonchianLimitsMetaData object containing the metadata to be stored.
        """
        self.position_metadata[metadata.trade_id] = metadata
        self._position_store.save_metadata(metadata.trade_id, metadata)
        logger.debug(f"Stored metadata for trade ID {metadata.trade_id}: {metadata}")

    def get_momentum_scaler(
        self,
        *,
        breakout_score: float,
        rvol: float,
        min_mult: float = 0.75,
        max_mult: float = 1.50,
    ) -> float:
        """
        Compute bounded momentum sizing multiplier using:
            - breakout score
            - realized volatility

        Returns
        -------
        float
            Position sizing multiplier.
        """

        mult = 1.0

        # --------------------------------------------------
        # Breakout Score
        # --------------------------------------------------

        if not pd.isna(breakout_score):
            # strongest breakout regimes
            if breakout_score >= 2.5:
                mult *= 2

            elif breakout_score >= 2:
                mult *= 1.6

            elif breakout_score >= 1.5:
                mult *= 1.40

            elif breakout_score >= 1:
                mult *= 1.15

            # weak breakout
            elif breakout_score < 0.25:
                mult *= 0.90

        # --------------------------------------------------
        # Realized Volatility
        # --------------------------------------------------

        if not pd.isna(rvol):
            # strongest observed region
            if rvol >= 0.75 and rvol < 1.00:
                mult *= 2

            elif 0.60 <= rvol < 0.75:
                mult *= 1.6

            elif 0.50 <= rvol < 0.60:
                mult *= 1.40

            elif 0.40 <= rvol < 0.50:
                mult *= 1.15

            # compressed / weak momentum
            elif rvol < 0.15:
                mult *= 0.85

            # ultra-extreme / unstable
            elif rvol >= 1.00:
                mult *= 0.95

        # --------------------------------------------------
        # Final Clamp
        # --------------------------------------------------
        mult = max(min_mult, mult)
        mult = min(max_mult, mult)

        return float(mult)

    def _analyze_impl(self, context: PositionAnalysisContext) -> CogActions:
        """
        Analyze open positions for DTE-based roll triggers.

        Process:
            1. Iterate over all open positions in portfolio
            2. Calculate DTE using get_dte_and_moneyness_from_trade_id()
            3. If dte_limit_enabled and dte < dte_threshold:
               - Create ROLL opinion with reason
            4. Return aggregated CogActions

        Args:
            context: Portfolio snapshot with positions, date, and backtest parameters.

        Returns:
            CogActions: Opinions generated by this cog (roll recommendations).
        """
        opinions = []

        if not self.config.dte_limit_enabled:
            logger.debug("DTE limit disabled; no roll checks performed")
            return CogActions(date=context.date, source_cog=self.name, opinions=opinions)

        positions = context.portfolio.positions
        portfolio_meta = context.portfolio_meta
        last_updated = context.portfolio.last_updated
        t_plus_n = portfolio_meta.t_plus_n
        t_plus_n_bdays = pd.offsets.BusinessDay(max(t_plus_n, 1))
        backtest_start = portfolio_meta.start_date

        for pos_state in positions:
            try:
                if not self.is_donchian_strategy(pos_state.signal_id):
                    continue

                # Calculate DTE using exact same method as limits cog
                dte, _ = get_dte_and_moneyness_from_trade_id(
                    trade_id=pos_state.trade_id,
                    check_date=pos_state.last_updated,
                    check_price=pos_state.current_underlier_data.chain_spot["close"],
                    start=backtest_start,
                    is_backtest=portfolio_meta.is_backtest,
                )

                qty = pos_state.quantity

                # Check roll threshold
                if dte < self.config.dte_threshold:
                    logger.info(
                        f"Position {pos_state.trade_id}: DTE {dte} below threshold {self.config.dte_threshold}. "
                        f"Recommending ROLL."
                    )

                    # quantity_diff = -abs(qty) signals close-first for live; new_quantity is post-roll size.
                    action = ROLL(
                        trade_id=pos_state.trade_id,
                        action=Changes(quantity_diff=-abs(qty), new_quantity=qty),
                    )
                    action.reason = (
                        f"DTE {dte} below threshold {self.config.dte_threshold}. Rolling to extend duration."
                    )
                    action.analysis_date = last_updated
                    action.effective_date = last_updated + t_plus_n_bdays

                    # Create opinion for this position
                    action.verbose_info = (
                        f"Analysis: {action.analysis_date} | Effective: {action.effective_date} | "
                        f"Trade: {pos_state.trade_id} | DTE: {dte} | Threshold: {self.config.dte_threshold}"
                    )
                    pos_state.action = action
                    opinions.append(pos_state)
                else:
                    logger.debug(
                        f"Position {pos_state.trade_id}: DTE {dte} >= threshold {self.config.dte_threshold}. No action."
                    )

            except Exception as e:
                logger.warning(f"Error calculating DTE for position {pos_state.trade_id}: {e}. Skipping roll check.")
                logger.warning(f"Stack trace: ", exc_info=True)
                continue

        logger.info(
            f"DonchianCog analysis complete. {len(opinions)} roll opinion(s) generated for {len(positions)} position(s)."
        )

        return CogActions(date=context.date, source_cog=self.name, opinions=opinions)
