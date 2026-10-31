"""Cash ledger and capital limits for the shared-cash simulator.

``Account`` holds starting capital, the live cash balance, and the rules that
cap new opens (cash floor, gross leverage, fixed size vs percent of equity,
buying-power policy). It does not hold positions; equity and gross exposure are
passed in from ``Portfolio`` when headroom is computed.

Comment density: domain policy

Core Classes:
    Account: Cash balance plus open-sizing and buying-power rules.

Processing Flow:
    1. ``resolve_notional`` turns a signal multiplier into a target dollar size.
    2. ``leverage_headroom`` / ``cash_headroom`` say how much of that size still fits.
    3. ``apply_buying_power`` picks the tighter limit and either returns the
       (possibly scaled) notional or records a capital-limit skip via ``record_skip``.
    4. Book vetoes also call ``record_skip`` (through ``Broker.skip_by_book``) so
       capital and book refusals share one ``SkipEvent`` list.

Risk/Assumptions:
    Shorts are not limited by ``min_cash`` (they add cash under our accounting).
    Closes never call this class for sizing; they always exit in full.
    Commission and slippage are percent-of-notional cash costs charged on open
    and close; they do not change fill prices (MAE/MFE stay raw price paths).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from trade.backtester_.simulator.core import BuyingPowerPolicy, PendingOrder, SkipEvent, StrategyKey
from trade.helpers.Logging import setup_logger

logger = setup_logger(__name__)


class Account:
    """Cash balance and the capital rules that gate new opens.

    Args:
        initial_capital: Starting cash for the shared pool.
        trade_size: Fixed dollar base size per open, or None when using percent.
        percent_of_capital: Fraction of current equity per open, or None when fixed.
        max_gross_leverage: Cap on total position size over equity; ``inf`` = none.
        min_cash: Cash that must remain after a long open; ``None`` = no floor.
        buying_power_policy: Skip or shrink opens that breach a limit.
        commission_pct: Fraction of trade notional taken as commission (default 0).
        slippage_pct: Fraction of trade notional taken as slippage cost (default 0).
    """

    def __init__(
        self,
        initial_capital: float,
        *,
        trade_size: Optional[float] = None,
        percent_of_capital: Optional[float] = None,
        max_gross_leverage: float = np.inf,
        min_cash: Optional[float] = 0.0,
        buying_power_policy: BuyingPowerPolicy = BuyingPowerPolicy.SKIP,
        commission_pct: float = 0.0,
        slippage_pct: float = 0.0,
    ) -> None:
        """Store capital rules and set cash to ``initial_capital``."""
        self.initial_capital = float(initial_capital)
        self.trade_size = trade_size
        self.percent_of_capital = percent_of_capital
        self.max_gross_leverage = float(max_gross_leverage)
        self.min_cash = None if min_cash is None else float(min_cash)
        self.buying_power_policy = BuyingPowerPolicy(buying_power_policy)
        if commission_pct < 0 or slippage_pct < 0:
            raise ValueError("commission_pct and slippage_pct must be >= 0")
        self.commission_pct = float(commission_pct)
        self.slippage_pct = float(slippage_pct)
        self.cash: float = self.initial_capital
        ## Capital-limit refusals share one list with book vetoes on the Broker.
        self.skipped: List[SkipEvent] = []

    def reset(self) -> None:
        """Restore cash to starting capital and clear capital-limit skips."""
        self.cash = self.initial_capital
        self.skipped = []

    def record_skip(
        self,
        ts: pd.Timestamp,
        strategy_key: StrategyKey,
        *,
        requested_notional: float,
        filled_notional: float,
        reason: str,
        available_cash: Optional[float] = None,
    ) -> None:
        """Append one capital-limit or book-veto skip to the shared log.

        Args:
            ts: Decision date.
            strategy_key: Strategy that tried to open.
            requested_notional: Size asked before the cap (``nan`` for book vetoes).
            filled_notional: Size allowed after the policy (0 when fully skipped).
            reason: Limit text or ``book: ...`` veto.
            available_cash: Cash at decision; defaults to current ledger cash.
        """
        self.skipped.append(
            SkipEvent(
                date=ts,
                strategy_key=StrategyKey(strategy_key),
                requested_notional=float(requested_notional),
                filled_notional=float(filled_notional),
                available_cash=float(self.cash if available_cash is None else available_cash),
                reason=reason,
            )
        )

    def credit(self, amount: float) -> None:
        """Add dollars to cash (short open proceeds, long close proceeds)."""
        self.cash += amount

    def debit(self, amount: float) -> None:
        """Remove dollars from cash (long open cost, short cover cost)."""
        self.cash -= amount

    def transaction_costs(self, notional: float) -> Tuple[float, float]:
        """Commission and slippage dollars for a trade notional.

        Args:
            notional: Absolute dollar size of the fill (sign ignored).

        Returns:
            ``(commission, slippage)`` cash costs.
        """
        size = abs(float(notional))
        return size * self.commission_pct, size * self.slippage_pct

    def charge_costs(self, notional: float) -> Tuple[float, float]:
        """Debit commission + slippage for ``notional`` and return both fee legs.

        Args:
            notional: Absolute dollar size of the fill (sign ignored).

        Returns:
            ``(commission, slippage)`` that were removed from cash.
        """
        commission, slippage = self.transaction_costs(notional)
        self.debit(commission + slippage)
        return commission, slippage

    def resolve_notional(self, equity: float, multiplier: float) -> float:
        """Target dollar size for an open: base size times the signal multiplier.

        Args:
            equity: Current portfolio equity (cash plus marks).
            multiplier: Signal leverage multiplier (1.0 = base size).

        Returns:
            Dollar notional to attempt before buying-power caps.
        """
        base = self.trade_size if self.trade_size is not None else self.percent_of_capital * equity
        return float(base) * multiplier

    def leverage_headroom(self, equity: float, gross_exposure: float) -> float:
        """How many more dollars of position size the leverage cap still allows."""
        ## Short-circuit the uncapped case: inf * equity is nan when equity is 0.
        if np.isinf(self.max_gross_leverage):
            return np.inf
        return max(self.max_gross_leverage * equity - gross_exposure, 0.0)

    def cost_rate(self) -> float:
        """Combined commission + slippage fraction of notional."""
        return self.commission_pct + self.slippage_pct

    def cash_headroom(self, side: int) -> float:
        """Max long notional that still leaves ``min_cash`` after the open and fees.

        Longs spend ``notional * (1 + cost_rate)`` from cash. This returns that
        budget converted back into position notional so ``apply_buying_power``
        can compare apples-to-apples with leverage headroom. Shorts add sale
        proceeds and are not limited by the cash floor (fees still leave them
        cash-positive while ``cost_rate < 1``).
        """
        ## Shorts never eat into the floor. Only longs spend cash.
        if self.min_cash is None or side < 0:
            return np.inf
        available = max(self.cash - self.min_cash, 0.0)
        rate = self.cost_rate()
        ## Reserve fee drag inside the cash budget before sizing the position.
        if rate <= 0.0:
            return available
        return available / (1.0 + rate)

    def apply_buying_power(
        self,
        order: PendingOrder,
        ts: pd.Timestamp,
        notional: float,
        *,
        equity: float,
        gross_exposure: float,
    ) -> float:
        """Fit an open inside leverage and cash-after-fees limits, or skip / scale it.

        The tighter of the two limits decides. Cash headroom already shrinks for
        commission and slippage on longs. ``buying_power_policy`` applies the
        same way whichever limit binds.

        Args:
            order: Pending open being sized.
            ts: Fill date (for the skip log).
            notional: Requested dollar size before caps.
            equity: Current portfolio equity.
            gross_exposure: Current gross position size.

        Returns:
            Notional to fill, or 0.0 when the open is skipped.
        """
        limits = {
            f"gross leverage cap {self.max_gross_leverage}x": self.leverage_headroom(equity, gross_exposure),
            f"min cash {self.min_cash}": self.cash_headroom(order.side),
        }
        reason, headroom = min(limits.items(), key=lambda kv: kv[1])
        if notional <= headroom:
            return notional
        filled = headroom if self.buying_power_policy == BuyingPowerPolicy.SCALE_DOWN else 0.0
        self.record_skip(
            ts,
            order.strategy_key,
            requested_notional=notional,
            filled_notional=filled,
            reason=reason,
        )
        return filled
