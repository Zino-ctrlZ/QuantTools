"""Tests for the portfolio Simulator: simulate() delegation, start_date, capital limits, cash and exposure stats."""

from __future__ import annotations

import unittest
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from EventDriven.types import PositionEffect, SignalID
from trade.backtester_._strategy import StrategyBase, TradeDecision
from trade.backtester_._types import SideInt
from trade.backtester_.data import PTDataset
from trade.backtester_.simulator import (
    BuyingPowerPolicy,
    Simulator,
    compute_cash_stats,
    compute_exposure_stats,
    drawdown_stats,
    eq_stats,
    extended_stats,
    yearly_returns,
)


def _make_dataset(closes: list[float], ticker: str = "TEST") -> PTDataset:
    """Build a minimal OHLCV dataset on business days from 2020-01-01."""
    dates = pd.date_range("2020-01-01", periods=len(closes), freq="B")
    frame = pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1_000},
        index=dates,
    )
    return PTDataset(ticker, frame)


class _ScriptedStrategy(StrategyBase):
    """Open and close long positions on fixed bar indices."""

    bt_params = {"opens": (), "closes": ()}

    def __init__(
        self,
        data: PTDataset,
        start_trading_date: Optional[str] = None,
        ticker: Optional[str] = None,
        tplusn: int = 0,
        *,
        opens: Iterable[int] = (),
        closes: Iterable[int] = (),
    ) -> None:
        """Store the scripted bar indices and initialize the base strategy."""
        self._opens = set(opens)
        self._closes = set(closes)
        self._signal_id = SignalID("test::USO20200101LONG")
        super().__init__(data, start_trading_date=start_trading_date, ticker=ticker or data.name, tplusn=tplusn)

    def setup(self) -> None:
        """No indicators required for scripted tests."""

    def open_action(self, *, signal_id=None, entry_price=None, side=None, date=None, index=None) -> None:
        """Record position info via the base implementation."""
        super().open_action(signal_id=signal_id, entry_price=entry_price, side=side, date=date, index=index)

    def close_action(self, *, date=None, index=None) -> None:
        """Clear position info via the base implementation."""
        super().close_action(date=date, index=index)

    def is_open_signal(self, *, date: pd.Timestamp = None, index: int = None) -> bool:
        """Return True on scripted open bars."""
        idx, _ = self._resolve(date=date, index=index)
        return idx in self._opens

    def is_close_signal(self, *, date: pd.Timestamp = None, index: int = None) -> bool:
        """Return True on scripted close bars."""
        idx, _ = self._resolve(date=date, index=index)
        return idx in self._closes

    def should_open(self, *, date: pd.Timestamp = None, index: int = None) -> TradeDecision:
        """Open long on scripted bars when flat."""
        if not self.is_open_signal(date=date, index=index) or self.position_open:
            return TradeDecision(ok=False, side=0)
        return TradeDecision(
            ok=True, side=int(SideInt.BUY), pos_effect=PositionEffect.OPEN, signal_id=self._signal_id
        )

    def should_close(self, *, date: pd.Timestamp = None, index: int = None) -> TradeDecision:
        """Close on scripted bars when in a position."""
        if not self.position_open or not self.is_close_signal(date=date, index=index):
            return TradeDecision(ok=False, side=0)
        return TradeDecision(ok=True, side=int(SideInt.BUY), pos_effect=PositionEffect.CLOSE, signal_id="N/A")


class TestStrategySimulateDelegation(unittest.TestCase):
    """StrategyBase.simulate runs through a one-strategy Simulator."""

    def test_simulate_compounds_sequential_trades(self) -> None:
        """Each open is sized to equity at entry, so the second trade compounds the first."""
        closes = [100.0, 110.0, 99.0, 120.0, 60.0, 90.0]
        strategy = _ScriptedStrategy(_make_dataset(closes), opens=[0, 3], closes=[2, 5])

        trades, equity = strategy.simulate()

        ## Prior flat seed, then trade path. Trade 1: 100 -> 99 leaves 0.99.
        ## Trade 2: 0.99 at 120, marked at 60 then 90.
        expected = [1.0, 1.0, 1.1, 0.99, 0.99, 0.495, 0.7425]
        for actual, want in zip(equity.tolist(), expected):
            self.assertAlmostEqual(actual, want)
        self.assertEqual(equity.name, "equity")
        self.assertEqual([t["action"] for t in trades], ["open", "close", "open", "close"])

    def test_simulate_trims_to_start_trading_date(self) -> None:
        """Equity seeds one bus day before start_trading_date; trading starts on that date."""
        data = _make_dataset([100.0] * 5)
        start = data.data.index[2]
        strategy = _ScriptedStrategy(data, start_trading_date=start.strftime("%Y-%m-%d"), opens=[], closes=[])

        _, equity = strategy.simulate()

        seed = start - pd.tseries.offsets.BDay(1)
        self.assertEqual(equity.index[0], seed)
        self.assertEqual(len(equity), 4)
        self.assertAlmostEqual(float(equity.iloc[0]), 1.0)

    def test_simulate_without_finalize_leaves_trade_open(self) -> None:
        """finalize=False keeps the last position open and marked to market."""
        strategy = _ScriptedStrategy(_make_dataset([100.0, 110.0, 120.0]), opens=[0], closes=[])

        trades, equity = strategy.simulate(finalize=False)

        self.assertEqual([t["action"] for t in trades], ["open"])
        self.assertAlmostEqual(equity.iloc[-1], 1.2)

    def test_enforce_open_drops_stale_delayed_open(self) -> None:
        """With t+1 and enforcement on, an open whose signal is gone at execution never fills."""
        data = _make_dataset([100.0, 110.0, 120.0])
        strategy = _ScriptedStrategy(data, tplusn=1, opens=[0], closes=[])

        trades, _ = strategy.simulate(enforce_open_on_signal=True)
        self.assertEqual(trades, [])

        trades, _ = strategy.simulate(enforce_open_on_signal=False)
        self.assertEqual(trades[0]["date"], data.data.index[1])


class TestCommissionAndSlippage(unittest.TestCase):
    """Percent-of-notional cash costs; MAE/MFE and return_pct stay raw price paths."""

    def test_zero_costs_match_baseline_equity(self) -> None:
        """Default zero fees leave equity identical to an explicit zero-cost run."""
        data = _make_dataset([100.0, 110.0, 120.0])
        strategy = _ScriptedStrategy(data, opens=[0], closes=[2])
        baseline = Simulator([strategy], capital=10_000.0, percent_of_capital=1.0).run()
        with_zeros = Simulator(
            [_ScriptedStrategy(data, opens=[0], closes=[2])],
            capital=10_000.0,
            percent_of_capital=1.0,
            commission_pct=0.0,
            slippage_pct=0.0,
        ).run()
        self.assertAlmostEqual(baseline.equity.iloc[-1], with_zeros.equity.iloc[-1])

    def test_round_trip_charges_open_and_close_and_keeps_raw_return(self) -> None:
        """Fees debit cash twice; return_pct ignores fees; legacy events expose them."""
        ## 100 -> 110 fully invested: raw return +10%. min_cash=None so fee-aware
        ## buying power does not shrink/skip the open (that path has its own test).
        data = _make_dataset([100.0, 110.0])
        strategy = _ScriptedStrategy(data, opens=[0], closes=[1])
        commission_pct = 0.001
        slippage_pct = 0.002
        result = Simulator(
            [strategy],
            capital=10_000.0,
            percent_of_capital=1.0,
            min_cash=None,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
        ).run()

        record = next(iter(result.trades.values()))[0]
        entry_fee = 10_000.0 * (commission_pct + slippage_pct)
        exit_fee = 11_000.0 * (commission_pct + slippage_pct)
        self.assertAlmostEqual(record.entry_commission, 10_000.0 * commission_pct)
        self.assertAlmostEqual(record.entry_slippage, 10_000.0 * slippage_pct)
        self.assertAlmostEqual(record.exit_commission, 11_000.0 * commission_pct)
        self.assertAlmostEqual(record.exit_slippage, 11_000.0 * slippage_pct)
        ## Price PnL +1000, then both fee legs.
        self.assertAlmostEqual(record.return_pct, 0.1)
        self.assertAlmostEqual(record.pnl, 1_000.0 - entry_fee - exit_fee)
        self.assertAlmostEqual(record.mae, 0.0)
        self.assertAlmostEqual(record.mfe, 0.1)

        legacy = record.to_legacy_events()
        self.assertEqual(legacy[0]["commission"], record.entry_commission)
        self.assertEqual(legacy[0]["slippage"], record.entry_slippage)
        self.assertEqual(legacy[1]["commission"], record.exit_commission)
        self.assertEqual(legacy[1]["slippage"], record.exit_slippage)
        self.assertAlmostEqual(legacy[1]["return_pct"], 0.1)
        ## Ending cash: start 10000, buy 10000, pay entry_fee, sell 11000, pay exit_fee.
        self.assertAlmostEqual(result.cash.iloc[-1], 10_000.0 - entry_fee + 1_000.0 - exit_fee)

    def test_buying_power_reserves_fees_on_longs(self) -> None:
        """With min_cash=0 and fees, a full-equity long scales so cash never goes negative."""
        data = _make_dataset([100.0, 100.0])
        strategy = _ScriptedStrategy(data, opens=[0], closes=[1])
        result = Simulator(
            [strategy],
            capital=10_000.0,
            percent_of_capital=1.0,
            min_cash=0.0,
            buying_power_policy=BuyingPowerPolicy.SCALE_DOWN,
            commission_pct=0.01,
            slippage_pct=0.0,
        ).run()
        record = next(iter(result.trades.values()))[0]
        ## notional * 1.01 <= 10000 => notional <= 10000/1.01
        self.assertAlmostEqual(record.notional, 10_000.0 / 1.01, places=6)
        self.assertGreaterEqual(result.cash.min(), -1e-9)


class TestSimulatorStartDate(unittest.TestCase):
    """start_date trims the calendar; None falls back to the strategies' start_trading_date."""

    def test_none_uses_full_calendar(self) -> None:
        """Without any start date, equity is every bar plus the prior flat seed day."""
        strategy = _ScriptedStrategy(_make_dataset([100.0] * 5), opens=[], closes=[])
        result = Simulator([strategy], capital=1.0, percent_of_capital=1.0).run()
        self.assertEqual(len(result.equity), 6)
        self.assertAlmostEqual(float(result.equity.iloc[0]), 1.0)

    def test_start_date_trims_calendar_and_skips_earlier_signals(self) -> None:
        """Trading starts at start_date; equity seeds one bus day earlier; prior signals skip."""
        data = _make_dataset([100.0, 100.0, 100.0, 120.0, 120.0])
        strategy = _ScriptedStrategy(data, opens=[0], closes=[4])
        start = data.data.index[2]
        seed = start - pd.tseries.offsets.BDay(1)

        result = Simulator([strategy], capital=1.0, percent_of_capital=1.0, start_date=start.strftime("%Y-%m-%d")).run()

        self.assertEqual(result.eq_start_date, seed)
        self.assertEqual(len(result.equity), 4)
        self.assertEqual(result.num_trades, 0)
        self.assertAlmostEqual(result.equity.iloc[-1], 1.0)
        self.assertAlmostEqual(float(result.equity.iloc[0]), 1.0)

    def test_none_defaults_to_earliest_strategy_start(self) -> None:
        """Without start_date, warmup bars before every strategy's start_trading_date are dropped."""
        data_a = _make_dataset([100.0] * 6, ticker="A")
        data_b = _make_dataset([100.0] * 6, ticker="B")
        idx = data_a.data.index
        strat_a = _ScriptedStrategy(data_a, start_trading_date=idx[3].strftime("%Y-%m-%d"), opens=[], closes=[])
        strat_b = _ScriptedStrategy(data_b, start_trading_date=idx[2].strftime("%Y-%m-%d"), opens=[], closes=[])

        result = Simulator([strat_a, strat_b], capital=1.0, percent_of_capital=0.5).run()

        seed = idx[2] - pd.tseries.offsets.BDay(1)
        self.assertEqual(result.eq_start_date, seed)
        for series in (result.equity, result.cash, result.cash_utilization, result.gross_exposure):
            self.assertEqual(series.index.min(), seed)
        self.assertEqual(result.strategy_pnl.index.min(), seed)

    def test_day_one_open_leaves_starting_equity_at_capital(self) -> None:
        """A same-day open on the first trade bar does not change equity.iloc[0]."""
        data = _make_dataset([100.0, 110.0])
        strategy = _ScriptedStrategy(data, opens=[0], closes=[1])
        capital = 10_000.0
        result = Simulator(
            [strategy],
            capital=capital,
            percent_of_capital=1.0,
            min_cash=None,
            wipeout_floor=-np.inf,
        ).run()

        self.assertAlmostEqual(float(result.equity.iloc[0]), capital)
        self.assertAlmostEqual(float(result.cash.iloc[0]), capital)
        ## First trade day is the second equity point; long open at unchanged price.
        self.assertEqual(result.equity.index[1], data.data.index[0])
        self.assertAlmostEqual(float(result.equity.iloc[1]), capital)
        self.assertEqual(result.num_trades, 1)

    def test_start_date_after_data_raises(self) -> None:
        """A start_date past every strategy's data is rejected at run time."""
        strategy = _ScriptedStrategy(_make_dataset([100.0] * 3), opens=[], closes=[])
        sim = Simulator([strategy], capital=1.0, percent_of_capital=1.0, start_date="2030-01-01")
        with self.assertRaises(ValueError):
            sim.run()


class TestSimulatorCapitalLimits(unittest.TestCase):
    """min_cash and the buying-power policy bound same-day opens."""

    def _two_strategies(self) -> list[StrategyBase]:
        """Return two strategies that both open on bar 0 and close on bar 2."""
        return [
            _ScriptedStrategy(_make_dataset([100.0, 100.0, 100.0], "AAA"), opens=[0], closes=[2]),
            _ScriptedStrategy(_make_dataset([100.0, 100.0, 100.0], "BBB"), opens=[0], closes=[2]),
        ]

    def test_min_cash_zero_skips_second_open(self) -> None:
        """At 60% each with no borrowing, the second open is skipped."""
        result = Simulator(self._two_strategies(), capital=1.0, percent_of_capital=0.6, min_cash=0.0).run()
        self.assertEqual(result.num_trades, 1)
        self.assertEqual(len(result.skipped), 1)
        self.assertGreaterEqual(result.cash.min(), 0.0)
        ## After the first 0.6 open, 0.4 cash remains — not enough for another 0.6.
        self.assertAlmostEqual(float(result.skipped.iloc[0]["available_cash"]), 0.4)
        self.assertIn("available_cash", result.skipped.columns)

    def test_scale_down_fills_remaining_cash(self) -> None:
        """SCALE_DOWN shrinks the second open to the 40% of cash left."""
        result = Simulator(
            self._two_strategies(),
            capital=1.0,
            percent_of_capital=0.6,
            min_cash=0.0,
            buying_power_policy=BuyingPowerPolicy.SCALE_DOWN,
        ).run()
        self.assertEqual(result.num_trades, 2)
        self.assertAlmostEqual(sorted(result.trades_df["notional"])[0], 0.4)
        self.assertAlmostEqual(result.cash_utilization.max(), 1.0)

    def test_no_floor_allows_borrowing(self) -> None:
        """With min_cash=None both opens fill and cash goes negative."""
        result = Simulator(self._two_strategies(), capital=1.0, percent_of_capital=0.6, min_cash=None).run()
        self.assertEqual(result.num_trades, 2)
        self.assertAlmostEqual(result.cash.min(), -0.2)


class TestCashStats(unittest.TestCase):
    """compute_cash_stats on a hand-checkable series."""

    def test_hand_computed_values(self) -> None:
        """Seven days at equity 100 with known cash levels."""
        cash = pd.Series([100, 50, 5, 8, 30, 100, 2], dtype=float)
        equity = pd.Series([100.0] * 7)
        gross = pd.Series([0, 50, 95, 92, 70, 0, 98], dtype=float)

        stats = compute_cash_stats(cash, equity, gross)

        self.assertAlmostEqual(stats["avg_cash_pct"], 2.95 / 7)
        self.assertAlmostEqual(stats["median_cash_pct"], 0.30)
        self.assertAlmostEqual(stats["min_cash_pct"], 0.02)
        self.assertAlmostEqual(stats["pct_days_cash_below_10pct"], 3 / 7)
        self.assertAlmostEqual(stats["pct_days_all_cash"], 2 / 7)
        self.assertEqual(stats["longest_streak_cash_below_10%_days"], 2)
        self.assertAlmostEqual(stats["min_cash"], 2.0)


class TestExposureStats(unittest.TestCase):
    """compute_exposure_stats and Simulator exposure series."""

    def test_hand_computed_portfolio_and_strategy(self) -> None:
        """Three days at equity 100: flat, half long, fully long in one strategy."""
        eq = pd.Series([100.0, 100.0, 100.0])
        gross = pd.Series([0.0, 50.0, 100.0])
        net = gross.copy()
        long = gross.copy()
        short = pd.Series([0.0, 0.0, 0.0])
        lev = gross / eq
        n_open = pd.Series([0, 1, 1], dtype=float)
        strat = pd.DataFrame({"A": [0.0, 50.0, 100.0]})

        port, by_s = compute_exposure_stats(eq, gross, net, long, short, lev, n_open, strat)

        self.assertAlmostEqual(port["avg_leverage"], 0.5)
        self.assertAlmostEqual(port["max_leverage"], 1.0)
        self.assertAlmostEqual(port["max_gross_exposure"], 100.0)
        self.assertAlmostEqual(port["pct_days_flat"], 1 / 3)
        self.assertAlmostEqual(port["pct_days_leverage_above_1"], 0.0)
        self.assertAlmostEqual(by_s.loc["avg_abs_exposure", "A"], 50.0)
        self.assertAlmostEqual(by_s.loc["max_share_of_gross", "A"], 1.0)
        self.assertAlmostEqual(by_s.loc["pct_days_held", "A"], 2 / 3)

    def test_child_keys_roll_up_to_sleeve(self) -> None:
        """mom:AAPL + mom:NVDA summarize as one mom column; mr stays separate."""
        eq = pd.Series([100.0, 100.0, 100.0])
        gross = pd.Series([0.0, 50.0, 100.0])
        net = gross.copy()
        long = gross.copy()
        short = pd.Series([0.0, 0.0, 0.0])
        lev = gross / eq
        n_open = pd.Series([0, 1, 2], dtype=float)
        strat = pd.DataFrame(
            {
                "mom:AAPL": [0.0, 50.0, 40.0],
                "mom:NVDA": [0.0, 0.0, 40.0],
                "mr:AAPL": [0.0, 0.0, 20.0],
            }
        )

        _, by_s = compute_exposure_stats(eq, gross, net, long, short, lev, n_open, strat)

        self.assertEqual(list(by_s.columns), ["mom", "mr"])
        ## Day 3: mom abs = 40+40 = 80; mr abs = 20.
        self.assertAlmostEqual(by_s.loc["max_abs_exposure", "mom"], 80.0)
        self.assertAlmostEqual(by_s.loc["max_abs_exposure", "mr"], 20.0)
        self.assertAlmostEqual(by_s.loc["pct_days_held", "mom"], 2 / 3)
        self.assertAlmostEqual(by_s.loc["pct_days_held", "mr"], 1 / 3)

    def test_solo_strategy_exposure_matches_gross(self) -> None:
        """One long strategy: abs strategy exposure equals portfolio gross every day."""
        strategy = _ScriptedStrategy(_make_dataset([100.0, 110.0, 100.0, 90.0]), opens=[0], closes=[2])
        result = Simulator([strategy], capital=100.0, percent_of_capital=1.0).run()

        self.assertEqual(list(result.strategy_exposure.columns), ["TEST"])
        np.testing.assert_allclose(
            result.strategy_exposure["TEST"].abs().to_numpy(),
            result.gross_exposure.to_numpy(),
        )
        self.assertAlmostEqual(result.exposure_stats["max_gross_exposure"], result.gross_exposure.max())
        self.assertAlmostEqual(result.leverage.max(), result.exposure_stats["max_leverage"])

    def test_two_strategies_abs_sum_equals_gross(self) -> None:
        """Two longs: sum of abs strategy exposures equals gross each day."""
        a = _ScriptedStrategy(_make_dataset([100.0] * 5, "AAA"), ticker="AAA", opens=[0], closes=[3])
        b = _ScriptedStrategy(_make_dataset([100.0] * 5, "BBB"), ticker="BBB", opens=[1], closes=[3])
        result = Simulator([a, b], capital=100.0, percent_of_capital=0.5).run()

        abs_sum = result.strategy_exposure.abs().sum(axis=1)
        np.testing.assert_allclose(abs_sum.to_numpy(), result.gross_exposure.to_numpy())
        self.assertEqual(set(result.strategy_exposure.columns), {"AAA", "BBB"})
        self.assertIn("avg_share_of_gross", result.exposure_stats_by_strategy.index)

    def test_track_exposure_false_skips_series_keeps_equity(self) -> None:
        """Turning tracking off empties exposure fields but leaves the same equity path."""
        strategy = _ScriptedStrategy(_make_dataset([100.0, 110.0, 100.0]), opens=[0], closes=[2])
        on = Simulator([strategy], capital=100.0, percent_of_capital=1.0, track_exposure=True).run()
        off = Simulator(
            [_ScriptedStrategy(_make_dataset([100.0, 110.0, 100.0]), opens=[0], closes=[2])],
            capital=100.0,
            percent_of_capital=1.0,
            track_exposure=False,
        ).run()

        pd.testing.assert_series_equal(on.equity, off.equity)
        self.assertTrue(off.strategy_exposure.empty)
        self.assertTrue(off.exposure_stats.empty)
        self.assertTrue(off.net_exposure.empty)
        self.assertFalse(on.strategy_exposure.empty)
        ## Gross used by cash_stats / leverage cap is still recorded.
        self.assertFalse(off.gross_exposure.empty)

    def test_stats_level_lean_keeps_equity_score_keys(self) -> None:
        """Lean skips trade/cash/exposure summaries but keeps sharpe_ratio and equity."""
        full = Simulator(
            [_ScriptedStrategy(_make_dataset([100.0, 110.0, 100.0]), opens=[0], closes=[2])],
            capital=100.0,
            percent_of_capital=1.0,
            stats_level="full",
        ).run()
        lean = Simulator(
            [_ScriptedStrategy(_make_dataset([100.0, 110.0, 100.0]), opens=[0], closes=[2])],
            capital=100.0,
            percent_of_capital=1.0,
            stats_level="lean",
        ).run()

        pd.testing.assert_series_equal(full.equity, lean.equity)
        self.assertIn("sharpe_ratio", lean.full_stats.index)
        self.assertAlmostEqual(
            float(full.full_stats["sharpe_ratio"]),
            float(lean.full_stats["sharpe_ratio"]),
        )
        self.assertTrue(lean.trades_all.empty)
        self.assertTrue(lean.cash_stats.empty)
        self.assertTrue(lean.exposure_stats.empty)
        self.assertTrue(lean.strategy_exposure.empty)
        self.assertFalse(full.trades_all.empty)

    def test_stats_level_rejects_unknown(self) -> None:
        """Unknown stats_level fails before the run starts."""
        with self.assertRaises(ValueError):
            Simulator(
                [_ScriptedStrategy(_make_dataset([100.0, 110.0]), opens=[0], closes=[1])],
                capital=100.0,
                percent_of_capital=1.0,
                stats_level="partial",  # type: ignore[arg-type]
            )


class TestExtendedStats(unittest.TestCase):
    """drawdown_stats, yearly_returns, and extended_stats on hand-checkable inputs."""

    def test_drawdown_stats(self) -> None:
        """Peak 1.2 -> trough 0.9 -> recovery: depth 0.3, one day under water."""
        eq = pd.Series([1.0, 1.2, 0.9, 1.2, 1.3], index=pd.date_range("2024-01-01", periods=5))
        stats = drawdown_stats(eq)
        self.assertAlmostEqual(stats["max_drawdown_value"], 0.3)
        self.assertAlmostEqual(stats["avg_drawdown"], 0.25 / 5)
        self.assertEqual(stats["max_drawdown_duration_days"], 1.0)
        self.assertAlmostEqual(stats["avg_drawdown_duration_days"], 0.2)

    def test_yearly_returns_measure_within_each_year(self) -> None:
        """Each year runs from its own first to last value."""
        eq = pd.Series([1.0, 1.1, 1.21, 1.331], index=pd.to_datetime(["2023-01-02", "2023-12-29", "2024-01-02", "2024-12-31"]))
        yearly = yearly_returns(eq)
        self.assertAlmostEqual(yearly[2023], 0.1)
        self.assertAlmostEqual(yearly[2024], 0.1)

    def test_trade_extras(self) -> None:
        """Streaks, SQN, percentiles, durations, exposure, and dollar profit factor."""
        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        eq = pd.Series(1.0, index=idx)
        trades = pd.DataFrame(
            {
                "entry_date": idx[[0, 2, 4, 6]],
                "exit_date": idx[[1, 3, 5, 8]],
                "return_pct": [0.10, 0.05, -0.02, 0.0],
                "pnl": [10.0, 5.0, -2.0, -1.0],
            }
        )
        stats = extended_stats(eq, trades)
        r = trades["return_pct"]
        self.assertEqual(stats["winning_streak"], 2)
        self.assertEqual(stats["losing_streak"], 2)
        self.assertAlmostEqual(stats["sqn"], r.mean() * 2 / r.std(ddof=0))
        self.assertAlmostEqual(stats["p25_trade"], r.quantile(0.25))
        self.assertEqual(stats["max_loss_duration"], 2)
        self.assertEqual(stats["avg_win_duration"], 1)
        ## Bars 0-1, 2-3, 4-5, 6-8 are inside a trade; bar 9 is not.
        self.assertAlmostEqual(stats["exposure_time"], 0.9)
        self.assertAlmostEqual(stats["profit_factor_dollar"], 15.0 / 3.0)

    def test_zero_risk_free_matches_eq_stats(self) -> None:
        """With risk_free_rate=0 the rf-adjusted ratios equal eq_stats' ratios."""
        eq = pd.Series([1.0, 1.02, 0.99, 1.05, 1.03, 1.08], index=pd.date_range("2024-01-01", periods=6))
        ext = extended_stats(eq, pd.DataFrame())
        base = eq_stats(eq)
        self.assertAlmostEqual(ext["sharpe_ratio_rf"], base["sharpe_ratio"])
        self.assertAlmostEqual(ext["sortino_ratio_rf"], base["sortino_ratio"])

    def test_simulator_full_stats_combines_all_three(self) -> None:
        """full_stats holds equity, trade, and extended stats with clashing trade keys prefixed."""
        strategy = _ScriptedStrategy(_make_dataset([100.0, 110.0, 121.0]), opens=[0], closes=[2])
        result = Simulator([strategy], capital=1.0, percent_of_capital=1.0).run()
        full = result.full_stats

        self.assertFalse(full.index.duplicated().any())
        self.assertAlmostEqual(full["cagr"], result.equity_stats["cagr"])
        self.assertEqual(full["num_trades"], result.trades_all["num_trades"])
        self.assertEqual(full["win_rate"], result.equity_stats["win_rate"])
        self.assertEqual(full["trade_win_rate"], result.trades_all["win_rate"])
        self.assertAlmostEqual(full["buy_hold_return"], 0.21)
        self.assertAlmostEqual(full["equity_final"], 1.21)
        self.assertIn("return_2020", full.index)


if __name__ == "__main__":
    unittest.main()
