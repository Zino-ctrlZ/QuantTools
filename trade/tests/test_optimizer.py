"""Tests for the grid-search ParamOptimizer: spaces, batching, scoring, per-asset loops, and fork parity."""

from __future__ import annotations

import unittest
import warnings
from typing import Optional
from unittest.mock import patch

import numpy as np
import pandas as pd

from EventDriven.types import PositionEffect, SignalID
from trade.backtester_._multi_asset_strategy import MultiAssetStrategy
from trade.backtester_._strategy import StrategyBase, TradeDecision
from trade.backtester_._types import SideInt
from trade.backtester_.data import PTDataset
from trade.backtester_.optimizer import ParamOptimizer, SOLO_SIMULATOR_KWARGS, optimize_per_ticker
from trade.backtester_.optimizer import _pool
from trade.backtester_.optimizer._pool import _make_batches
from trade.backtester_.simulator import Simulator
from trade.backtester_.systematic import Sleeve, SystematicMultiAsset


def _make_dataset(ticker: str, seed: int, n: int = 300) -> PTDataset:
    """Build a random-walk OHLCV dataset on business days."""
    rng = np.random.default_rng(seed)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    frame = pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1_000}, index=dates
    )
    return PTDataset(ticker, frame)


class _MeanRevert(StrategyBase):
    """Buy below a moving average by a threshold, sell after a fixed hold."""

    bt_params = {"ma_length": 10, "threshold": 0.02, "hold": 5, "explode": False}

    def __init__(
        self,
        data: PTDataset,
        start_trading_date: Optional[str] = None,
        ticker: Optional[str] = None,
        tplusn: int = 0,
        *,
        ma_length: int = 10,
        threshold: float = 0.02,
        hold: int = 5,
        explode: bool = False,
    ) -> None:
        """Store parameters and initialize the base strategy."""
        if explode:
            raise RuntimeError("bad combination")
        self.ma_length = int(ma_length)
        self.threshold = float(threshold)
        self.hold = int(hold)
        self._signal_id = SignalID("test::USO20200101LONG")
        self._entry_index: Optional[int] = None
        super().__init__(data, start_trading_date=start_trading_date, ticker=ticker or data.name, tplusn=tplusn)

    def setup(self) -> None:
        """Precompute the moving average."""
        close = pd.Series(self._close)
        self._ma = close.rolling(self.ma_length).mean().to_numpy()

    def open_action(self, *, signal_id=None, entry_price=None, side=None, date=None, index=None) -> None:
        """Remember the entry bar."""
        self._entry_index = index
        super().open_action(signal_id=signal_id, entry_price=entry_price, side=side, date=date, index=index)

    def close_action(self, *, date=None, index=None) -> None:
        """Forget the entry bar."""
        self._entry_index = None
        super().close_action(date=date, index=index)

    def reset_strategy_state(self) -> None:
        """Clear the entry bar along with base state."""
        self._entry_index = None
        super().reset_strategy_state()

    def is_open_signal(self, *, date: pd.Timestamp = None, index: int = None) -> bool:
        """Close is below the average by more than the threshold."""
        i, _ = self._resolve(date=date, index=index)
        ma = self._ma[i]
        return bool(not np.isnan(ma) and self._close[i] < ma * (1 - self.threshold))

    def is_close_signal(self, *, date: pd.Timestamp = None, index: int = None) -> bool:
        """Held for ``hold`` bars."""
        i, _ = self._resolve(date=date, index=index)
        return self._entry_index is not None and i - self._entry_index >= self.hold

    def should_open(self, *, date: pd.Timestamp = None, index: int = None) -> TradeDecision:
        """Open long on signal when flat."""
        if self.position_open or not self.is_open_signal(date=date, index=index):
            return TradeDecision(ok=False, side=0)
        return TradeDecision(ok=True, side=int(SideInt.BUY), pos_effect=PositionEffect.OPEN, signal_id=self._signal_id)

    def should_close(self, *, date: pd.Timestamp = None, index: int = None) -> TradeDecision:
        """Close after the hold period."""
        if not self.position_open or not self.is_close_signal(date=date, index=index):
            return TradeDecision(ok=False, side=0)
        return TradeDecision(ok=True, side=int(SideInt.BUY), pos_effect=PositionEffect.CLOSE, signal_id="N/A")


DATA = {"AAA": _make_dataset("AAA", 1), "BBB": _make_dataset("BBB", 2)}
SIM_KWARGS = {"capital": 100_000, "percent_of_capital": 0.5}


SOLO_KWARGS = {"capital": 100_000, "percent_of_capital": 1.0, "min_cash": None, "wipeout_floor": -np.inf}


def _opt(space: Optional[dict] = None, **kwargs) -> ParamOptimizer:
    """Build a one-process optimizer over both tickers with no progress bar."""
    space = space if space is not None else {"ma_length": [5, 10, 20], "threshold": [0.01, 0.03]}
    kwargs.setdefault("n_jobs", 1)
    sim_kwargs = kwargs.pop("simulator_kwargs", SIM_KWARGS)
    return ParamOptimizer(
        _MeanRevert, data=DATA, space=space, start_trading_date="2020-03-02",
        simulator_kwargs=sim_kwargs, show_progress=False, **kwargs,
    )


def _manual(params: dict, tickers: tuple, sim_kwargs: dict) -> pd.Series:
    """Run a hand-built Simulator and return its full_stats."""
    strategies = [_MeanRevert(DATA[t], start_trading_date="2020-03-02", ticker=t, **params) for t in tickers]
    return Simulator(strategies, **sim_kwargs).run().full_stats


class TestSpace(unittest.TestCase):
    """The optimizer turns its space dict into mixes, filters, and validates them."""

    def test_grid_product_and_filter(self) -> None:
        """Every mix in odometer order, then param_filter drops the unwanted ones."""
        opt = _opt({"ma_length": [1, 2], "hold": [3, 4]})
        self.assertEqual([tuple(c.values()) for c in opt.combos], [(1, 3), (1, 4), (2, 3), (2, 4)])
        filtered = _opt({"ma_length": [1, 2], "hold": [1, 2]}, param_filter=lambda p: p["ma_length"] < p["hold"])
        self.assertEqual(filtered.combos, [{"ma_length": 1, "hold": 2}])

    def test_empty_space_means_fixed(self) -> None:
        """An empty dict is one empty mix, so the strategy runs untuned."""
        self.assertEqual(_opt({}).combos, [{}])

    def test_filter_removing_everything_raises(self) -> None:
        """Nothing left to test is an error, not a silent zero-trial run."""
        with self.assertRaises(ValueError):
            _opt({"ma_length": [1]}, param_filter=lambda p: False)

    def test_unknown_param_name_raises(self) -> None:
        """A misspelled parameter fails when the optimizer is made, before any trial."""
        with self.assertRaises(ValueError):
            _opt({"ma_lenght": [5]})


class TestBatches(unittest.TestCase):
    """Batches are lists of settings dicts, capped in size, with nothing dropped."""

    def test_cap_makes_more_batches_not_fewer_trials(self) -> None:
        """1,000 dicts on 8 workers: capped at 20 per batch, 50 batches, same order."""
        combos = [{"a": i} for i in range(1000)]
        batches = _make_batches(combos, 8)
        self.assertEqual(len(batches), 50)
        self.assertTrue(all(len(b) == 20 for b in batches))
        self.assertEqual([c for b in batches for c in b], combos)

    def test_second_run_blocked(self) -> None:
        """While a run is in progress, starting another one raises."""
        with patch.object(_pool, "_RUN", object()):
            with self.assertRaises(RuntimeError):
                _pool.run_trials(
                    [{}],
                    {},
                    build=_opt()._build,
                    tickers=["AAA"],
                    n_jobs=1,
                    show_progress=False,
                )


class TestBook(unittest.TestCase):
    """One mix wins for the whole book."""

    def test_table_indexed_by_values_and_best_matches_manual(self) -> None:
        """Rows are labelled by (ma_length, threshold); the winner equals a hand-built run."""
        res = _opt().run()
        self.assertEqual(list(res.trials.index.names), ["ma_length", "threshold"])
        self.assertEqual(len(res.trials), 6)
        self.assertAlmostEqual(res.best_score, res.trials["score"].max())
        params = res.best_params
        ## Picks are the user's own values, not numpy numbers read back off the index.
        self.assertIs(type(params["ma_length"]), int)
        manual = _manual(params, ("AAA", "BBB"), SIM_KWARGS)
        self.assertAlmostEqual(manual["sharpe_ratio"], res.best_score)
        self.assertAlmostEqual(res.trials.loc[tuple(params.values()), "score"], res.best_score)

    def test_constraints_exclude_but_keep_rows(self) -> None:
        """Mixes that break a constraint stay in the table but cannot win.

        Uses ``stats_level="full"`` because ``num_trades`` is a trade-summary key
        that lean trials omit from worker ``full_stats``.
        """
        res = _opt(
            constraints=[("num_trades", ">=", 15)],
            simulator_kwargs={**SIM_KWARGS, "stats_level": "full"},
        ).run()
        few = res.trials["num_trades"] < 15
        ## The test only means something if the constraint actually cuts some mixes.
        self.assertTrue(few.any() and (~few).any())
        self.assertFalse(res.trials.loc[few, "feasible"].any())
        self.assertGreaterEqual(res.best_result.full_stats["num_trades"], 15)

    def test_failed_mix_reported_not_fatal(self) -> None:
        """A mix that raises becomes a failed row; the rest of the grid still runs."""
        res = _opt({"ma_length": [5, 10], "explode": [False, True]}).run()
        failed = res.trials[res.trials["error"].notna()]
        self.assertEqual(len(failed), 2)
        self.assertTrue(all(label[1] for label in failed.index))
        self.assertIn("bad combination", failed["error"].iloc[0])
        self.assertFalse(res.best_params["explode"])

    def test_no_passing_mix_warns(self) -> None:
        """When nothing passes, best_params is empty and a warning says why."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = _opt(constraints=[lambda s: False]).run()
        self.assertEqual(res.best_params, {})
        self.assertIsNone(res.best_result)
        self.assertTrue(any("No trial passed" in str(w.message) for w in caught))

    def test_lean_defaults_match_full_winner_and_keep_top_is_rich(self) -> None:
        """Default lean trials pick the same mix as an all-full grid; keep_top is full."""
        lean = _opt(space={"ma_length": [5, 10], "threshold": [0.01, 0.03]}, keep_top=1).run()
        full = _opt(
            space={"ma_length": [5, 10], "threshold": [0.01, 0.03]},
            keep_top=1,
            simulator_kwargs={**SIM_KWARGS, "stats_level": "full", "track_exposure": True},
        ).run()

        self.assertEqual(lean.best_params, full.best_params)
        self.assertAlmostEqual(lean.best_score, full.best_score)
        ## Lean worker stats omit trade summaries; keep_top re-run restores them.
        self.assertNotIn("num_trades", lean.trials.columns)
        self.assertFalse(lean.best_result.trades_all.empty)
        self.assertFalse(lean.best_result.exposure_stats.empty)

    def test_trial_kwargs_setdefault_full_kwargs_force(self) -> None:
        """Trials fill lean defaults; keep_top always forces full + exposure."""
        opt = _opt()
        trial = opt._trial_kwargs()
        self.assertEqual(trial["stats_level"], "lean")
        self.assertFalse(trial["track_exposure"])
        self.assertEqual(trial["capital"], 100_000)

        forced = ParamOptimizer(
            _MeanRevert,
            data=DATA,
            space={"ma_length": [5]},
            start_trading_date="2020-03-02",
            simulator_kwargs={**SIM_KWARGS, "stats_level": "lean", "track_exposure": False},
            n_jobs=1,
            show_progress=False,
        )
        full = forced._full_kwargs()
        self.assertEqual(full["stats_level"], "full")
        self.assertTrue(full["track_exposure"])


class TestBookFactory(unittest.TestCase):
    """Trials can build a PortfolioBook so simulator hooks apply."""

    @staticmethod
    def _factory(params: dict) -> SystematicMultiAsset:
        """Build a fresh mr sleeve book with the trial's child params and a tight open cap."""
        return SystematicMultiAsset(
            [Sleeve("mr", _MeanRevert, params=params)],
            DATA,
            start_trading_date="2020-03-02",
            max_open_trades=1,
        )

    def test_book_factory_matches_manual_book_and_keep_top_is_full(self) -> None:
        """Winner equals a hand-built Simulator(book); keep_top restores full stats."""
        res = ParamOptimizer(
            space={"ma_length": [5, 10], "threshold": [0.01, 0.03]},
            simulator_kwargs=SIM_KWARGS,
            book_factory=self._factory,
            keep_top=1,
            n_jobs=1,
            show_progress=False,
        ).run()
        manual = Simulator(self._factory(res.best_params), **SIM_KWARGS).run()
        self.assertAlmostEqual(manual.full_stats["sharpe_ratio"], res.best_score)
        self.assertFalse(res.best_result.trades_all.empty)
        self.assertFalse(res.best_result.exposure_stats.empty)
        ## Cap of 1 open must have bitten; otherwise this is not exercising book hooks.
        self.assertTrue(
            (not res.best_result.skipped.empty)
            and res.best_result.skipped["reason"].astype(str).str.contains("max_open_trades").any()
        )

    def test_book_factory_differs_from_flat_list(self) -> None:
        """Same space: capped book and uncapped flat list do not pick the same score path."""
        space = {"ma_length": [5, 10], "threshold": [0.01, 0.03]}
        book = ParamOptimizer(
            space=space,
            simulator_kwargs=SIM_KWARGS,
            book_factory=self._factory,
            keep_top=1,
            n_jobs=1,
            show_progress=False,
        ).run()
        flat = ParamOptimizer(
            _MeanRevert,
            data=DATA,
            space=space,
            start_trading_date="2020-03-02",
            simulator_kwargs=SIM_KWARGS,
            keep_top=1,
            n_jobs=1,
            show_progress=False,
        ).run()
        ## Cap changes fills, so either the winner or its score should diverge.
        self.assertTrue(
            book.best_params != flat.best_params or not np.isclose(book.best_score, flat.best_score)
        )

    def test_book_factory_or_flat_required(self) -> None:
        """Without book_factory, strategy_class/data/start_trading_date are required."""
        with self.assertRaises(ValueError):
            ParamOptimizer(space={"ma_length": [5]}, simulator_kwargs=SIM_KWARGS)


class TestOneTickerBook(unittest.TestCase):
    """Per-asset tuning is a loop: one optimizer per ticker, each a one-ticker book."""

    def test_loop_per_ticker_matches_solo_runs(self) -> None:
        """Each ticker's pick is its best solo mix, scored exactly like a hand-built solo run."""
        for ticker in ("AAA", "BBB"):
            res = ParamOptimizer(
                _MeanRevert, data={ticker: DATA[ticker]}, space={"ma_length": [5, 10, 20], "threshold": [0.01, 0.03]},
                start_trading_date="2020-03-02", simulator_kwargs=SOLO_KWARGS, n_jobs=1, show_progress=False,
            ).run()
            self.assertEqual(res.best_params, dict(zip(["ma_length", "threshold"], res.trials["score"].idxmax())))
            solo = _manual(res.best_params, (ticker,), SOLO_KWARGS)
            self.assertAlmostEqual(solo["sharpe_ratio"], res.best_score)


class TestMultiAssetOptimize(unittest.TestCase):
    """MultiAssetStrategy.optimize is the per-ticker ParamOptimizer loop."""

    def test_optimize_matches_optimize_per_ticker_and_solo(self) -> None:
        """Wrapper results match optimize_per_ticker and a hand-built solo Simulator."""
        space = {"ma_length": [5, 10], "threshold": [0.01, 0.03]}
        multi = MultiAssetStrategy(
            name="mr",
            start_date="2020-03-02",
            params={t: {"ma_length": 10, "threshold": 0.02, "hold": 5} for t in DATA},
            strategy_class=_MeanRevert,
            data=DATA,
            tplusn=0,
        )
        wrapped = multi.optimize(space, simulator_kwargs=SOLO_KWARGS, n_jobs=1, show_progress=False, keep_top=1)
        direct = optimize_per_ticker(
            _MeanRevert,
            DATA,
            space,
            start_trading_date="2020-03-02",
            simulator_kwargs=SOLO_KWARGS,
            base_params=multi.params,
            n_jobs=1,
            show_progress=False,
            keep_top=1,
        )
        self.assertEqual(set(wrapped), set(DATA))
        for ticker in DATA:
            self.assertEqual(wrapped[ticker].best_params, direct[ticker].best_params)
            self.assertAlmostEqual(wrapped[ticker].best_score, direct[ticker].best_score)
            solo = _manual(wrapped[ticker].best_params, (ticker,), SOLO_KWARGS)
            self.assertAlmostEqual(solo["sharpe_ratio"], wrapped[ticker].best_score)

    def test_default_simulator_kwargs_are_solo(self) -> None:
        """Omitting simulator_kwargs uses the shared solo defaults."""
        self.assertEqual(SOLO_SIMULATOR_KWARGS["percent_of_capital"], 1.0)
        self.assertIsNone(SOLO_SIMULATOR_KWARGS["min_cash"])


class TestSystematicOptimize(unittest.TestCase):
    """SystematicMultiAsset.optimize: shared book by default, per_asset loop optional."""

    def test_shared_optimize_uses_book_hooks(self) -> None:
        """Shared search goes through with_trial_params so max_open_trades still bites."""
        book = SystematicMultiAsset(
            [Sleeve("mr", _MeanRevert)],
            DATA,
            start_trading_date="2020-03-02",
            max_open_trades=1,
        )
        res = book.optimize(
            {"ma_length": [5, 10], "threshold": [0.01, 0.03]},
            simulator_kwargs=SIM_KWARGS,
            n_jobs=1,
            show_progress=False,
            keep_top=1,
        )
        manual = Simulator(book.with_trial_params(res.best_params), **SIM_KWARGS).run()
        self.assertAlmostEqual(manual.full_stats["sharpe_ratio"], res.best_score)
        self.assertTrue(
            (not res.best_result.skipped.empty)
            and res.best_result.skipped["reason"].astype(str).str.contains("max_open_trades").any()
        )

    def test_shared_optimize_book_params(self) -> None:
        """Space keys that are book bt_params go through with_params."""
        book = SystematicMultiAsset(
            [Sleeve("mr", _MeanRevert, params={"ma_length": 5, "threshold": 0.01})],
            DATA,
            start_trading_date="2020-03-02",
            max_open_trades=5,
        )
        res = book.optimize(
            {"max_open_trades": [1, 5]},
            simulator_kwargs=SIM_KWARGS,
            n_jobs=1,
            show_progress=False,
            keep_top=1,
        )
        self.assertIn(res.best_params["max_open_trades"], (1, 5))
        rebuilt = book.with_trial_params(res.best_params)
        self.assertEqual(rebuilt.max_open_trades, res.best_params["max_open_trades"])

    def test_per_asset_matches_multiasset(self) -> None:
        """Single-sleeve per_asset picks match MultiAssetStrategy.optimize."""
        space = {"ma_length": [5, 10], "threshold": [0.01, 0.03]}
        params = {t: {"ma_length": 10, "threshold": 0.02, "hold": 5} for t in DATA}
        multi = MultiAssetStrategy(
            name="mr",
            start_date="2020-03-02",
            params=params,
            strategy_class=_MeanRevert,
            data=DATA,
            tplusn=0,
        )
        book = SystematicMultiAsset(
            [Sleeve.from_multiasset(multi)],
            DATA,
            start_trading_date="2020-03-02",
        )
        multi_res = multi.optimize(space, simulator_kwargs=SOLO_KWARGS, n_jobs=1, show_progress=False, keep_top=1)
        book_res = book.optimize(
            space, simulator_kwargs=SOLO_KWARGS, per_asset=True, n_jobs=1, show_progress=False, keep_top=1
        )
        for ticker in DATA:
            self.assertEqual(multi_res[ticker].best_params, book_res[ticker].best_params)
            self.assertAlmostEqual(multi_res[ticker].best_score, book_res[ticker].best_score)

    def test_optimize_rejects_multi_sleeve(self) -> None:
        """Any optimize on a multi-sleeve book fails before trials run."""
        book = SystematicMultiAsset(
            [
                Sleeve("mr", _MeanRevert, params={"ma_length": 5}),
                Sleeve("mr2", _MeanRevert, params={"ma_length": 10}),
            ],
            DATA,
            start_trading_date="2020-03-02",
        )
        with self.assertRaises(ValueError) as ctx:
            book.optimize(
                {"max_open_trades": [1, 5]},
                simulator_kwargs=SIM_KWARGS,
                n_jobs=1,
                show_progress=False,
            )
        self.assertIn("one-sleeve", str(ctx.exception))


@unittest.skipUnless(
    _pool.fork_unsafe_reason() is None, "fork not allowed here (set OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES)"
)
class TestForkParity(unittest.TestCase):
    """Forked workers give exactly the same results as one process."""

    def test_parallel_equals_serial(self) -> None:
        """Same table and pick from 1 process and from 4 forked workers."""
        serial = _opt(n_jobs=1).run()
        forked = _opt(n_jobs=4).run()
        pd.testing.assert_frame_equal(serial.trials, forked.trials)
        self.assertEqual(serial.best_params, forked.best_params)


class TestForkSafetyFallback(unittest.TestCase):
    """A missing macOS fork-safety setting falls back to one process with a warning."""

    def test_missing_env_runs_serially(self) -> None:
        """With the reason set, n_jobs>1 still runs and warns instead of forking."""
        with patch.object(_pool, "fork_unsafe_reason", return_value="test reason"), patch.object(
            _pool, "ProcessPoolExecutor", side_effect=AssertionError("must not fork")
        ), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = _opt(n_jobs=4).run()
        self.assertEqual(len(res.trials), 6)
        self.assertTrue(any("test reason" in str(w.message) for w in caught))


if __name__ == "__main__":
    unittest.main()
