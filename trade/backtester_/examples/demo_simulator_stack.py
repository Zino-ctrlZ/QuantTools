"""End-to-end demo of Simulator / MultiAsset / SystematicMultiAsset / Optimizer.

Self-contained: synthetic prices + tiny ``StrategyBase`` subclasses (no
``strategy_research``). Grids are small so the walk finishes quickly.

Run from a QuantTools-capable env, e.g.::

    python trade/backtester_/examples/demo_simulator_stack.py
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from EventDriven.types import PositionEffect, SignalID
from trade.backtester_._multi_asset_strategy import MultiAssetStrategy
from trade.backtester_._strategy import StrategyBase, TradeDecision
from trade.backtester_._types import SideInt
from trade.backtester_.backtester_ import PTBacktester
from trade.backtester_.data import PTDataset
from trade.backtester_.optimizer import ParamOptimizer, SOLO_SIMULATOR_KWARGS
from trade.backtester_.optimizer.core import OptimizationResult
from trade.backtester_.simulator import BuyingPowerPolicy, SimulationResult, Simulator
from trade.backtester_.systematic import Sleeve, SystematicMultiAsset


# ---------------------------------------------------------------------------
# Toy strategies (drop-in stand-ins for research strategies)
# ---------------------------------------------------------------------------
class LongMeanRevert(StrategyBase):
    """Buy when price is below a moving average by a threshold; exit after ``hold`` bars."""

    bt_params = {"ma_length": 50, "threshold": 0.02, "hold": 10}

    def __init__(
        self,
        data: PTDataset,
        start_trading_date: Optional[str] = None,
        ticker: Optional[str] = None,
        tplusn: int = 0,
        *,
        ma_length: int = 50,
        threshold: float = 0.02,
        hold: int = 10,
    ) -> None:
        """Store mean-reversion knobs and initialize the base strategy."""
        self.ma_length = int(ma_length)
        self.threshold = float(threshold)
        self.hold = int(hold)
        ## Placeholder SignalID; format is {slug}::{TICKER}{YYYYMMDD}{LONG|SHORT}.
        self._signal_id = SignalID("demo_mr::AAPL20190101LONG")
        self._entry_index: Optional[int] = None
        super().__init__(
            data,
            start_trading_date=start_trading_date,
            ticker=ticker or data.name,
            tplusn=tplusn,
        )

    def setup(self) -> None:
        """Precompute the moving average on close."""
        close = pd.Series(self._close)
        self._ma = close.rolling(self.ma_length).mean().to_numpy()

    def open_action(self, *, signal_id=None, entry_price=None, side=None, date=None, index=None) -> None:
        """Remember the entry bar for the hold timer."""
        self._entry_index = index
        super().open_action(signal_id=signal_id, entry_price=entry_price, side=side, date=date, index=index)

    def close_action(self, *, date=None, index=None) -> None:
        """Clear the entry bar."""
        self._entry_index = None
        super().close_action(date=date, index=index)

    def reset_strategy_state(self) -> None:
        """Clear entry state with the base reset."""
        self._entry_index = None
        super().reset_strategy_state()

    def is_open_signal(self, *, date: pd.Timestamp = None, index: int = None) -> bool:
        """Close is below the MA by more than ``threshold``."""
        i, _ = self._resolve(date=date, index=index)
        ma = self._ma[i]
        return bool(not np.isnan(ma) and self._close[i] < ma * (1.0 - self.threshold))

    def is_close_signal(self, *, date: pd.Timestamp = None, index: int = None) -> bool:
        """Held for ``hold`` bars."""
        i, _ = self._resolve(date=date, index=index)
        return self._entry_index is not None and i - self._entry_index >= self.hold

    def should_open(self, *, date: pd.Timestamp = None, index: int = None) -> TradeDecision:
        """Open long on signal when flat."""
        if self.position_open or not self.is_open_signal(date=date, index=index):
            return TradeDecision(ok=False, side=0)
        return TradeDecision(
            ok=True, side=int(SideInt.BUY), pos_effect=PositionEffect.OPEN, signal_id=self._signal_id
        )

    def should_close(self, *, date: pd.Timestamp = None, index: int = None) -> TradeDecision:
        """Close after the hold period."""
        if not self.position_open or not self.is_close_signal(date=date, index=index):
            return TradeDecision(ok=False, side=0)
        return TradeDecision(ok=True, side=int(SideInt.BUY), pos_effect=PositionEffect.CLOSE, signal_id="N/A")


class LongBreakout(StrategyBase):
    """Buy on a close above the prior ``lookback`` high; exit after ``hold`` bars."""

    bt_params = {"lookback": 20, "hold": 15}

    def __init__(
        self,
        data: PTDataset,
        start_trading_date: Optional[str] = None,
        ticker: Optional[str] = None,
        tplusn: int = 0,
        *,
        lookback: int = 20,
        hold: int = 15,
    ) -> None:
        """Store breakout knobs and initialize the base strategy."""
        self.lookback = int(lookback)
        self.hold = int(hold)
        ## Placeholder SignalID; format is {slug}::{TICKER}{YYYYMMDD}{LONG|SHORT}.
        self._signal_id = SignalID("demo_bo::AAPL20190101LONG")
        self._entry_index: Optional[int] = None
        super().__init__(
            data,
            start_trading_date=start_trading_date,
            ticker=ticker or data.name,
            tplusn=tplusn,
        )

    def setup(self) -> None:
        """Precompute the rolling prior high (shifted so it is causal)."""
        high = pd.Series(self._high if hasattr(self, "_high") else self._close)
        self._prior_high = high.rolling(self.lookback).max().shift(1).to_numpy()

    def open_action(self, *, signal_id=None, entry_price=None, side=None, date=None, index=None) -> None:
        """Remember the entry bar for the hold timer."""
        self._entry_index = index
        super().open_action(signal_id=signal_id, entry_price=entry_price, side=side, date=date, index=index)

    def close_action(self, *, date=None, index=None) -> None:
        """Clear the entry bar."""
        self._entry_index = None
        super().close_action(date=date, index=index)

    def reset_strategy_state(self) -> None:
        """Clear entry state with the base reset."""
        self._entry_index = None
        super().reset_strategy_state()

    def is_open_signal(self, *, date: pd.Timestamp = None, index: int = None) -> bool:
        """Close breaks above the prior lookback high."""
        i, _ = self._resolve(date=date, index=index)
        level = self._prior_high[i]
        return bool(not np.isnan(level) and self._close[i] > level)

    def is_close_signal(self, *, date: pd.Timestamp = None, index: int = None) -> bool:
        """Held for ``hold`` bars."""
        i, _ = self._resolve(date=date, index=index)
        return self._entry_index is not None and i - self._entry_index >= self.hold

    def should_open(self, *, date: pd.Timestamp = None, index: int = None) -> TradeDecision:
        """Open long on breakout when flat."""
        if self.position_open or not self.is_open_signal(date=date, index=index):
            return TradeDecision(ok=False, side=0)
        return TradeDecision(
            ok=True, side=int(SideInt.BUY), pos_effect=PositionEffect.OPEN, signal_id=self._signal_id
        )

    def should_close(self, *, date: pd.Timestamp = None, index: int = None) -> TradeDecision:
        """Close after the hold period."""
        if not self.position_open or not self.is_close_signal(date=date, index=index):
            return TradeDecision(ok=False, side=0)
        return TradeDecision(ok=True, side=int(SideInt.BUY), pos_effect=PositionEffect.CLOSE, signal_id="N/A")


def _make_dataset(ticker: str, seed: int, n: int = 500) -> PTDataset:
    """Build a seeded random-walk OHLCV book on business days from 2018-01-01."""
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    frame = pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1_000},
        index=dates,
    )
    return PTDataset(ticker, frame)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def _banner(step: str, title: str) -> None:
    """Print a spaced section header."""
    line = "=" * 72
    print(f"\n\n{line}")
    print(f"  STEP {step}: {title}")
    print(f"{line}\n")


def _kv(label: str, value: Any, *, indent: int = 2) -> None:
    """Print one labeled field."""
    pad = " " * indent
    print(f"{pad}{label:<28} {value}")


def _fmt_money(x: float) -> str:
    """Format dollars with sign-aware commas."""
    return f"${x:,.2f}"


def _fmt_pct(x: float) -> str:
    """Format a fraction as percent."""
    return f"{100.0 * x:.2f}%"


def _stat(stats: Mapping[str, Any], key: str, default: str = "n/a") -> str:
    """Pull one full_stats value as a short string."""
    value = stats.get(key, default) if hasattr(stats, "get") else default
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "n/a"
    if isinstance(value, (float, np.floating)):
        if "ratio" in key or key in {"sharpe_ratio", "sortino_ratio", "calmar_ratio"}:
            return f"{float(value):.4f}"
        if "drawdown" in key or "return" in key or key.endswith("_pct"):
            return _fmt_pct(float(value)) if abs(float(value)) <= 5 else f"{float(value):.4f}"
        return f"{float(value):,.4f}"
    return str(value)


def report_series_equity(name: str, equity: pd.Series) -> None:
    """Describe a standalone equity curve (StrategyBase.simulate)."""
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    ret = end / start - 1.0 if start else float("nan")
    print(f"  {name}")
    _kv("bars", len(equity))
    _kv("first date", equity.index[0].date())
    _kv("last date", equity.index[-1].date())
    _kv("start equity", f"{start:.6f}  (solo simulate normalizes to ~1.0)")
    _kv("end equity", f"{end:.6f}")
    _kv("total return", _fmt_pct(ret))
    print()


def report_sim_result(name: str, result: SimulationResult, *, extra: Optional[Dict[str, Any]] = None) -> None:
    """Pretty-print a shared-capital SimulationResult."""
    stats = result.full_stats if result.full_stats is not None else pd.Series(dtype=float)
    eq = result.equity
    cash = result.cash
    start_eq = float(eq.iloc[0])
    end_eq = float(eq.iloc[-1])
    total_ret = end_eq / start_eq - 1.0 if start_eq else float("nan")

    print(f"  {name}\n")
    _kv("calendar bars", len(eq))
    _kv("first date", eq.index[0].date())
    _kv("last date", eq.index[-1].date())
    _kv("starting equity", _fmt_money(start_eq))
    _kv("ending equity", _fmt_money(end_eq))
    _kv("total return", _fmt_pct(total_ret))
    _kv("ending cash", _fmt_money(float(cash.iloc[-1])))
    _kv("min cash", _fmt_money(float(cash.min())))
    _kv("closed trades", result.num_trades)
    open_at_end = int(result.n_open.iloc[-1]) if result.n_open is not None and len(result.n_open) else 0
    _kv("open positions at end", open_at_end)
    _kv("skipped opens", len(result.skipped))
    print()
    print("  Risk / return stats")
    _kv("sharpe_ratio", _stat(stats, "sharpe_ratio"))
    _kv("sortino_ratio", _stat(stats, "sortino_ratio"))
    _kv("max_drawdown", _stat(stats, "max_drawdown"))
    _kv("cagr", _stat(stats, "cagr"))
    _kv("volatility", _stat(stats, "volatility"))
    _kv("win_rate", _stat(stats, "win_rate"))
    if extra:
        print()
        print("  Notes")
        for k, v in extra.items():
            _kv(k, v)
    if result.skipped is not None and not result.skipped.empty:
        print()
        print("  Top skip reasons")
        counts = result.skipped["reason"].astype(str).value_counts().head(5)
        for reason, n in counts.items():
            _kv(f"x{n}", reason, indent=4)
    print()


def report_opt_result(name: str, result: OptimizationResult) -> None:
    """Pretty-print one OptimizationResult."""
    n_trials = len(result.trials)
    n_ok = int(result.trials["feasible"].sum()) if "feasible" in result.trials.columns else 0
    n_err = int(result.trials["error"].notna().sum()) if "error" in result.trials.columns else 0
    print(f"  {name}\n")
    _kv("trials run", n_trials)
    _kv("feasible", n_ok)
    _kv("errors", n_err)
    _kv("best_params", result.best_params)
    _kv("best_score", f"{result.best_score:.6f}" if result.best_params else "n/a")
    if result.best_result is not None:
        br = result.best_result
        _kv("best equity end", _fmt_money(float(br.equity.iloc[-1])))
        _kv("best num_trades", br.num_trades)
        _kv("best trades_all rows", len(br.trades_all))
        _kv("best sharpe (full)", _stat(br.full_stats, "sharpe_ratio"))
    print()


def report_opt_by_ticker(title: str, by_ticker: Mapping[str, OptimizationResult]) -> None:
    """Pretty-print per-ticker optimize results."""
    print(f"  {title}\n")
    _kv("tickers optimized", len(by_ticker))
    print()
    for ticker, res in by_ticker.items():
        score = f"{res.best_score:.6f}" if res.best_params else "n/a"
        end_eq = (
            _fmt_money(float(res.best_result.equity.iloc[-1]))
            if res.best_result is not None
            else "n/a"
        )
        print(f"    {ticker}")
        _kv("best_params", res.best_params, indent=6)
        _kv("best_score", score, indent=6)
        _kv("best equity end", end_eq, indent=6)
        _kv("feasible / trials", f"{int(res.trials['feasible'].sum())} / {len(res.trials)}", indent=6)
        print()


def main() -> None:
    """Run the end-to-end simulator stack walkthrough."""
    # ---------------------------------------------------------------------------
    # 0) Data — at least 7 synthetic names
    # ---------------------------------------------------------------------------
    _banner("0", "Build synthetic market data")

    TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "JPM", "XOM"]
    TRADE_START = "2019-01-01"

    print("  Request")
    _kv("tickers", ", ".join(TICKERS))
    _kv("bars each", 500)
    _kv("trade start", TRADE_START)
    _kv("source", "seeded random walk (no external data / strategy_research)")
    print()

    data = {t: _make_dataset(t, seed=i + 1) for i, t in enumerate(TICKERS)}
    assert len(data) >= 7, data.keys()

    print("  Built")
    _kv("count", len(data))
    for t, ds in data.items():
        _kv(t, f"{len(ds.data)} bars  |  {ds.data.index[0].date()} → {ds.data.index[-1].date()}")
    print()

    ## Shared-capital sizing used by Simulator / book runs
    SIM_KW = dict(
        capital=100_000.0,
        percent_of_capital=0.1,
        min_cash=0.0,
        buying_power_policy=BuyingPowerPolicy.SCALE_DOWN,
        commission_pct=0.0005,
        slippage_pct=0.0005,
        wipeout_floor=0.0,
        start_date=TRADE_START,
        finalize=True,
        track_exposure=True,
        stats_level="full",
    )

    print("  Shared-capital Simulator kwargs (steps 2, 4, 5a, 6, 7)")
    for k, v in SIM_KW.items():
        _kv(k, v)
    print()

    ## Baseline MR params (optimize will overlay tested keys)
    MR_PARAMS = {"ma_length": 50, "threshold": 0.02, "hold": 10}
    MR_SPACE = {"ma_length": [30, 50], "threshold": [0.015, 0.03]}
    BO_PARAMS = {"lookback": 20, "hold": 15}

    print("  LongMeanRevert baseline params")
    for k, v in MR_PARAMS.items():
        _kv(k, v)
    print()
    print("  Optimize space (4 mixes)")
    for k, v in MR_SPACE.items():
        _kv(k, v)
    print()

    # ---------------------------------------------------------------------------
    # 1) Single-strategy run — StrategyBase.simulate (solo capital)
    # ---------------------------------------------------------------------------
    _banner("1", "Single strategy — StrategyBase.simulate (solo sizing)")

    print("  What this is")
    print("    One ticker, one strategy. Capital is normalized (~1.0), fully invested,")
    print("    no cash floor, no wipeout. Returns legacy open/close event dicts.\n")

    solo = LongMeanRevert(
        data["AAPL"],
        start_trading_date=TRADE_START,
        ticker="AAPL",
        tplusn=0,
        **MR_PARAMS,
    )
    solo_trades, solo_equity = solo.simulate(finalize=True)

    n_opens = sum(1 for e in solo_trades if e.get("action") == "open")
    n_closes = sum(1 for e in solo_trades if e.get("action") == "close")
    print("  Run config")
    _kv("strategy", "LongMeanRevert")
    _kv("ticker", "AAPL")
    _kv("start_trading_date", TRADE_START)
    _kv("tplusn", 0)
    print()
    report_series_equity("Result", solo_equity)
    _kv("legacy events", len(solo_trades))
    _kv("opens / closes", f"{n_opens} / {n_closes}")
    if solo_trades:
        first = solo_trades[0]
        last = solo_trades[-1]
        _kv("first event", f"{first.get('action')} @ {first.get('date')}  px={first.get('price')}")
        _kv("last event", f"{last.get('action')} @ {last.get('date')}  px={last.get('price')}")
    print()

    # ---------------------------------------------------------------------------
    # 2) Flat shared-capital Simulator (list of strategies, no book hooks)
    # ---------------------------------------------------------------------------
    _banner("2", "Flat Simulator — shared cash, NO book hooks")

    print("  What this is")
    print("    All tickers in one Simulator([...]). Shared capital + buying power,")
    print("    but no max_open_trades / rank_opens / sleeve vetoes.\n")

    flat_strats = [
        LongMeanRevert(
            data[t],
            start_trading_date=TRADE_START,
            ticker=t,
            tplusn=0,
            **MR_PARAMS,
        )
        for t in TICKERS
    ]
    print("  Run config")
    _kv("n strategies", len(flat_strats))
    _kv("tickers", ", ".join(TICKERS))
    _kv("capital", _fmt_money(SIM_KW["capital"]))
    _kv("percent_of_capital", SIM_KW["percent_of_capital"])
    _kv("commission_pct", SIM_KW["commission_pct"])
    _kv("slippage_pct", SIM_KW["slippage_pct"])
    print()

    flat_result = Simulator(flat_strats, **SIM_KW).run()
    report_sim_result("Result — flat shared-capital Simulator", flat_result)

    # ---------------------------------------------------------------------------
    # 3) MultiAssetStrategy — per-ticker simulate_all + optimize (solo loop)
    # ---------------------------------------------------------------------------
    _banner("3a", "MultiAssetStrategy.simulate_all — independent solos")

    print("  What this is")
    print("    Container of per-ticker strategies. Each runs StrategyBase.simulate")
    print("    alone (NOT shared capital). Good for scanning signals per name.\n")

    multi = MultiAssetStrategy(
        name="mr",
        start_date=TRADE_START,
        params={t: dict(MR_PARAMS) for t in TICKERS},
        strategy_class=LongMeanRevert,
        data=data,
        tplusn=0,
    )
    multi_sim = multi.simulate_all(finalize=True)

    print("  Run config")
    _kv("name", multi.name)
    _kv("strategy_class", LongMeanRevert.__name__)
    _kv("tickers", ", ".join(TICKERS))
    _kv("start_date", multi.start_date)
    print()
    print("  Per-ticker ending equity (normalized solo curves)\n")
    for t, eq in multi_sim.equity.items():
        start = float(eq.iloc[0])
        end = float(eq.iloc[-1])
        n_ev = len(multi_sim.trades.get(t, []))
        _kv(t, f"end={end:.4f}  return={_fmt_pct(end / start - 1)}  events={n_ev}")
    print()

    _banner("3b", "MultiAssetStrategy.optimize — one ParamOptimizer per ticker")

    print("  What this is")
    print("    Always per-asset. Each ticker gets its own grid with solo sizing")
    print("    (SOLO_SIMULATOR_KWARGS: 100% equity, min_cash=None, no wipeout).\n")
    print("  Optimize config")
    _kv("space", MR_SPACE)
    _kv("combos per ticker", int(np.prod([len(v) for v in MR_SPACE.values()])))
    _kv("simulator_kwargs", "SOLO_SIMULATOR_KWARGS")
    _kv("n_jobs", 1)
    _kv("keep_top", 1)
    print()

    multi_opt = multi.optimize(
        MR_SPACE,
        simulator_kwargs=SOLO_SIMULATOR_KWARGS,
        n_jobs=1,
        keep_top=1,
        show_progress=True,
    )
    report_opt_by_ticker("Result — best mix per ticker", multi_opt)

    # ---------------------------------------------------------------------------
    # 4) One-sleeve SystematicMultiAsset — shared book simulate
    # ---------------------------------------------------------------------------
    _banner("4", "SystematicMultiAsset.simulate — one sleeve, book hooks ON")

    print("  What this is")
    print("    Shared cash pool through PortfolioBook hooks: max_open_trades,")
    print("    allow_multiple_per_asset, rank_opens / should_open vetoes.\n")

    mr_sleeve = Sleeve(
        "mr",
        LongMeanRevert,
        params=MR_PARAMS,
        tickers=TICKERS,
    )
    book = SystematicMultiAsset(
        [mr_sleeve],
        data,
        start_trading_date=TRADE_START,
        tplusn=0,
        max_open_trades=5,
        allow_multiple_per_asset=False,
    )
    print("  Book config")
    _kv("sleeves", [s.name for s in book.spec.sleeves])
    _kv("strategy", LongMeanRevert.__name__)
    _kv("children", f"{len(TICKERS)}  ({', '.join(TICKERS)})")
    _kv("max_open_trades", book.max_open_trades)
    _kv("allow_multiple_per_asset", book.allow_multiple_per_asset)
    _kv("start_trading_date", book.spec.start_trading_date)
    print()

    book_result = book.simulate(**SIM_KW)
    report_sim_result(
        "Result — one-sleeve shared book",
        book_result,
        extra={
            "vs flat Simulator": (
                "same strategy/params/tickers; book can skip opens when 5 already open"
            ),
        },
    )

    # ---------------------------------------------------------------------------
    # 5) Book optimize — shared capital (hooks on) and per_asset solo loop
    # ---------------------------------------------------------------------------
    _banner("5a", "SystematicMultiAsset.optimize(per_asset=False) — shared book grid")

    print("  What this is")
    print("    One grid for the whole book. Each trial rebuilds via with_trial_params")
    print("    so max_open_trades and other hooks still bind. Requires one sleeve.\n")
    print("  Optimize config")
    _kv("per_asset", False)
    _kv("space", MR_SPACE)
    _kv("n_jobs", 1)
    _kv("keep_top", 1)
    _kv("trial stats_level", "lean (default inside ParamOptimizer)")
    print()

    book_shared_opt = book.optimize(
        MR_SPACE,
        simulator_kwargs={k: v for k, v in SIM_KW.items() if k != "stats_level"},
        per_asset=False,
        n_jobs=1,
        keep_top=1,
        show_progress=True,
    )
    report_opt_result("Result — shared-capital book search", book_shared_opt)

    _banner("5b", "SystematicMultiAsset.optimize(per_asset=True) — solo loop")

    print("  What this is")
    print("    Same as MultiAsset.optimize for a one-sleeve book: one optimizer")
    print("    per ticker with solo sizing. Useful to compare vs step 3b.\n")

    book_per_asset = book.optimize(
        MR_SPACE,
        simulator_kwargs=SOLO_SIMULATOR_KWARGS,
        per_asset=True,
        n_jobs=1,
        keep_top=1,
        show_progress=True,
    )
    report_opt_by_ticker("Result — book per_asset (should align with MultiAsset)", book_per_asset)

    print("  Cross-check vs MultiAsset.optimize (step 3b)\n")
    for t in TICKERS:
        same = multi_opt[t].best_params == book_per_asset[t].best_params
        _kv(
            t,
            f"params_match={same}  multi={multi_opt[t].best_params}  book={book_per_asset[t].best_params}",
        )
    print()

    # ---------------------------------------------------------------------------
    # 6) Direct ParamOptimizer + book_factory (same path book.optimize uses)
    # ---------------------------------------------------------------------------
    _banner("6", "ParamOptimizer directly — book_factory tunes max_open_trades")

    print("  What this is")
    print("    Same engine book.optimize wraps. Here the space is a book bt_param")
    print("    (max_open_trades), applied through with_trial_params.\n")

    template = SystematicMultiAsset(
        [Sleeve("mr", LongMeanRevert, params=MR_PARAMS, tickers=TICKERS)],
        data,
        start_trading_date=TRADE_START,
        max_open_trades=3,
    )
    cap_space = {"max_open_trades": [2, 3, 5]}
    print("  Optimize config")
    _kv("space", cap_space)
    _kv("book_factory", "template.with_trial_params")
    _kv("starting max_open_trades", template.max_open_trades)
    print()

    direct = ParamOptimizer(
        space=cap_space,
        simulator_kwargs={k: v for k, v in SIM_KW.items() if k != "stats_level"},
        book_factory=template.with_trial_params,
        n_jobs=1,
        keep_top=1,
        show_progress=False,
    ).run()
    report_opt_result("Result — cap search over the shared book", direct)

    # ---------------------------------------------------------------------------
    # 7) Two-sleeve book — simulate OK; optimize must raise (one-sleeve rule)
    # ---------------------------------------------------------------------------
    _banner("7a", "Two-sleeve book.simulate — mean-revert + breakout (hooks ON)")

    print("  What this is")
    print("    Multi-sleeve shared capital is fine for simulate. Optimize is NOT")
    print("    allowed on multi-sleeve books (must optimize each sleeve alone).\n")

    dual = SystematicMultiAsset(
        [
            Sleeve("mr", LongMeanRevert, params=MR_PARAMS, tickers=TICKERS),
            Sleeve("mom", LongBreakout, params=BO_PARAMS, tickers=TICKERS),
        ],
        data,
        start_trading_date=TRADE_START,
        max_open_trades=8,
        allow_multiple_per_asset=True,
    )
    print("  Book config")
    _kv("sleeves", [s.name for s in dual.spec.sleeves])
    _kv("children", len(TICKERS) * 2)
    _kv("max_open_trades", dual.max_open_trades)
    _kv("allow_multiple_per_asset", dual.allow_multiple_per_asset)
    _kv("percent_of_capital (this run)", 0.05)
    print()

    dual_result = dual.simulate(**{**SIM_KW, "percent_of_capital": 0.05})
    report_sim_result("Result — dual-sleeve shared book", dual_result)

    _banner("7b", "Two-sleeve book.optimize — must raise ValueError")

    print("  Expectation")
    print("    SystematicMultiAsset.optimize requires exactly one sleeve.\n")

    try:
        dual.optimize(MR_SPACE, simulator_kwargs=SIM_KW, n_jobs=1, show_progress=False)
        raise AssertionError("expected multi-sleeve optimize to raise")
    except ValueError as exc:
        print("  Caught as expected\n")
        _kv("exception type", type(exc).__name__)
        _kv("message", str(exc))
        print()

    # ---------------------------------------------------------------------------
    # 8) Legacy PTBacktester — per-ticker backtesting.py (not shared cash)
    # ---------------------------------------------------------------------------
    _banner("8", "Legacy PTBacktester — one account per ticker (backtesting.py)")

    print("  What this is")
    print("    Older stack. Each ticker runs alone with its own cash through")
    print("    backtesting.py. This is NOT a shared-wallet portfolio like Simulator")
    print("    or SystematicMultiAsset. Prefer the modern path for new work.\n")

    ## PTBacktester wants a list of PTDataset, not a dict.
    pt_datalist = [data[t] for t in TICKERS]
    pt_cash = 100_000.0
    print("  Run config")
    _kv("strategy class", LongMeanRevert.__name__)
    _kv("tickers", ", ".join(TICKERS))
    _kv("cash per ticker", _fmt_money(pt_cash))
    _kv("start_overwrite", TRADE_START)
    _kv("commission", 0.0)
    _kv("shared capital?", "No — each name is independent")
    print()

    pt = PTBacktester(
        datalist=pt_datalist,
        strategy=LongMeanRevert,
        cash=pt_cash,
        strategy_settings={t: dict(MR_PARAMS) for t in TICKERS},
        start_overwrite=TRADE_START,
        commission=0.0,
        plot_indicators=False,
        verbose=False,
    )
    pt_stats = pt.run()
    pt_equity = pt._equity  # columns per ticker + Total

    print("  Result — per-ticker stats (selected rows)\n")
    stat_rows = [
        "Return [%]",
        "Sharpe Ratio",
        "# Trades",
        "Max. Drawdown [%]",
        "Win Rate [%]",
    ]
    for row in stat_rows:
        if row in pt_stats.index:
            vals = ", ".join(f"{t}={pt_stats.loc[row, t]}" for t in TICKERS[:4])
            _kv(row, f"{vals}, ...")
    print()

    print("  Combined equity (sum of independent accounts)\n")
    _kv("bars", len(pt_equity))
    _kv("first date", pt_equity.index[0].date())
    _kv("last date", pt_equity.index[-1].date())
    _kv("Total start", _fmt_money(float(pt_equity["Total"].iloc[0])))
    _kv("Total end", _fmt_money(float(pt_equity["Total"].iloc[-1])))
    total_ret = float(pt_equity["Total"].iloc[-1] / pt_equity["Total"].iloc[0] - 1.0)
    _kv("Total return", _fmt_pct(total_ret))
    print()
    print("  Ending equity by ticker\n")
    for t in TICKERS:
        start_v = float(pt_equity[t].iloc[0])
        end_v = float(pt_equity[t].iloc[-1])
        _kv(t, f"end={_fmt_money(end_v)}  return={_fmt_pct(end_v / start_v - 1.0)}")
    print()

    print("  Note")
    print("    PTBacktester.optimize / position_optimize still exist for old notebooks.")
    print("    For new parameter search, use ParamOptimizer or MultiAsset/book.optimize.")
    print()

    # ---------------------------------------------------------------------------
    # Done
    # ---------------------------------------------------------------------------
    _banner("done", "Demo finished")

    print("  Optional plots (uncomment in the script):")
    print("    flat_result.equity.plot(title='Flat Simulator')")
    print("    book_result.equity.plot(title='One-sleeve book')")
    print("    dual_result.equity.plot(title='MR + breakout book')")
    print("    pt_equity['Total'].plot(title='PTBacktester Total (independent accounts)')")
    print("    # pt.plot_portfolio()")
    print("    # pt.plot_position('AAPL')")
    print()


if __name__ == "__main__":
    main()
