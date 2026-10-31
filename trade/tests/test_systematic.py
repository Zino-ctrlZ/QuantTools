"""Tests for ``trade.backtester_.systematic`` and the ``Simulator(book)`` route.

Uses synthetic data and a toy mean-reversion child so every test is offline.

Core Classes:
    MeanRev: Toy long mean-reversion child strategy.

Core Functions:
    make_ds: Synthetic OHLCV ``PTDataset``.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd
import pytest

from EventDriven.types import PositionEffect, SignalID
from trade.backtester_._multi_asset_strategy import MultiAssetStrategy
from trade.backtester_._strategy import StrategyBase, TradeDecision
from trade.backtester_.data import PTDataset
from trade.backtester_.simulator import Simulator
from trade.backtester_.systematic import (
    BookConfigError,
    ChildBuildError,
    OpenCandidate,
    BookHookError,
    PortfolioBook,
    PortfolioContext,
    Sleeve,
    SystematicMultiAsset,
    Veto,
)

SIM = dict(capital=100_000, percent_of_capital=0.25, min_cash=None)


def make_ds(name: str, seed: int, n: int = 400) -> PTDataset:
    """Return a synthetic OHLCV dataset."""
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.015, n)))
    frame = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1e6}, index=idx)
    return PTDataset(name, frame)


class MeanRev(StrategyBase):
    """Toy long mean reversion: buy below the moving average, exit above it or after ``hold`` bars."""

    bt_params = {"ma_length": 20, "band": 0.02, "hold": 10}

    def __init__(
        self,
        data: PTDataset,
        start_trading_date: Optional[str] = None,
        ticker: Optional[str] = None,
        tplusn: int = 0,
        ma_length: int = 20,
        band: float = 0.02,
        hold: int = 10,
    ) -> None:
        """Store params, then run the base constructor (which calls ``setup``)."""
        self.ma_length, self.band, self.hold = ma_length, band, hold
        super().__init__(data=data, start_trading_date=start_trading_date, ticker=ticker, tplusn=tplusn)

    def setup(self) -> None:
        """Register the distance from the moving average."""
        close = self._df["close"]
        self.add_indicator("dist", close / close.rolling(self.ma_length).mean() - 1)

    def is_open_signal(self, *, date: Any = None, index: Optional[int] = None) -> bool:
        """Open when price is more than ``band`` below the average."""
        i, _ = self._resolve(date=date, index=index)
        return bool(self.indicators["dist"].values[i] < -self.band)

    def is_close_signal(self, *, date: Any = None, index: Optional[int] = None) -> bool:
        """Close when price is back above the average."""
        i, _ = self._resolve(date=date, index=index)
        return bool(self.indicators["dist"].values[i] > 0)

    def should_open(self, *, date: Any = None, index: Optional[int] = None) -> TradeDecision:
        """Open on signal when flat, trading is allowed, and child rules pass."""
        i, ts = self._resolve(date=date, index=index)
        if self.should_trade(index=i) and not self.position_open and self.is_open_signal(index=i) and self.rules.check(index=i):
            return TradeDecision(ok=True, side=1, pos_effect=PositionEffect.OPEN, signal_id=SignalID.generate(self.ticker, ts, "LONG"))
        return TradeDecision(ok=False, side=0)

    def should_close(self, *, date: Any = None, index: Optional[int] = None) -> TradeDecision:
        """Close on signal or after ``hold`` bars."""
        i, _ = self._resolve(date=date, index=index)
        if not self.position_open:
            return TradeDecision(ok=False, side=0)
        held = i - self._dates_map[self.position_info.entry_date]
        if self.is_close_signal(index=i) or held >= self.hold:
            return TradeDecision(ok=True, side=1, pos_effect=PositionEffect.CLOSE, signal_id=self.position_info.signal_id)
        return TradeDecision(ok=False, side=0)

    def open_action(self, *, date: Any = None, index: Optional[int] = None, signal_id: Any = None, entry_price: Any = None, side: Any = None) -> None:
        """Record the position."""
        _, ts = self._resolve(date=date, index=index)
        self.set_position_info(entry_date=ts, entry_price=entry_price, side=side, signal_id=signal_id)

    def close_action(self, *, date: Any = None, index: Optional[int] = None) -> None:
        """Clear the position."""
        self.remove_position_info()


@pytest.fixture(scope="module")
def data() -> dict:
    """Return three synthetic tickers."""
    return {t: make_ds(t, seed) for t, seed in (("AAA", 1), ("BBB", 2), ("CCC", 3))}


@pytest.fixture
def sleeves() -> List[Sleeve]:
    """Return two sleeves on every ticker with different params, so no twins."""
    return [
        Sleeve("mr", MeanRev, params={"band": 0.02}, asset_params={"CCC": {"band": 0.03}}),
        Sleeve("fast", MeanRev, params={"ma_length": 10, "hold": 5}),
    ]


def _max_concurrent(trades: pd.DataFrame) -> int:
    """Return the most positions open at any one close (closes fill before opens)."""
    dates = pd.DatetimeIndex(sorted(set(trades["entry_date"])))
    return max(int(((trades["entry_date"] <= d) & (trades["exit_date"] > d)).sum()) for d in dates)


# ---------------------------------------------------------------------- #
# Sleeve helpers
# ---------------------------------------------------------------------- #
def test_sleeve_from_multiasset(data: dict) -> None:
    """``Sleeve.from_multiasset`` copies class, per-ticker params, and universe."""
    params = {
        "AAA": {"ma_length": 10, "band": 0.02, "hold": 5},
        "BBB": {"ma_length": 20, "band": 0.03, "hold": 8},
    }
    multi = MultiAssetStrategy(
        name="mr_live",
        start_date="2020-03-02",
        params=params,
        strategy_class=MeanRev,
        data={t: data[t] for t in params},
        tplusn=0,
    )
    sleeve = Sleeve.from_multiasset(multi)
    assert sleeve.name == "mr_live"
    assert sleeve.strategy_class is MeanRev
    assert sleeve.params == {}
    assert sleeve.tickers == ["AAA", "BBB"]
    assert sleeve.asset_params["AAA"]["ma_length"] == 10
    assert sleeve.asset_params["BBB"]["hold"] == 8
    ## Renamed sleeve still carries the multi's params.
    renamed = Sleeve.from_multiasset(multi, name="mr")
    assert renamed.name == "mr"
    book = SystematicMultiAsset([renamed], multi.data, start_trading_date=multi.start_date)
    assert set(book.build()) == {"mr:AAA", "mr:BBB"}


# ---------------------------------------------------------------------- #
# Routing and parity
# ---------------------------------------------------------------------- #
def test_default_book_matches_plain_simulator(data: dict, sleeves: List[Sleeve]) -> None:
    """With default hooks the book path must equal today's Simulator on the same children."""
    book = SystematicMultiAsset(sleeves, data)
    children = book.build()
    plain = Simulator(list(children.values()), strategy_keys=list(children), **SIM).run()
    routed = book.simulate(**SIM)
    assert routed.num_trades == plain.num_trades > 0
    pd.testing.assert_series_equal(routed.eq, plain.eq)


def test_simulator_book_equals_book_simulate(data: dict, sleeves: List[Sleeve]) -> None:
    """``Simulator(book).run()`` and ``book.simulate()`` are the same route."""
    book = SystematicMultiAsset(sleeves, data, max_open_trades=2)
    a = Simulator(book, **SIM).run()
    b = book.simulate(**SIM)
    pd.testing.assert_series_equal(a.eq, b.eq)


def test_plain_list_has_no_book(data: dict) -> None:
    """A list of strategies keeps the original fill path."""
    sim = Simulator([MeanRev(data["AAA"], ticker="AAA")], **SIM)
    assert sim.book is None


def test_book_route_rejects_keys_and_nesting(data: dict, sleeves: List[Sleeve]) -> None:
    """The book owns its keys, and must not be hidden inside a list."""
    book = SystematicMultiAsset(sleeves, data)
    with pytest.raises(TypeError, match="sets strategy_keys from the book"):
        Simulator(book, strategy_keys=["x"], **SIM)
    with pytest.raises(TypeError, match="not inside a list"):
        Simulator([book], **SIM)


def test_strategy_is_not_a_book(data: dict) -> None:
    """Only PortfolioBook subclasses route as books."""
    assert not isinstance(MeanRev(data["AAA"], ticker="AAA"), PortfolioBook)


def test_portfolio_book_requires_build_and_defaults_the_rest(data: dict, sleeves: List[Sleeve]) -> None:
    """build is abstract; a build-only book uses the base hooks and trades like plain Simulator."""

    class NoBuild(PortfolioBook):
        """Implements nothing."""

    with pytest.raises(TypeError, match="abstract method.*build"):
        NoBuild()

    children_src = SystematicMultiAsset(sleeves, data)

    class BuildOnly(PortfolioBook):
        """Only supplies children; every hook is the base default."""

        def build(self) -> dict:
            """Return fresh children from the default book."""
            return children_src.build()

    children = children_src.build()
    plain = Simulator(list(children.values()), strategy_keys=list(children), **SIM).run()
    pd.testing.assert_series_equal(Simulator(BuildOnly(), **SIM).run().eq, plain.eq)
    assert isinstance(children_src, PortfolioBook)


def test_hook_exceptions_propagate_unchanged(data: dict, sleeves: List[Sleeve]) -> None:
    """The simulator calls hooks directly, so a hook's own error reaches the caller as-is."""

    class Raising(SystematicMultiAsset):
        """Book whose ranking always fails."""

        bt_params = {**SystematicMultiAsset.bt_params}

        def rank_opens(self, candidates: List[OpenCandidate], ctx: PortfolioContext) -> List[OpenCandidate]:
            """Fail on purpose."""
            raise KeyError("missing score")

    with pytest.raises(KeyError, match="missing score"):
        Raising(sleeves, data).simulate(**SIM)


def test_book_subclass_contract_still_enforced() -> None:
    """Inheriting the abstract base must not switch off the bt_params check."""
    with pytest.raises(TypeError, match="not listed in .*bt_params"):

        class Undeclared(SystematicMultiAsset):
            """Adds an __init__ arg without declaring it in bt_params."""

            bt_params = {**SystematicMultiAsset.bt_params}

            def __init__(self, *args: Any, secret: int = 1, **kwargs: Any) -> None:
                """Forward everything to the base book."""
                super().__init__(*args, **kwargs)


def test_repeat_runs_do_not_stack_rules(data: dict, sleeves: List[Sleeve]) -> None:
    """Each run builds fresh children, so set_rules applies once per run."""
    book = SystematicMultiAsset(sleeves, data)
    book.set_rules({"mr:*": lambda bar: bar.ind("dist") < -0.03})
    first, second = book.simulate(**SIM), book.simulate(**SIM)
    pd.testing.assert_series_equal(first.eq, second.eq)
    assert all(len(c.rules) == (1 if k.startswith("mr:") else 0) for k, c in book.build().items())


# ---------------------------------------------------------------------- #
# Default policies
# ---------------------------------------------------------------------- #
def test_max_open_trades_caps_concurrency(data: dict, sleeves: List[Sleeve]) -> None:
    """The cap holds at every close, and vetoes are logged by category."""
    result = SystematicMultiAsset(sleeves, data, max_open_trades=2).simulate(**SIM)
    assert _max_concurrent(result.trades_df) <= 2
    assert result.skipped["reason"].str.startswith("book: max_open_trades reached").any()


def test_one_sleeve_per_asset(data: dict, sleeves: List[Sleeve]) -> None:
    """With allow_multiple_per_asset=False no ticker is held by two sleeves at once."""
    result = SystematicMultiAsset(sleeves, data, allow_multiple_per_asset=False).simulate(**SIM)
    trades = result.trades_df
    for ticker, group in trades.groupby("ticker"):
        assert _max_concurrent(group) <= 1, ticker
    assert result.skipped["reason"].str.startswith("book: asset already held").any()


# ---------------------------------------------------------------------- #
# Params
# ---------------------------------------------------------------------- #
def test_param_layers(data: dict, sleeves: List[Sleeve]) -> None:
    """Class < sleeve params < asset params, and sleeves never share layers."""
    table = SystematicMultiAsset(sleeves, data).resolved_params
    assert table.loc[("mr", "BBB"), "band"] == 0.02
    assert table.loc[("mr", "CCC"), "band"] == 0.03
    assert table.loc[("mr", "AAA"), "hold"] == 10
    assert table.loc[("fast", "AAA"), "hold"] == 5
    assert table.loc[("fast", "CCC"), "band"] == 0.02


def test_overrides_land_in_asset_params(data: dict, sleeves: List[Sleeve]) -> None:
    """with_overrides merges into the sleeve, keeping existing asset params for that ticker."""
    tuned = SystematicMultiAsset(sleeves, data).with_overrides({("mr", "CCC"): {"hold": 3}})
    mr = next(s for s in tuned.spec.sleeves if s.name == "mr")
    assert dict(mr.asset_params["CCC"]) == {"band": 0.03, "hold": 3}
    assert next(s for s in sleeves if s.name == "mr").asset_params["CCC"] == {"band": 0.03}


def test_bad_overrides_are_reported(data: dict, sleeves: List[Sleeve]) -> None:
    """Unknown sleeves, tickers, and params in overrides fail with hints."""
    book = SystematicMultiAsset(sleeves, data)
    with pytest.raises(BookConfigError, match="no sleeve named 'mrr'.*did you mean 'mr'"):
        book.with_overrides({("mrr", "AAA"): {"hold": 3}})
    with pytest.raises(BookConfigError, match="'ZZZ' is not in this sleeve's tickers"):
        book.with_overrides({("mr", "ZZZ"): {"hold": 3}})
    with pytest.raises(BookConfigError, match="unknown param 'hld'.*did you mean 'hold'"):
        book.with_overrides({("mr", "AAA"): {"hld": 3}})


def test_with_overrides_and_with_params_return_new_books(data: dict, sleeves: List[Sleeve]) -> None:
    """Copies carry rules and book params; the original is untouched."""
    book = SystematicMultiAsset(sleeves, data, max_open_trades=2)
    book.set_rules({"AAA": lambda bar: True})
    tuned = book.with_overrides({("fast", "BBB"): {"hold": 7}})
    assert tuned.resolved_params.loc[("fast", "BBB"), "hold"] == 7
    assert book.resolved_params.loc[("fast", "BBB"), "hold"] == 5
    assert tuned.max_open_trades == 2 and len(tuned.spec.rules) == 1
    assert book.with_params(max_open_trades=1).max_open_trades == 1


# ---------------------------------------------------------------------- #
# Errors
# ---------------------------------------------------------------------- #
def test_config_errors_are_aggregated_with_hints(data: dict) -> None:
    """Every problem is listed at once, with did-you-mean hints."""
    bad = [
        Sleeve("mr", MeanRev, params={"bnd": 0.02}, tickers=["AAA", "AAAX"]),
        Sleeve("mr", MeanRev),
    ]
    with pytest.raises(BookConfigError) as exc:
        SystematicMultiAsset(bad, data, max_open_trades=0)
    message = str(exc.value)
    assert len(exc.value.issues) >= 4
    assert "did you mean 'band'?" in message
    assert "'AAAX' has no data" in message
    assert "name used more than once" in message
    assert "max_open_trades" in message


def test_twins_are_rejected(data: dict) -> None:
    """Same class, ticker, and params in two sleeves would double every trade."""
    with pytest.raises(BookConfigError, match="identical class and params"):
        SystematicMultiAsset([Sleeve("a", MeanRev), Sleeve("b", MeanRev)], data)


def test_unknown_selector_is_rejected(data: dict, sleeves: List[Sleeve]) -> None:
    """A rule selector that matches nothing fails fast with a hint."""
    book = SystematicMultiAsset(sleeves, data)
    with pytest.raises(BookConfigError, match="matches no child"):
        book.set_rules({"mrr:*": lambda bar: True})


def test_child_build_error_names_child_and_params(data: dict) -> None:
    """A child that raises in setup is named with its resolved params."""

    class Broken(MeanRev):
        """Child whose setup always fails."""

        bt_params = {"ma_length": 20, "band": 0.02, "hold": 10}

        def setup(self) -> None:
            """Fail on purpose."""
            raise RuntimeError("boom")

    book = SystematicMultiAsset([Sleeve("br", Broken, tickers=["AAA"])], data)
    with pytest.raises(ChildBuildError, match="Building Broken for 'br:AAA' failed: RuntimeError: boom"):
        book.simulate(**SIM)


def test_bad_hook_return_names_hook_date_and_key(data: dict, sleeves: List[Sleeve]) -> None:
    """A hook returning the wrong type fails with where and when."""

    class Sloppy(SystematicMultiAsset):
        """Book whose should_open returns a bool."""

        bt_params = {**SystematicMultiAsset.bt_params}

        def should_open(self, candidate: OpenCandidate, ctx: PortfolioContext) -> Any:
            """Return the wrong type."""
            return True

    with pytest.raises(BookHookError, match=r"Sloppy\.should_open for '.+' on \d{4}-\d{2}-\d{2}: returned bool"):
        Sloppy(sleeves, data).simulate(**SIM)


# ---------------------------------------------------------------------- #
# Inheritance
# ---------------------------------------------------------------------- #
def test_subclass_adds_param_and_hook(data: dict, sleeves: List[Sleeve]) -> None:
    """A subclass adds a tunable param via **kwargs and overrides a hook."""

    class SleeveCap(SystematicMultiAsset):
        """Book that caps open positions per sleeve."""

        bt_params = {**SystematicMultiAsset.bt_params, "per_sleeve": 1}

        def __init__(self, *args: Any, per_sleeve: int = 1, **kwargs: Any) -> None:
            """Store the per-sleeve cap, then build the base book."""
            self.per_sleeve = per_sleeve
            super().__init__(*args, **kwargs)

        def should_open(self, candidate: OpenCandidate, ctx: PortfolioContext) -> Any:
            """Veto when the candidate's sleeve already holds ``per_sleeve`` positions."""
            sleeve = self.sleeve_of(candidate.key)
            held = sum(self.sleeve_of(k) == sleeve for k in ctx.open_keys)
            if held >= self.per_sleeve:
                return Veto(f"sleeve cap ({sleeve} holds {held})")
            return super().should_open(candidate, ctx)

    book = SleeveCap(sleeves, data, per_sleeve=1)
    result = book.simulate(**SIM)
    for sleeve, group in result.trades_df.groupby(result.trades_df["strategy_key"].str.split(":").str[0]):
        assert _max_concurrent(group) <= 1, sleeve
    assert book.with_params(per_sleeve=2).per_sleeve == 2
