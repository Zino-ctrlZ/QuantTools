"""Compute equity, trade, and cash statistics for portfolio simulations.

These builders summarize an equity curve, a round-trip trades table, and the
daily cash and exposure series produced by ``Simulator``. ``eq_stats`` and
``trade_stats`` must stay numerically identical to
``strategy_research.resources.helpers`` so simulator results compare directly
with ``run_strategy_leg`` output. The statistics ``AggregatorParent.aggregate``
reports that those two lack live in ``extended_stats`` instead, so adding them
never breaks that parity.

Core Functions:
    eq_stats: Return, risk, and drawdown statistics for one equity curve.
    trade_stats: Win rate, payoff, MAE/MFE, and duration statistics for round trips.
    compute_cash_stats: Idle capital, deployment, and funding-pressure statistics.
    compute_exposure_stats: Portfolio and per-strategy exposure summaries.
    drawdown_stats: Drawdown depth in dollars, average depth, and time below peak.
    yearly_returns: Calendar-year returns from an equity curve.
    extended_stats: Every ``AggregatorParent`` statistic missing from the two above.
    combine_stats: One flat Series of equity, trade, and extended statistics.

Usage:
    >>> from trade.backtester_.simulator.stats import eq_stats
    >>> eq_stats(result.equity)["sharpe_ratio"]
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def eq_stats(eq: pd.Series) -> pd.Series:
    """Summarize an equity curve with return, risk, and drawdown statistics.

    Args:
        eq: Equity values indexed by date. Any positive starting level works.

    Returns:
        Series with total return, CAGR, volatility, Sharpe, Sortino, Calmar,
        max drawdown, skew, kurtosis, daily win rate, average win and loss,
        payoff ratio, and tail ratio. Annualization assumes 252 bars per year.
    """
    returns = eq.pct_change().dropna()

    total_return = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (252 / len(eq)) - 1

    volatility = returns.std() * np.sqrt(252)

    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else np.nan

    downside_std = returns[returns < 0].std() * np.sqrt(252)

    sortino = (returns.mean() * 252) / downside_std if downside_std > 0 else np.nan

    running_max = eq.cummax()
    drawdowns = (running_max - eq) / running_max
    max_drawdown = drawdowns.max()

    calmar = cagr / max_drawdown if max_drawdown > 0 else np.nan

    skew = returns.skew()
    kurtosis = returns.kurtosis()

    win_rate = (returns > 0).mean()

    avg_win = returns[returns > 0].mean()
    avg_loss = returns[returns < 0].mean()

    payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 and not np.isnan(avg_loss) else np.nan

    tail_ratio = returns.quantile(0.95) / abs(returns.quantile(0.05)) if returns.quantile(0.05) != 0 else np.nan

    return pd.Series(
        {
            "total_return": total_return,
            "cagr": cagr,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": max_drawdown,
            "skew": skew,
            "kurtosis": kurtosis,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "payoff_ratio": payoff_ratio,
            "tail_ratio": tail_ratio,
        }
    )


def trade_stats(trades: pd.DataFrame) -> pd.Series:
    """Summarize round-trip trades.

    Args:
        trades: One row per round trip. Uses ``return_pct``, and when present
            ``mae``, ``mfe``, ``entry_date``, and ``exit_date``. Holding-day stats
            are only filled when the index is a ``DatetimeIndex`` of entry dates.

    Returns:
        Series of trade statistics, or an empty Series when ``trades`` is empty.
    """
    if trades.empty:
        return pd.Series(dtype=float)

    r = trades["return_pct"].dropna()

    wins = r[r > 0]
    losses = r[r < 0]

    avg_win = wins.mean()
    avg_loss = losses.mean()

    gross_profit = wins.sum()
    gross_loss = losses.sum()

    profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else np.nan

    expectancy = r.mean()

    payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 and not np.isnan(avg_loss) else np.nan

    win_rate = (r > 0).mean()
    loss_rate = (r < 0).mean()

    best_trade = r.max()
    worst_trade = r.min()
    duration = (
        trades["exit_date"] - trades["entry_date"]
        if "exit_date" in trades.columns and "entry_date" in trades.columns
        else None
    )

    holding_days = None
    if "exit_date" in trades.columns:
        entry_dates = trades.index
        exit_dates = pd.to_datetime(trades["exit_date"], errors="coerce")

        if isinstance(trades.index, pd.DatetimeIndex):
            holding_days = (exit_dates - entry_dates).dt.days

    return pd.Series(
        {
            "num_trades": len(r),
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "avg_return": expectancy,
            "median_return": r.median(),
            "avg_mae": trades["mae"].mean() if "mae" in trades.columns else np.nan,
            "avg_mfe": trades["mfe"].mean() if "mfe" in trades.columns else np.nan,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_duration": duration.mean().days if duration is not None else np.nan,
            "median_duration": duration.median().days if duration is not None else np.nan,
            "max_duration": duration.max().days if duration is not None else np.nan,
            "min_duration": duration.min().days if duration is not None else np.nan,
            "payoff_ratio": payoff_ratio,
            "profit_factor": profit_factor,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "return_std": r.std(),
            "return_skew": r.skew(),
            "return_kurtosis": r.kurtosis(),
            "p95_trade": r.quantile(0.95),
            "p05_trade": r.quantile(0.05),
            "tail_ratio": (r.quantile(0.95) / abs(r.quantile(0.05)) if r.quantile(0.05) != 0 else np.nan),
            "avg_holding_days": holding_days.mean() if holding_days is not None else np.nan,
            "median_holding_days": holding_days.median() if holding_days is not None else np.nan,
            "max_holding_days": holding_days.max() if holding_days is not None else np.nan,
        }
    )


def compute_cash_stats(
    cash: pd.Series,
    equity: pd.Series,
    gross_exposure: pd.Series,
    low_cash_threshold: float = 0.10,
) -> pd.Series:
    """Summarize idle capital, deployment, and funding pressure from daily cash and equity.

    Args:
        cash: Daily cash in dollars (negative means borrowed).
        equity: Daily portfolio equity in dollars.
        gross_exposure: Daily sum of absolute position values.
        low_cash_threshold: Cash-to-equity level that counts as a low-cash day for the streak.

    Returns:
        Series of cash statistics. Percent stats are fractions (0.25 means 25%).
        Empty when equity is never positive.

    Examples:
        >>> cash = pd.Series([100.0, 50.0, 5.0])
        >>> equity = pd.Series([100.0, 100.0, 100.0])
        >>> gross = pd.Series([0.0, 50.0, 95.0])
        >>> round(float(compute_cash_stats(cash, equity, gross)["avg_cash_pct"]), 4)
        0.5167
    """
    ## cash / equity is meaningless once equity is zero or negative (wipeout days).
    valid = equity > 0
    cash_pct = (cash[valid] / equity[valid]).astype(float)
    if cash_pct.empty:
        return pd.Series(dtype=float)
    utilization = 1.0 - cash_pct

    ## Run-length of consecutive low-cash days: a new group starts each time the flag flips.
    low = cash_pct < low_cash_threshold
    streak_ids = (low != low.shift()).cumsum()
    longest_low_streak = int(low.groupby(streak_ids).sum().max()) if low.any() else 0

    return pd.Series(
        {
            "avg_cash_pct": cash_pct.mean(),
            "median_cash_pct": cash_pct.median(),
            "min_cash_pct": cash_pct.min(),
            "p5_cash_pct": cash_pct.quantile(0.05),
            "avg_capital_utilization": utilization.mean(),
            "max_capital_utilization": utilization.max(),
            "pct_days_cash_below_10pct": (cash_pct < 0.10).mean(),
            "pct_days_cash_below_25pct": (cash_pct < 0.25).mean(),
            ## Judged by open positions, not cash_pct >= 1: shorts push cash above equity
            ## while still deployed, so cash alone would miscount them.
            "pct_days_all_cash": (gross_exposure[valid] == 0).mean(),
            f"longest_streak_cash_below_{low_cash_threshold:.0%}_days": longest_low_streak,
            "min_cash": cash.min(),
        }
    )


def _strategy_group_name(column: str) -> str:
    """Return the strategy/sleeve name for a child exposure column.

    Book children are labeled ``sleeve:ticker`` (for example ``mom:AAPL``);
    the summary rolls those up to ``mom``. A plain ticker or class label with
    no ``:`` is left as-is.
    """
    name = str(column)
    return name.split(":", 1)[0] if ":" in name else name


def _roll_exposure_by_strategy(strategy_exposure: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sum child columns into one signed and one absolute series per strategy.

    Absolute size is the sum of each child's absolute size, so two longs in the
    same sleeve both count toward how loaded that sleeve is.

    Returns:
        Signed totals and absolute totals, columns ordered by first appearance.
    """
    groups: Dict[str, List[str]] = {}
    for col in strategy_exposure.columns:
        groups.setdefault(_strategy_group_name(col), []).append(str(col))
    signed: Dict[str, pd.Series] = {}
    absolute: Dict[str, pd.Series] = {}
    for group, cols in groups.items():
        part = strategy_exposure[cols]
        signed[group] = part.sum(axis=1)
        absolute[group] = part.abs().sum(axis=1)
    return pd.DataFrame(signed), pd.DataFrame(absolute)


def compute_exposure_stats(
    equity: pd.Series,
    gross_exposure: pd.Series,
    net_exposure: pd.Series,
    long_exposure: pd.Series,
    short_exposure: pd.Series,
    leverage: pd.Series,
    n_open: pd.Series,
    strategy_exposure: pd.DataFrame,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Turn daily exposure curves into short portfolio and per-strategy tables.

    Portfolio numbers answer: how leveraged was the book, how often was it flat,
    and what was the peak dollars invested. Strategy numbers answer: how large
    each sleeve (or strategy type) ran across all its tickers, what share of the
    book it took, and how often it was on.

    Child columns such as ``mom:AAPL`` and ``mom:NVDA`` are summed into one
    ``mom`` column before summarizing. Daily ``strategy_exposure`` on the result
    still keeps the per-child detail.

    Leverage means total position size divided by equity that day. Percent
    fields are fractions (0.25 means 25%). The strategy table has metrics as
    rows and strategies as columns, same idea as trade stats by ticker.

    Args:
        equity: Portfolio worth each day.
        gross_exposure: Total dollars in positions each day.
        net_exposure: Long dollars minus short dollars each day.
        long_exposure: Dollars in longs only.
        short_exposure: Dollars in shorts only.
        leverage: Total size divided by equity (missing when equity is not positive).
        n_open: How many positions were open each day.
        strategy_exposure: How much each child held each day (may be ``sleeve:ticker``).

    Returns:
        A portfolio summary Series, and a per-strategy summary DataFrame
        (columns are sleeve / strategy names, not ticker pairs).
        Both are empty when there is no equity series.

    Examples:
        >>> eq = pd.Series([100.0, 100.0, 100.0])
        >>> gross = pd.Series([0.0, 50.0, 100.0])
        >>> net = gross.copy()
        >>> long = gross.copy()
        >>> short = pd.Series([0.0, 0.0, 0.0])
        >>> lev = gross / eq
        >>> n = pd.Series([0, 1, 1])
        >>> strat = pd.DataFrame({"mom:AAPL": [0.0, 50.0, 40.0], "mom:NVDA": [0.0, 0.0, 60.0]})
        >>> port, by_s = compute_exposure_stats(eq, gross, net, long, short, lev, n, strat)
        >>> float(port["max_gross_exposure"])
        100.0
        >>> list(by_s.columns)
        ['mom']
        >>> float(by_s.loc["max_abs_exposure", "mom"])
        100.0
    """
    if equity.empty:
        return pd.Series(dtype=float), pd.DataFrame()

    valid = equity > 0
    ## Leverage is already NaN on non-positive equity days; drop those for mean/max.
    lev = leverage[valid].dropna()
    net_pct = (net_exposure[valid] / equity[valid]).astype(float)
    long_pct = (long_exposure[valid] / equity[valid]).astype(float)
    short_pct = (short_exposure[valid] / equity[valid]).astype(float)

    portfolio = pd.Series(
        {
            "avg_leverage": float(lev.mean()) if not lev.empty else np.nan,
            "median_leverage": float(lev.median()) if not lev.empty else np.nan,
            "max_leverage": float(lev.max()) if not lev.empty else np.nan,
            "max_gross_exposure": float(gross_exposure.max()),
            "avg_net_pct": float(net_pct.mean()) if not net_pct.empty else np.nan,
            "avg_long_pct": float(long_pct.mean()) if not long_pct.empty else np.nan,
            "avg_short_pct": float(short_pct.mean()) if not short_pct.empty else np.nan,
            "avg_n_open": float(n_open.mean()),
            "max_n_open": float(n_open.max()),
            "pct_days_flat": float((gross_exposure == 0).mean()),
            "pct_days_leverage_above_1": float((lev > 1).mean()) if not lev.empty else np.nan,
            "pct_days_net_long": float((net_exposure > 0).mean()),
            "pct_days_net_short": float((net_exposure < 0).mean()),
        }
    )

    if strategy_exposure.empty:
        return portfolio, pd.DataFrame()

    ## Roll mom:AAPL + mom:NVDA → mom before summarizing; daily series stays detailed.
    signed, absolute = _roll_exposure_by_strategy(strategy_exposure)
    ## Share of book gross only on days the book is invested; flat days would be 0/0.
    gross_pos = gross_exposure.where(gross_exposure > 0)
    rows: Dict[str, pd.Series] = {}
    for name in signed.columns:
        net_series = signed[name]
        abs_exp = absolute[name]
        share = (abs_exp / gross_pos).dropna()
        rows[str(name)] = pd.Series(
            {
                "avg_exposure": float(net_series.mean()),
                "avg_abs_exposure": float(abs_exp.mean()),
                "max_abs_exposure": float(abs_exp.max()),
                "avg_share_of_gross": float(share.mean()) if not share.empty else np.nan,
                "max_share_of_gross": float(share.max()) if not share.empty else np.nan,
                "pct_days_held": float((abs_exp != 0).mean()),
                "pct_days_long": float((net_series > 0).mean()),
                "pct_days_short": float((net_series < 0).mean()),
            }
        )
    ## Metrics as rows, strategy/sleeve names as columns.
    by_strategy = pd.DataFrame(rows)
    return portfolio, by_strategy


def drawdown_stats(eq: pd.Series) -> pd.Series:
    """Summarize drawdown depth in dollars, average depth, and time spent below the prior peak.

    Args:
        eq: Equity values indexed by date.

    Returns:
        Series with ``max_drawdown_value`` (largest peak-to-trough drop in equity
        units, positive), ``avg_drawdown`` (mean daily drawdown as a positive
        fraction, days at a peak count as zero), and ``max_drawdown_duration_days``
        / ``avg_drawdown_duration_days`` (calendar days since the last peak).

    Examples:
        >>> eq = pd.Series([1.0, 1.2, 0.9, 1.2, 1.3], index=pd.date_range("2024-01-01", periods=5))
        >>> float(drawdown_stats(eq)["max_drawdown_duration_days"])
        1.0
    """
    running_max = eq.cummax()
    drawdown = (running_max - eq) / running_max

    ## A bar at or above the running max resets the clock; every later bar measures
    ## days since that reset. Flat equity at a peak (e.g. all-cash days) counts as zero,
    ## unlike AggregatorParent.mdd_duration, which keeps counting from the first time
    ## that value appeared.
    dates = pd.Series(eq.index, index=eq.index)
    last_peak = dates.where(eq >= running_max).ffill()
    duration_days = (dates - last_peak).dt.days.astype(float)

    return pd.Series(
        {
            "max_drawdown_value": float((running_max - eq).max()),
            "avg_drawdown": float(drawdown.mean()),
            "max_drawdown_duration_days": float(duration_days.max()),
            "avg_drawdown_duration_days": float(duration_days.mean()),
        }
    )


def yearly_returns(eq: pd.Series) -> pd.Series:
    """Return each calendar year's return, from that year's first to last equity value.

    Matches ``AggregatorParent.yearly_retrns``: a partial first or last year is
    measured over the days present, and the year boundary is not chained (the
    move from the last bar of one year to the first bar of the next is dropped).

    Args:
        eq: Equity values indexed by date.

    Returns:
        Series indexed by year with fractional returns.
    """
    by_year = eq.groupby(eq.index.year)
    return by_year.last() / by_year.first() - 1


def _longest_streak(flags: pd.Series) -> int:
    """Return the longest run of consecutive True values."""
    if not flags.any():
        return 0
    run_ids = (flags != flags.shift()).cumsum()
    return int(flags.groupby(run_ids).sum().max())


def _exposure_fraction(index: pd.DatetimeIndex, trades: pd.DataFrame) -> float:
    """Return the fraction of bars that fall inside at least one trade's entry-to-exit window."""
    ## Difference-array sweep: +1 at each entry bar, -1 after each exit bar, then a
    ## running sum > 0 marks bars with any open trade. Inclusive of both ends, like
    ## AggregatorParent.ExposureDays.
    marks = np.zeros(len(index) + 1, dtype=int)
    starts = index.searchsorted(pd.DatetimeIndex(trades["entry_date"]), side="left")
    ends = index.searchsorted(pd.DatetimeIndex(trades["exit_date"]), side="right")
    np.add.at(marks, starts, 1)
    np.add.at(marks, ends, -1)
    return float((np.cumsum(marks[:-1]) > 0).mean())


def extended_stats(
    eq: pd.Series,
    trades: pd.DataFrame,
    *,
    risk_free_rate: float = 0.0,
    buy_hold_return: float = np.nan,
) -> pd.Series:
    """Return the ``AggregatorParent.aggregate`` statistics that ``eq_stats`` and ``trade_stats`` lack.

    Units follow ``eq_stats``: fractions, not percents, and drawdowns are positive.
    Losing trades are ``return_pct <= 0``, the same split ``AggregatorParent`` uses.

    Args:
        eq: Portfolio equity indexed by date.
        trades: Round trips with ``entry_date``, ``exit_date``, ``return_pct``,
            and optionally ``pnl`` (dollar profit and loss).
        risk_free_rate: Annual risk-free rate for ``sharpe_ratio_rf`` / ``sortino_ratio_rf``.
        buy_hold_return: Benchmark return computed by the caller, passed through.

    Returns:
        Series of extended statistics, with one ``return_{year}`` entry per calendar year.
    """
    returns = eq.pct_change().dropna()
    log_returns = np.log(eq / eq.shift(1)).dropna()

    ## Risk-free-adjusted ratios use the same 252-bar convention as eq_stats, so with
    ## risk_free_rate=0 they equal eq_stats' sharpe_ratio and sortino_ratio.
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess = returns - daily_rf
    downside_std = excess[excess < 0].std()
    sharpe_rf = excess.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else np.nan
    sortino_rf = excess.mean() * np.sqrt(252) / downside_std if downside_std > 0 else np.nan

    stats = {
        "duration_days": (eq.index[-1] - eq.index[0]).days,
        "equity_final": float(eq.iloc[-1]),
        "equity_peak": float(eq.max()),
        "buy_hold_return": buy_hold_return,
        "median_daily_return": returns.median(),
        ## Named after AggregatorParent's "VaR 95%" (5th percentile) and "VaR 05%" (95th).
        "daily_var_95": returns.quantile(0.05),
        "daily_return_p95": returns.quantile(0.95),
        "log_return_skew": log_returns.skew(),
        "risk_free_rate": risk_free_rate,
        "sharpe_ratio_rf": sharpe_rf,
        "sortino_ratio_rf": sortino_rf,
        **drawdown_stats(eq).to_dict(),
    }

    if trades.empty:
        stats["exposure_time"] = 0.0
    else:
        ## Chronological order matters for streaks; trades_df is grouped by strategy.
        t = trades.sort_values(["entry_date", "exit_date"]).reset_index(drop=True)
        r = t["return_pct"].astype(float)
        wins, losses = t[r > 0], t[r <= 0]
        hold_days = (t["exit_date"] - t["entry_date"]).dt.days
        stats.update(
            {
                "exposure_time": _exposure_fraction(eq.index, t),
                "p25_trade": r.quantile(0.25),
                "p75_trade": r.quantile(0.75),
                "avg_win_duration": hold_days[r > 0].mean(),
                "avg_loss_duration": hold_days[r <= 0].mean(),
                "max_win_duration": hold_days[r > 0].max(),
                "max_loss_duration": hold_days[r <= 0].max(),
                ## Population std (ddof=0) to match AggregatorParent.SQN's np.std.
                "sqn": r.mean() * np.sqrt(len(r)) / r.std(ddof=0) if r.std(ddof=0) > 0 else np.nan,
                "winning_streak": _longest_streak(r > 0),
                "losing_streak": _longest_streak(r <= 0),
                ## Dollar-weighted, unlike trade_stats' profit_factor on return_pct. Under
                ## portfolio sizing trades differ in size, so this is the one that ties to equity.
                "profit_factor_dollar": (
                    wins["pnl"].sum() / abs(losses["pnl"].sum())
                    if "pnl" in t.columns and losses["pnl"].sum() != 0
                    else np.nan
                ),
            }
        )

    yearly = yearly_returns(eq)
    stats.update({f"return_{year}": ret for year, ret in yearly.items()})
    return pd.Series(stats)


def combine_stats(equity: pd.Series, trades: pd.Series, extended: pd.Series) -> pd.Series:
    """Merge equity, trade, and extended statistics into one flat Series.

    ``eq_stats`` and ``trade_stats`` share five names with different meanings
    (``win_rate``, ``avg_win``, ``avg_loss``, ``payoff_ratio``, ``tail_ratio``:
    daily in one, per-trade in the other). The trade versions get a ``trade_``
    prefix so both survive; equity keys keep their names.

    Args:
        equity: Output of ``eq_stats``.
        trades: Output of ``trade_stats`` (may be empty).
        extended: Output of ``extended_stats``.

    Returns:
        Equity stats, then trade stats, then extended stats.

    Raises:
        ValueError: If extended stats collide with an equity or trade key.
    """
    trades = trades.rename(lambda k: f"trade_{k}" if k in equity.index else k)
    combined = pd.concat([equity, trades, extended])
    duplicated = combined.index[combined.index.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"Duplicate stat names when combining: {duplicated}")
    return combined
