# Backtester guide (plain English)

This folder is how we **test trading ideas on past prices**.

You give it:

1. **A strategy** — rules for when to buy and sell  
2. **Price history** for one or more tickers  
3. **Money rules** — how much cash to start with, and how big each trade can be  

It gives back:

- An **equity curve** (how the account grew day by day)  
- A **trade list** (every round trip)  
- **Stats** (return, drawdown, Sharpe, and so on)  

You do **not** need to understand the internal code to use this. Think of each tool as a different way to run the same idea.

---

## Which tool should I use?

| I want to… | Use this |
|---|---|
| Test **one ticker** by itself | `StrategyBase.simulate()` |
| Test **many tickers**, each with its **own** money (no sharing) | `MultiAssetStrategy` |
| Test **many tickers** that share **one wallet** | `Simulator` or `SystematicMultiAsset` |
| Add portfolio rules (max open trades, ranking, vetoes) | `SystematicMultiAsset` (a “book”) |
| Search for better strategy settings | `ParamOptimizer` / `.optimize(...)` |
| Run the **old** per-ticker backtesting.py stack | `PTBacktester` (**legacy**) |

**Rule of thumb**

- **Solo / MultiAsset** = each name is graded alone (like separate paper accounts).  
- **Simulator / Book** = one real portfolio wallet; trades compete for cash and slots.  
- **Optimizer** = try many setting mixes and keep the winners.  
- **PTBacktester** = older path; still works, but new work should use Simulator / Book.

There is a runnable walkthrough (modern stack **and** legacy `PTBacktester`) in:

`trade/backtester_/examples/demo_simulator_stack.py`

---

## Words we use a lot

**Strategy**  
A Python class with buy/sell rules (for example mean reversion or momentum).

**Dataset (`PTDataset`)**  
Price history for one ticker (open / high / low / close / volume).

**Equity**  
What the account is worth each day (cash + open positions).

**Shared capital**  
One pile of cash for every ticker/strategy in the run.

**Solo sizing**  
One ticker uses (almost) all of its own account each time it opens — good for “how good is this name alone?”

**Book / sleeve**  
A **book** is a portfolio of strategies. A **sleeve** is one strategy style inside that book (for example “mean reversion” or “momentum”), usually over many tickers.

**Commission / slippage**  
Optional cash costs as a **percent of trade size**. They do **not** change the fill price used for returns / MAE / MFE.

**Prior flat day**  
The modern simulator always records one business day **before** trading starts, with cash = starting capital and nothing open. So the first equity point is always your starting money, even if a trade opens on day one.

---

## 1. One ticker alone

Simplest path. The strategy runs through the shared simulator with solo sizing (fully invested, no cash floor, no wipeout stop).

```python
from strategy_research.resources.strategies.base import LongZScoreMeanReversion

# `data` is a PTDataset for one ticker
strat = LongZScoreMeanReversion(
    data,
    start_trading_date="2019-01-01",
    ticker="AAPL",
    ma_length=50,
    entry_zscore_threshold=-1.5,
)

trades, equity = strat.simulate(finalize=True)
equity.plot(title="AAPL solo")
```

**What you get**

- `trades` — legacy open/close event list (easy to skim)  
- `equity` — normalized curve (starts near `1.0`)  

**When to use**  
Quick check: “Does this idea work on AAPL at all?”

---

## 2. Many tickers, each alone (`MultiAssetStrategy`)

Same strategy class on many names. Each ticker is simulated **independently**. They do **not** share cash.

```python
from trade.backtester_._multi_asset_strategy import MultiAssetStrategy
from strategy_research.resources.strategies.base import LongZScoreMeanReversion

params = {
    "AAPL": {"ma_length": 50, "entry_zscore_threshold": -1.5},
    "MSFT": {"ma_length": 40, "entry_zscore_threshold": -2.0},
    # ... more tickers ...
}

multi = MultiAssetStrategy(
    name="mr",
    start_date="2019-01-01",
    params=params,
    strategy_class=LongZScoreMeanReversion,
    data=data_by_ticker,   # {"AAPL": PTDataset(...), "MSFT": ...}
    tplusn=0,
)

results = multi.simulate_all(finalize=True)
# results.equity["AAPL"], results.trades["AAPL"], ...
```

**Optimize each ticker separately**

```python
from trade.backtester_.optimizer import SOLO_SIMULATOR_KWARGS

by_ticker = multi.optimize(
    space={
        "ma_length": [40, 50],
        "entry_zscore_threshold": [-1.5, -2.0],
    },
    simulator_kwargs=SOLO_SIMULATOR_KWARGS,  # or omit for the same defaults
    n_jobs=1,
    keep_top=1,
)

for ticker, res in by_ticker.items():
    print(ticker, res.best_params, res.best_score)
```

**When to use**  
“What settings does each name like on its own?” before you put them in one portfolio.

---

## 3. Shared wallet — plain `Simulator`

Many strategies (or many tickers) **share one cash pile**. No special portfolio rules beyond cash, size, leverage, and fees.

```python
from trade.backtester_.simulator import Simulator, BuyingPowerPolicy
from strategy_research.resources.strategies.base import LongZScoreMeanReversion

strategies = [
    LongZScoreMeanReversion(data[t], start_trading_date="2019-01-01", ticker=t, ma_length=50)
    for t in data
]

result = Simulator(
    strategies,
    capital=100_000,
    percent_of_capital=0.10,          # each new open ~10% of current equity
    min_cash=0.0,                     # longs must leave cash >= 0
    buying_power_policy=BuyingPowerPolicy.SCALE_DOWN,  # or SKIP
    commission_pct=0.0005,
    slippage_pct=0.0005,
    start_date="2019-01-01",
    finalize=True,
).run()

result.equity.plot(title="Shared-capital portfolio")
print(result.full_stats["sharpe_ratio"], result.num_trades)
```

**Useful result fields**

| Field | Meaning |
|---|---|
| `result.equity` | Account value each day |
| `result.cash` | Cash each day |
| `result.trades_df` | One row per completed trade |
| `result.skipped` | Opens that were blocked or shrunk |
| `result.full_stats` | Flat scorecard (Sharpe, drawdown, …) |
| `result.num_trades` | How many round trips closed |

**When to use**  
“If these signals share one account, how does the portfolio look?” without custom portfolio logic.

---

## 4. Shared wallet with portfolio rules (`SystematicMultiAsset`)

A **book** is the recommended way to run a multi-strategy / multi-ticker portfolio.

- **Sleeve** = one strategy style (class + params + tickers)  
- **Book** = one or more sleeves sharing cash  
- Hooks can **cap**, **rank**, or **veto** opens/closes  

```python
from trade.backtester_.systematic import Sleeve, SystematicMultiAsset
from strategy_research.resources.strategies.base import LongZScoreMeanReversion

book = SystematicMultiAsset(
    [
        Sleeve(
            "mr",
            LongZScoreMeanReversion,
            params={"ma_length": 50, "entry_zscore_threshold": -1.5},
            tickers=list(data),
        ),
    ],
    data,
    start_trading_date="2019-01-01",
    max_open_trades=5,                 # portfolio-wide open slot cap
    allow_multiple_per_asset=False,    # one sleeve per ticker at a time
)

result = book.simulate(
    capital=100_000,
    percent_of_capital=0.10,
    min_cash=0.0,
    buying_power_policy="scale_down",
    commission_pct=0.0005,
    slippage_pct=0.0005,
    start_date="2019-01-01",
)
```

**Two sleeves (still fine for simulate)**

```python
from strategy_research.resources.strategies.base import LongDonchianMomentum

dual = SystematicMultiAsset(
    [
        Sleeve("mr", LongZScoreMeanReversion, params={...}, tickers=list(data)),
        Sleeve("mom", LongDonchianMomentum, params={...}, tickers=list(data)),
    ],
    data,
    start_trading_date="2019-01-01",
    max_open_trades=8,
    allow_multiple_per_asset=True,
)
dual_result = dual.simulate(capital=100_000, percent_of_capital=0.05)
```

**Optimize a book**

Important: **`book.optimize` only works on a one-sleeve book.**  
If you have several sleeves, optimize each sleeve on its own one-sleeve book.

```python
# Shared-capital search (hooks stay on)
shared = book.optimize(
    space={"ma_length": [40, 50], "entry_zscore_threshold": [-1.5, -2.0]},
    simulator_kwargs={
        "capital": 100_000,
        "percent_of_capital": 0.10,
        "min_cash": 0.0,
    },
    per_asset=False,   # one grid for the whole book
    n_jobs=1,
    keep_top=1,
)
print(shared.best_params, shared.best_score)

# Per-ticker solo search (same idea as MultiAsset.optimize)
from trade.backtester_.optimizer import SOLO_SIMULATOR_KWARGS

per_name = book.optimize(
    space={"ma_length": [40, 50]},
    simulator_kwargs=SOLO_SIMULATOR_KWARGS,
    per_asset=True,
    n_jobs=1,
)
```

**When to use**  
Live-style portfolios: shared cash, open caps, multiple strategy styles.

---

## 5. Finding better settings (`ParamOptimizer`)

The optimizer tries every mix in a **space** (a small grid of settings), scores each run, and keeps the best ones.

Two shapes:

1. **Flat list of strategies** (shared capital, no book hooks)  
2. **Book factory** (rebuilds a book each trial so hooks apply)

```python
from trade.backtester_.optimizer import ParamOptimizer
from strategy_research.resources.strategies.base import LongZScoreMeanReversion

# A) Flat shared-capital grid
opt = ParamOptimizer(
    LongZScoreMeanReversion,
    data=data_by_ticker,
    start_trading_date="2019-01-01",
    space={"ma_length": [40, 50], "entry_zscore_threshold": [-1.5, -2.0]},
    simulator_kwargs={"capital": 100_000, "percent_of_capital": 0.10},
    objective="sharpe_ratio",
    constraints=[("max_drawdown", ">", -0.35)],  # optional
    n_jobs=1,
    keep_top=3,
    show_progress=True,
)
res = opt.run()
print(res.best_params, res.best_score)
res.best_result.equity.plot()

# B) Book-aware grid (hooks apply)
template = SystematicMultiAsset(
    [Sleeve("mr", LongZScoreMeanReversion, params={...}, tickers=list(data))],
    data,
    start_trading_date="2019-01-01",
    max_open_trades=3,
)
res = ParamOptimizer(
    space={"max_open_trades": [2, 3, 5]},
    simulator_kwargs={"capital": 100_000, "percent_of_capital": 0.10},
    book_factory=template.with_trial_params,
    n_jobs=1,
    keep_top=1,
).run()
```

**How scoring works (simple version)**

1. Every mix runs with **lean** stats (fast).  
2. Mixes are ranked by the objective (default: Sharpe).  
3. The top few (`keep_top`) are **re-run in full** so you get trade tables and exposure.

**Tips**

- Keep the grid small while learning (`2 × 2` mixes).  
- Use `n_jobs=1` first; parallel workers need a fork-safe environment.  
- Constraints that need trade counts may need `stats_level="full"` in `simulator_kwargs`.

---

## 6. Legacy: `PTBacktester`

`PTBacktester` is the **older** wrapper around [backtesting.py](https://kernc.github.io/backtesting.py/).

It runs **one ticker at a time** (iteratively), each with its own cash. It does **not** model a true shared-cash portfolio the way `Simulator` / `SystematicMultiAsset` do.

```python
from trade.backtester_.backtester_ import PTBacktester
from trade.backtester_.data import PTDataset
from strategy_research.resources.strategies.base import LongZScoreMeanReversion

datalist = [PTDataset(name=t, data=df) for t, df in frames.items()]

bt = PTBacktester(
    datalist=datalist,
    strategy=LongZScoreMeanReversion,   # class, not instance
    cash=100_000,                       # or {ticker: cash}
    strategy_settings={                 # optional per-ticker params
        "AAPL": {"ma_length": 50},
    },
    start_overwrite="2019-01-01",       # skip warmup before this date
    commission=0.0,
)

stats = bt.run()              # per-ticker stats table
bt.plot_portfolio()           # portfolio-style plot helper
bt.plot_position("AAPL")      # single-name backtesting.py plot

# Legacy optimize (backtesting.py style — different API from ParamOptimizer)
opt = bt.optimize(
    optimize_var={"ma_length": [40, 50, 60]},
    maximize="Return [%]",
)
```

**When to use PTBacktester**

- Old notebooks / research that already call it  
- Single-name backtesting.py plots you still need  

**When not to use it for new work**

- Shared cash across names → use `Simulator` / `SystematicMultiAsset`  
- Portfolio caps / ranking / vetoes → use a book  
- New parameter search → prefer `ParamOptimizer` / `.optimize` on MultiAsset or book  

---

## Money settings cheat sheet

| Setting | Plain meaning |
|---|---|
| `capital` | Starting cash of the shared account |
| `percent_of_capital` | New open size as a fraction of **current equity** (e.g. `0.1` = ~10%) |
| `trade_size` | Fixed dollar size per open (use this **or** percent, not both) |
| `min_cash` | Cash that must remain after a **long** open (`0` = no borrowing for longs; `None` = no floor) |
| `buying_power_policy` | If a trade is too big: `skip` it, or `scale_down` to what fits |
| `max_gross_leverage` | Cap on total position size ÷ equity |
| `commission_pct` / `slippage_pct` | Cash fee as % of notional (default `0`) |
| `wipeout_floor` | Stop the run if equity falls to this level |
| `start_date` | First day allowed to **trade** (series start one bus day earlier, flat) |
| `finalize` | Force-close leftovers on the last day |
| `tplusn` | Delay fills by N bars (0 = fill same day) |
| `max_open_trades` | Book-only: max positions open at once |
| `allow_multiple_per_asset` | Book-only: can two sleeves hold the same ticker? |

---

## Solo vs shared — one picture

```text
Solo / MultiAsset.optimize(per_asset)
┌─────────┐  ┌─────────┐  ┌─────────┐
│  AAPL   │  │  MSFT   │  │  NVDA   │   each has its own imaginary account
└─────────┘  └─────────┘  └─────────┘

Simulator / SystematicMultiAsset
┌───────────────────────────────────┐
│         one shared wallet         │
│   AAPL + MSFT + NVDA compete      │
└───────────────────────────────────┘
```

---

## Suggested workflow

1. **Solo** one ticker with `simulate()` — does the idea fire at all?  
2. **MultiAsset** across your universe — which names behave?  
3. **Optimize per ticker** (solo) — rough good settings per name.  
4. **One-sleeve book** with shared capital — real portfolio behavior + caps.  
5. **Book optimize** (shared) — tune settings under those portfolio rules.  
6. Only then add a **second sleeve** for simulate; optimize sleeves **one at a time**.

---

## Where things live in code

| Piece | Path |
|---|---|
| Shared-cash day loop | `trade/backtester_/simulator/` |
| Multi-ticker container | `trade/backtester_/_multi_asset_strategy.py` |
| Books / sleeves | `trade/backtester_/systematic/` |
| Grid optimizer | `trade/backtester_/optimizer/` |
| Strategy base class | `trade/backtester_/_strategy.py` |
| Legacy PTBacktester | `trade/backtester_/backtester_.py` |
| Demo script | `trade/backtester_/examples/demo_simulator_stack.py` |

---

## Need help reading a result?

Start with:

1. `result.equity.plot()` — did wealth go up?  
2. `result.num_trades` — did it trade enough to matter?  
3. `result.full_stats["max_drawdown"]` — how painful were the dips?  
4. `result.skipped` — were good signals blocked by cash or caps?  
5. `result.trades_df` — look at a few winners and losers by hand.  

That is usually enough to decide whether to keep digging or change the idea.
