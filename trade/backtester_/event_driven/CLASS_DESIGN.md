# `event_driven` — Class Design

> Living, high-level class plan for `trade/backtester_/event_driven/`.
> Last updated: 2026-11-01 · Status: design only, no code yet.
>
> Companion: detailed redesign notes in [`PLAN.md`](PLAN.md). When the two disagree, this file's settled choices win until `PLAN.md` is updated to match.

The old `EventDriven` package stays as it is. This package sits beside it.
`SystematicMultiAsset` is the only signal source. Stocks and options are both supported.
Long-only: an order with direction `SHORT` fails for every product. A signal of type `SHORT` that buys a put is fine.

---

## Settled choices

| Topic | Choice |
|---|---|
| Package path | `trade/backtester_/event_driven/` |
| Shared types | Live in this package; `EventDriven.types` re-exports them for old imports |
| Direction | Long-only. Shorting any product fails. Long puts are fine |
| Fills | Always at the close |
| Results | Subclass `SimulationResult` and add the new fields |
| Errors | Configurable: stop the run, or log and continue |
| Portfolio roles | `Portfolio` holds information; `PortfolioManager` only records changes |
| Portfolios per run | One |
| Position review | Lift and use `PositionAnalyzer` as-is |
| Review timing | Analyze at today's close; execute the next trading day |
| Legacy sizers | Not in the first build; revisit later |
| Corporate unsupported events | Close the position at the last price and record why |
| Type tooling | Whatever this repo already uses |
| Corporate date meaning and dividend cash timing | Decide later |
| Data loading | Done inside the backtester. Built-in loaders cover stocks and options. Users can register a custom data loader |

---

## How one day runs

0. Before the calendar starts, the backtester loads all data it needs for the run through the registered data loaders.
1. Day opens (`MarketEvent`).
2. Corporate events due today are applied.
3. Open positions are revalued at today's close.
4. Wipeout check.
5. Strategies emit signals; delayed signals due today are released.
6. Signals become orders; closes before opens; one order finishes before the next starts.
7. `PositionAnalyzer` reviews open positions at the close. Any close or roll it decides executes on the next trading day.
8. Books are checked and the day is saved.

---

## Events

Kept from EventDriven, same names and meaning:

- `MarketEvent`, `SignalEvent`, `OrderEvent`, `FillEvent`, `RollEvent`, `ExerciseEvent`

New:

- `OrderRejectedEvent` — broker gave up
- `PositionChangedEvent` — a position opened, reduced, closed or adjusted
- `CorporateEvent` family — split, dividend, merger, symbol change, delisting, …
- `CorporateAdjustmentEvent` — the portfolio change that follows a corporate event
- `StopEvent` — end the backtest after today

Day stages are not events. Skip notes go in the journal, not on the event list.

The `EventScheduler` is the ordered to-do list. Events are processed by `(datetime, priority, sequence)`.

---

## Class map

### Engine and wiring

| Class | Role |
|---|---|
| `EventDrivenBacktest` | Builds the pieces, runs the day stages, returns the result |
| `EngineConfig` | Typed settings for capital, products, fills, corporate policy, error mode |
| `EventScheduler` | Holds and orders events |
| `Journal` | Event log, skip notes, daily snapshots |
| `EventDrivenResult` | Extends `SimulationResult` with corporate-action records and related extras |
| `as_book` | Converts `StrategyBase` / `MultiAssetStrategy` into `SystematicMultiAsset` |

### Data loading

Data is loaded by the backtester, not by the caller beforehand. Products and corporate handlers ask the data layer for prices and schedules. Users can swap in their own loader.

| Class | Role |
|---|---|
| `DataLoader` | Base: load bars, quotes, and related series for a ticker set and date window |
| `StockDataLoader` | Default stock loader (strategy datasets / market data already used in this stack) |
| `OptionDataLoader` | Default option loader (existing option timeseries path) |
| `DataRegistry` | Holds the active loaders; `register(kind, loader)` lets the user plug in a custom one |
| `MarketData` | Read-only store the backtester fills once at start and that products read during the run |

### Shared types (new home)

| Class / type | Role |
|---|---|
| `SignalID`, `TradeID`, `PositionEffect`, `SignalTypes`, `EventTypes`, … | Moved here from `EventDriven.types`. Old package re-exports them so existing imports keep working |

### Actors (workers)

Every worker inherits from `Actor`.

| Class | Role |
|---|---|
| `Actor` | Base: start, handle stage, handle event, reset |
| `BookSignalSource` | Asks `SystematicMultiAsset` for decisions; emits `SignalEvent`; holds delayed signals |
| `OrderPlanner` | Approves signals, builds `OrderEvent`s; fails any order with direction `SHORT` |
| `Broker` | Turns orders into fills at the close; liquidity and costs; emits `FillEvent` or `OrderRejectedEvent` |
| `PortfolioManager` | Only writer of the `Portfolio`; applies fills and corporate adjustments; revalues; checks books |
| `PositionAnalyzer` | Lifted from EventDriven. Reviews positions at the close; actions execute next trading day |
| `RollCoordinator` | Runs a roll as close-then-open under one id |
| `CorporateEventActor` | Schedules and applies corporate events |
| `WipeoutGuard` | Closes everything and stops if equity hits the floor |
| `ContractLifecycleActor` | Option expiry / exercise — later |

### Portfolio

| Class | Role |
|---|---|
| `Portfolio` | The client's account: cash, positions, trades, corporate records. Answers questions. Enforces long-only and cash identity |
| `PortfolioView` | Read-only face of the portfolio for every other worker |
| `PortfolioManager` | Steward that records changes into the portfolio |
| `Lot`, `Position`, `PositionBook` | What is held |
| `TradeLedger` | Completed trades |
| `CashLedger` | Every cash movement with a reason |
| `PortfolioChange` | What a fill or adjustment just did (`Opened`, `Reduced`, `Closed`, `Adjusted`) |

### Cash

| Class | Role |
|---|---|
| `CashAllocator` | Base for how capital is pooled |
| `SharedCash` | One shared pool. Default. Uses the simulator `Account` |
| `CashPerAsset` | One pool per ticker from a weight map. Optional; for parity with old EventDriven |

### Products

| Class | Role |
|---|---|
| `Product` | Base: pick instrument, quote, mark, size, tradability |
| `StockProduct` | Stocks |
| `OptionProduct` | Options. Uses lifted `RiskManager` and `OrderPicker` |
| `StockInstrument`, `OptionInstrument`, `OptionContract` | Typed instruments |
| `ProductRegistry`, `ProductRouter` | Register and choose products per run / ticker / sleeve |

### Option machinery (lifted, not rewritten)

| Class | Role |
|---|---|
| `RiskManager` | Builds option orders (`get_order`) |
| `OrderPicker` | Picks the contract |
| `PositionAnalyzer` | Reviews open option positions |

Legacy sizers stay out of the first build.

### Routing and fills

| Class | Role |
|---|---|
| `SignalRouting` | Base for how signals are admitted |
| `DirectRouting` | Every signal goes through; closes first |
| `BookRouting` | Uses `PortfolioBook` hooks; one candidate at a time |
| `FillModel` | Base for how a fill price is chosen |
| `CloseFill` | Fill at the close. The only model for now |
| `QuoteFill` | Later, if needed |

### Corporate events

| Class | Role |
|---|---|
| `CorporateEvent` and subtypes | Split, dividend, stock dividend, merger, symbol change, delisting, spin-off |
| `CorporateAdjustment` and subtypes | Scale quantity, credit cash, swap instrument, force close |
| `CorporateEventHandler` | Turns an event into adjustments for a product |
| `CorporateEventSource` | Base for where events come from |
| `NoCorporateEvents` | Default; matches the simulator |
| `StaticCorporateEvents` | Fixed list for tests |
| `DataManagerCorporateEvents` | Later, from existing data managers |
| `CorporatePolicy` | Unsupported-event policy and related settings |

### Reused from outside this package

| Item | From |
|---|---|
| `SystematicMultiAsset`, `Sleeve`, `PortfolioBook` | `systematic` |
| `Account` | simulator |
| `SimulationResult` parent, `TradeRecord`, `DailySnapshot`, `DayRecorder`, `ResultBuilder` | simulator |
| Validation mixins | `trade.helpers.helper_types` |

---

## Build order (short)

| Phase | Delivers |
|---|---|
| 0 | Types, events, `Actor`, config, empty package |
| 1 | Scheduler, day stages, and data-loader registry with a stock loader |
| 2 | Stock path end to end; parity with `Simulator` |
| 3 | Book routing and wipeout |
| 4 | Costs, liquidity, retries |
| 5 | Corporate events for stocks |
| 6 | Strategy converters |
| 7 | Options with lifted `RiskManager` / `OrderPicker` / `PositionAnalyzer` |
| 8 | Reporting and comparison with old results |
| 9 | Extras (expiry, contract adjustment, …) |

---

## Open later

- How stock price series relate to splits and dividends when both are applied.
- Exact meaning of dividend dates in the data layer.
- Whether dividend cash is credited on the ex-date or the pay date.
- Whether to bring legacy sizers back.
- Whether every event should run through the validation mixin, or only configuration objects.
