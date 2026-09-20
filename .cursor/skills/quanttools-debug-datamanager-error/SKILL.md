---
name: quanttools-debug-datamanager-error
description: >-
  Debug QuantTools trade.datamanager failures by investigating the trade id that
  created the error, recreating the exact failure, then iteratively peeling layers
  until the cause is isolated. Use when debugging DataManager, TimeseriesDataManager,
  option_spot, ThetaDataNotFound, empty timeseries, certification, LoadRequest, NA
  logs, retrieve_eod_ohlc, trade_id, or ThetaData HTTP/API issues.
---

# Debug QuantTools DataManager errors

Do **not** patch, refactor, or "try a likely fix" until the original failure has been recreated.

**Raise site is not fault.** DataManager raising does not mean DataManager is at fault. Classify **fault of** (who broke the contract), then handle that layer — not the messenger.

**Focus on root cause fixing instead of front end patches.** After the defect is isolated, propose (and apply, if asked) the change at the layer that actually broke the contract — adapter mapping, `LoadRequest` dates, cache key, certification, dbase shaping, vendor params. Do not paper over the symptom in the caller, notebook, try/except, empty-frame default, or a one-off skip for this trade id.

Scratch scripts and dumps go in QuantTools `.sandbox/datamanager-debug/` (gitignored). Do not write debug artifacts under `logs/` or tracked paths.

## Required order

```
Task Progress:
- [ ] 1. Capture the failing call (exact API + args + env)
- [ ] 2. Investigate the trade id that created the error
- [ ] 3. Recreate the exact error
- [ ] 4. Iteratively pinpoint (one layer at a time)
- [ ] 5. Three-boundary check (manager call / retrieve_* / direct URL)
- [ ] 6. Classify: raised at vs fault of
- [ ] 7. Handle by fault (fix only if the user asked)
- [ ] 8. Report
```

## 1. Capture

From traceback, notebook cell, or user message, record:

| Field | Examples |
|---|---|
| Public API | `TimeseriesDataManager(...).option_spot.get_timeseries(...)`, `OptionSpotDataManager.get_option_spot_timeseries`, `load_full_option_data`, `LoadRequest` |
| Identity | `symbol`, `strike`, `right`, `expiration` |
| Trade id | `trade_id` / `TradeID` / `opttick` / `position_id` that produced the failing load |
| Window | `start_date`, `end_date`, `as_of`, `rt`, `on_date` |
| Options | `endpoint_source`, `undo_adjust`, `vol_model`, `fall_back_option` |
| Symptom | exception type/message, empty frame, NA rows, certification fail |
| Env | `THETADATA_USE_V3`, whether the ThetaData terminal is expected on `127.0.0.1:25510` |

`LoadRequest` rewrites windows (`rt` → today/`on_date`; same start=end → `as_of`). Capture **caller args and post-init args**.

## 2. Investigate the trade id that created the error

**Try to investigate for the trade id that created the error.** Do this before recreating a lower-level datamanager call. The trade id is the identity of the failing load; without it you will debug a stand-in contract.

Recover it from:

- Traceback / log: `order["data"]["trade_id"]`, `Calculating Greeks for {_id}`, `Loading option spot data for symbol`
- Walk-forward / `RiskManager.get_order` / backtest: `trade_id`, **opttick** (or `position_id` + parse), **req.date**
- Portfolio / trades CSV / analyzer: `trades_map`, `TradeID`, notebook locals

From the trade id, recover **opttick** (or parse `position_id`), **req.date**, and the timeseries window actually passed to `load_full_option_data` (`BacktestTimeseries.start_date` / `end_date`, then `_sync_date`). Log lines: `Calculate Greeks Dates Start/End`, `Calculating Greeks for {_id}`, `Loading option spot data for symbol`.

If identity, trade id, or window is missing from the traceback, **stop and find it** (notebook locals, log files, one print in the failing frame, re-run the user's cell). Do not invent a stand-in contract.

## 3. Recreate the exact error

The recreate **is the actual failing call**. Same public API, same identity (`symbol` / strike / right / expiration / opttick / **trade id**), same date window (caller + post-`LoadRequest` / post-`_sync_date`), same env (`THETADATA_USE_V3`, `OptionDataConfig`).

Write that call under `.sandbox/datamanager-debug/`. Prefer the user's entry point (`analyzer.build`, `load_full_option_data`, `TimeseriesDataManager`, …). Do not start one layer lower for the first recreate.

**Forbidden as the recreate** (these are later isolation steps only):

- Fake tickers (`ZZZRECREATE`, `TEST`, …)
- Invented strikes / expirations / windows
- `patch` of `retrieve_eod_ohlc` / `list_dates` / `ThetaDataNotFound`
- Hand-built DataFrames passed to `OptionSpotResult`
- A "similar" liquid contract (e.g. AAPL 105C) instead of the one that failed

Confirm the **same** exception type/message, empty shape, or NA/certification symptom before any further work.

If the actual call cannot be recovered: **stop**. Report what is missing. Do not debug a substitute failure.

Bypass cache only **after** that actual call has reproduced (`get_enable_caching` / cache clear), not as the first recreation.

## 4. Iteratively pinpoint

Peel **one layer at a time**. Keep all other inputs fixed. After each step, ask: does the original symptom still appear?

Outside → inside:

1. **Caller** — wrong strike/right/exp/dates (right `"C"`/`"P"` vs `"call"`/`"put"`).
2. **Adapter** — `trade/datamanager/timeseries.py` method name mapping.
3. **Request/window** — `LoadRequest.__post_init__`, `_sync_date` / `_sync_equity_date`, `list_dates` clipping in `utils/date.py`.
4. **Manager** — cache key, partial cache, `classify_option_spot_dates`, empty-frame handling.
5. **Certification** — `certify_manager_result`, `DataNotCertifiedException`, L1/L2/L3.
6. **dbase wrappers** — `retrieve_eod_ohlc`, `quote_to_eod_patch`, `retrieve_quote_rt`, `retrieve_ohlc`, `list_dates` (`dbase.DataAPI.ThetaData`; not in this repo — `inspect.getfile`).
7. **HTTP ThetaData** — raw URL against the terminal (see [thetadata-direct.md](references/thetadata-direct.md)).

Peel finds **raised at**. It does not decide **fault of**. Continue to the three-boundary check.

## 5. Three-boundary check

Same identity and window at three boundaries (after recreate; cache bypassed):

1. Public DataManager / `load_full_option_data` call.
2. `retrieve_eod_ohlc` / `retrieve_ohlc` / `list_dates` **kwargs + return**.
3. Direct terminal URL (see § Direct URL).

Do the direct URL whenever the peel reaches `retrieve_*`, empty/472, or a certification fail. Do not stop at the Python wrapper. Do not treat “exception before network” as a skip of (3) if `retrieve_*` was never reached for a reason other than a proven caller/adapter kwargs bug.

## 6. Classify: raised at vs fault of

**DataManager raising does not mean DataManager is at fault.** Fault is the first layer that produced something the next layer is entitled to reject. Everything above that is a messenger.

Certification is a **detector**, not a fault bucket:

- L1/L2/L3 fail on empty/NA → usually **returned data / response**
- Fail after a good vendor payload → **DataManager** (over-strict or wrong window)
- Fail after a good URL and a bad DataFrame → **dbase tooling**

### Fault buckets

| Fault of | Contract that broke |
|---|---|
| **Caller** | TFP-Algo / notebook / analyzer passed the wrong identity or window (strike, right `"C"`/`"P"` vs `"call"`/`"put"`, exp, dates) **before** QuantTools rewrote it. |
| **DataManager** | QuantTools built a bad request, reused a bad cache key, mapped the adapter wrong, clipped dates incorrectly, or treated a *valid* empty/partial as a crash. |
| **ThetaData dbase tooling** | FinanceDatabase `retrieve_*` / `list_dates` / `quote_to_eod_patch` / `raise_thetadata_exception` shaped, parsed, or patched so the DataFrame/exception is not what the terminal said. |
| **Returned data / response** | Terminal (or true empty tape) answered **this exact request** with no rows, vendor **472**, or junk that the wrapper passed through faithfully. |

Keep **caller** explicit. Fold it into DataManager **only** when `LoadRequest` / `_sync_date` / adapter rewrote identity.

### Discriminator

| 1 Manager | 2 Wrapper | 3 URL | Fault of |
|---|---|---|---|
| bad kwargs vs trade id / opttick | — | — | **Caller** (or DataManager if QuantTools rewrote them) |
| raises / empty / cert fail | rows that match the request | rows | **DataManager** |
| raises / empty | empty / wrong exception / wrong shape | rows or matching vendor status | **dbase tooling** |
| raises / empty / cert fail | empty / 472 / same junk | empty / 472 / same junk | **Returned data / response** |
| raises on empty index | empty DF, 200 + no rows | empty / 472 | **Returned data / response** (raise in DataManager is allowed) |

Empty after `list_dates` clip: **DataManager** if we rewrote a window the vendor calendar cannot serve; **returned data** if the contract has no sessions for the requested window.

Do not call missing tape HTTP 404. Use inner status + exception (§ ThetaData error names).

## ThetaData error names

Do **not** say HTTP 404 for missing option tape. Via the proxy the HTTP status is often 200; the **inner** `status_code` is what `raise_thetadata_exception` maps.

When talking about a failure, use **status code + Python exception** from `dbase.DataAPI.ThetaExceptions.raise_thetadata_exception`:

| Inner status | Exception | Meaning |
|---|---|---|
| 200 | (ok) | Data returned |
| 400 (future-date body only) | `ThetaDataContainsFutureDateError` | Session still in the future |
| 404 | `ThetDataNoImplementation` | Feature not implemented — **not** "no rows" |
| 429 | `ThetaDataOSLimit` | OS / rate limit |
| 470 | `ThetaDataGeneral` | General error |
| 471 | `ThetaDataPermission` | Permission denied |
| **472** | **`ThetaDataNotFound`** | **No data for that request** |
| 473 | `ThetaDataInvalidParameter` | Bad params |
| 474 | `ThetaDataDisconnected` | Disconnected |
| 475 | `ThetaDataParseError` | Parse error |
| 476 | `ThetaDataWrongIP` | Wrong IP |
| 477 | `ThetaDataNoPageFound` | No page |
| 570 | `ThetaDataLargeData` | Payload too large |
| 571 | `ThetaDataServerRestart` | Terminal restart |
| 572 | `ThetaDataUncaughtException` | Uncaught terminal error |
| other | `ThetaDataUnknownError` | Unmapped |

Say: vendor **472** / `ThetaDataNotFound`. Never call missing tape a 404.

## 7. Fault handling

Handle **fault of**, not exception type. DataManager **may raise** on returned-data faults; that raise can be correct. Do not patch the messenger.

Do not implement a change unless the user asked. Recreate after any change.

### Caller

Fix the call site (trade id → opttick / strike / right / exp / window). Do not add QuantTools guards, skips, or `fillna` for this contract.

### DataManager

Fix in QuantTools at the violating rule: adapter mapping, `LoadRequest` dates, cache key vs identity, `_sync_date` / `list_dates` policy, empty-index assumption, certification vs sanitizers, `INSTANCES` reuse.

Trace the reproduced kwargs through the call stack; do not guess from architecture docs alone. Quote file + function + the rule that was violated.

Not a fix: caller try/except, swallowing `ThetaDataNotFound`, skipping this trade id, empty-frame default, notebook workaround.

### ThetaData dbase tooling

Stop QuantTools work. Fix in FinanceDatabase (`inspect.getfile` on `retrieve_*`). Do not compensate in QuantTools (second strike×1000, local status-code map, “if empty try v2”).

Direct URL is mandatory. Without boundary (3) you cannot tell dbase from vendor.

### Returned data / response

Not a code bug by default. Do **not** invent rows in DataManager.

| Subtype | Handling |
|---|---|
| True no-tape (URL and wrapper agree: **472** / empty) | Propagate absence (`ThetaDataNotFound`, classified missing dates, NA log). Use `fall_back_option` / listed-session rules **only if already the contract**. |
| Faithful but unusable payload (wrong dtypes, future session, HTTP 200 + inner error passed as DF) | Prefer wrapper raising the mapped Theta exception (if mapping is wrong → **dbase**). If the vendor sent junk, certification **rejects**; do not sanitize into a fake series. Vendor/ticket, not a QuantTools reshape. |
| Partial tape (some dates missing, some NA columns) | Classification + NA logging + certification level. Tightening L2/L3 is a DataManager **policy** change. Filling holes is a workaround unless the holes are from a bad window (then DataManager). |

Live trading (fail closed vs fallback vs NA row) is TFP-Algo product policy. This skill does not invent a QuantTools default beyond **do not fabricate**.

## Direct URL

When the three-boundary check needs the terminal, **call the URL directly** (curl/`httpx`/`requests`), bypassing `dbase` and datamanager.

Build the same path/query the wrapper would use. Strike on v2 hist option endpoints is typically **price × 1000** as an integer (`150.0` → `150000`). Dates are `YYYYMMDD`.

Minimal probe:

```bash
curl -sS -D - "http://127.0.0.1:25510/v2/hist/option/eod?root=AAPL&exp=20250620&strike=150000&right=C&start_date=20250115&end_date=20250115&use_csv=true"
```

Interpret:

| Direct URL | Wrapper / manager | Fault of |
|---|---|---|
| Terminal refused / timeout | any | ThetaData process not reachable (ops / returned response) |
| Vendor **472** / empty body | wrapper `ThetaDataNotFound` or empty DF | **Returned data / response** (not manager cache) |
| URL returns rows, wrapper empty/raises | kwargs at wrapper were correct | **dbase tooling** |
| URL returns rows, wrapper empty/raises | manager kwargs were wrong | **DataManager** (or caller) |
| URL and wrapper both empty / both 472 | — | **Returned data / response** |

URL recipes, v2 vs v3, and how to extract the wrapper's URL: [thetadata-direct.md](references/thetadata-direct.md).

## 8. Report

```markdown
## Recreated
- Trade id: <trade_id / TradeID / opttick that created the error — or why it could not be recovered>
- Call: <exact API + kwargs recovered from the failure — not a stand-in>
- Symptom: ...

## Raised at
- caller | adapter | LoadRequest/dates | manager/cache | certification | dbase | ThetaData HTTP

## Fault of
- caller | DataManager | ThetaData dbase tooling | returned data/response

## Boundaries
- Manager: <kwargs + symptom>
- retrieve_*: <kwargs + return / exception>
- Direct URL: <command, inner status, body summary>  OR  not reached (and why)

## Evidence
- Broken rule: <file, function>  OR  none (data absence / vendor junk)

## Cause
- One sentence: raised in X because Y (fault of Z).

## Handling
- caller fix | QuantTools fix | FinanceDatabase fix | no code fix (propagate / certify / vendor)
- Do not apply unless asked.
- Rejected messenger patches: <try/except, skip-this-trade, fillna, empty default>
```

## Additional resources

- [thetadata-direct.md](references/thetadata-direct.md) — vendor URL encoding, v2 paths, curl probes, URL-vs-wrapper decision table.
