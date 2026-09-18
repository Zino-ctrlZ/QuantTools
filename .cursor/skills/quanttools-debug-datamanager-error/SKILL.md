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

**Focus on root cause fixing instead of front end patches.** After the defect is isolated, propose (and apply, if asked) the change at the layer that actually broke the contract — adapter mapping, `LoadRequest` dates, cache key, certification, dbase shaping, vendor params. Do not paper over the symptom in the caller, notebook, try/except, empty-frame default, or a one-off skip for this trade id.

Scratch scripts and dumps go in QuantTools `.sandbox/datamanager-debug/` (gitignored). Do not write debug artifacts under `logs/` or tracked paths.

## Required order

```
Task Progress:
- [ ] 1. Capture the failing call (exact API + args + env)
- [ ] 2. Investigate the trade id that created the error
- [ ] 3. Recreate the exact error
- [ ] 4. Iteratively pinpoint (one layer at a time)
- [ ] 5. If code structure: locate the structural defect
- [ ] 6. If ThetaData API: hit the vendor URL directly
- [ ] 7. Classify and report (fix only if the user asked)
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

Typical split:

- Exception or empty **before** any network / `retrieve_*` → treat as **code structure**.
- `ThetaDataNotFound` (vendor **472**), other ThetaData status codes, empty vendor payload, timeouts → treat as **ThetaData API** and do the direct URL call. Do not stop at the Python wrapper.

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

## 5. Code structure

If the failure is in QuantTools (or dbase parameter shaping), **find** the defect:

- Trace the reproduced kwargs through the call stack; do not guess from architecture docs alone.
- Check adapter mappings, `LoadRequest` date policy, cache keys vs identity fields, date guardrails vs vendor calendar, certification preflight vs sanitizers, singleton `INSTANCES` reuse.
- Quote file + function + the rule that was violated.
- Do not implement a fix unless the user asked. Recreate after any change.
- When a fix is in scope: **root cause, not a front-end patch.** A caller guard, extra `fillna`, swallowing `ThetaDataNotFound`, or skipping the failing trade is not a fix.

## 6. ThetaData API → direct URL

If the issue is from the ThetaData API, **try a direct call through the URL** (curl/`httpx`/`requests`), bypassing `dbase` and datamanager.

Build the same path/query the wrapper would use. Strike on v2 hist option endpoints is typically **price × 1000** as an integer (`150.0` → `150000`). Dates are `YYYYMMDD`.

Minimal probe:

```bash
curl -sS -D - "http://127.0.0.1:25510/v2/hist/option/eod?root=AAPL&exp=20250620&strike=150000&right=C&start_date=20250115&end_date=20250115&use_csv=true"
```

Interpret:

| Direct URL | Wrapper / manager | Meaning |
|---|---|---|
| Terminal refused / timeout | any | ThetaData process not reachable |
| Vendor **472** / `ThetaDataNotFound` or empty body | wrapper `ThetaDataNotFound` or empty DF | no tape for that request, not manager cache |
| URL returns rows, wrapper empty/raises | Python | **code structure** in dbase or datamanager mapping |
| URL and wrapper both empty | — | likely no data for that contract/window |

URL recipes, v2 vs v3, and how to extract the wrapper's URL: [thetadata-direct.md](references/thetadata-direct.md).

## 7. Report

```markdown
## Recreated
- Trade id: <trade_id / TradeID / opttick that created the error — or why it could not be recovered>
- Call: <exact API + kwargs recovered from the failure — not a stand-in>
- Symptom: ...

## Layer
- Isolated at: caller | adapter | LoadRequest/dates | manager/cache | certification | dbase | ThetaData HTTP

## Evidence
- Structure: <file, function, broken rule>  OR  none
- Direct URL: <command, status, body summary>  OR  not applicable

## Cause
- One sentence.

## Fix
- Root cause change at the isolated layer (do not apply unless asked).
- Rejected front-end patches: <caller/notebook/try-except/skip-this-trade workarounds that would hide the same defect>
```

## Additional resources

- [thetadata-direct.md](references/thetadata-direct.md) — vendor URL encoding, v2 paths, curl probes, URL-vs-wrapper decision table.
