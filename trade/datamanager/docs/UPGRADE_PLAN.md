# DataManager Timeseries Certification — Plan

Status: **Complete** (certification + point-in-time wired; deep smoke test 25/25)

This document records the datamanager timeseries upgrade: structural sanitize,
certification levels (L1/L2/L3), manager return-boundary wiring, point-in-time
helpers, and date-sync policy. See [FUTURE_CHANGES.md](FUTURE_CHANGES.md) for backlog.

Legend: `[x]` done · `[~]` partial · `[ ]` not started

---

## Current Architecture

### Certification package (`trade/datamanager/certification/`)

| Module | Role |
|--------|------|
| `findings.py` | `IssueCode`, `CertificationFinding`, `OPTION_ARTIFACT_TYPES` |
| `types.py` | `CertificationResult`, `CertificationReturn` (+ `start_date` / `end_date`) |
| `context.py` | `setup_context`, `effective_checked_missing_dates()` |
| `checks.py` | Detect-only structural checks; `duplicate_index_labels()` |
| `respond.py` | L1 log, L2 raise, L3 fix dispatch |
| `fixers.py` | L3 repairs; respects `checked_missing_dates` for option artifacts |
| `pipeline.py` | `certify()`, `l1/l2/l3_certification()` |
| `start.py` | `DataCertificationManager.certify_result()`; reports → `certification.report` logger |
| `integration.py` | **`certify_manager_result()`** — NA log pre-cert → certify → `is_certified=True` |
| `market_timeseries.py` | MT adapter: throwaway `MarketTimeseriesFactorResult` + synthetic key |
| `report.py` | `draft_certificate_report()` (includes date range) |
| `exceptions.py` | `DataNotCertifiedException` (L2), config errors |
| `helpers.py` | Legacy L1 helpers (superseded by `checks`/`pipeline` for manager path) |

### Return-boundary flow (all certified timeseries paths)

```
fetch / cache hit
  → merge (if partial)
  → _data_structure_sanitize (structural only)
  → cache write (post-sanitize, pre-certify)
  → certify_manager_result
       → log_retrieval_na (forensics on pre-cert payload)
       → resolve checked_missing_dates from vendor (option artifacts)
       → DataCertificationManager.certify_result (L1/L2/L3)
       → result.is_certified = True
```

**Cache boundary:** cache stores post-sanitize, **pre-certify** data. L3 fixes are not
persisted to cache.

**`is_certified`:** certifier ran on this return (distinct from `CertificationResult.success`).

---

## Master Checklist

### Foundation

- [x] `CertificationLevel` enum (`L1`, `L2`, `L3`) in `_enums.py`
- [x] `certification_level` on `OptionDataConfig` (default `L2`)
- [x] `certification_level` per-call override on certified timeseries public APIs
- [x] `DataNotCertifiedException` in `certification/exceptions.py` (L2 raises)

### Sanitize (structural only)

- [x] `_data_structure_sanitize` — tz strip, DatetimeIndex, sort, lowercase cols, clip, holiday filter
- [x] Removed from sanitize: dedupe, B-day reindex, ffill (moved to L3 fixers)
- [x] **MarketTimeseries cache writes** — `_sanitize_market_factor_for_cache` in `_load_*_into_cache`
      paths before `_cache_it_timeseries_data_structure`.

### Certification levels

- [x] L1: audit + log issues; return data as-is
- [x] L2: raise `DataNotCertifiedException` on unexplained violations
- [x] L3: fix (ffill / dedupe / reindex); residual re-check; write fixes to `result.timeseries`
- [x] Option artifacts: `checked_missing_dates` kwarg exempts NaNs and calendar gaps at cert time
- [x] Certification report includes date range; duplicate-index drops list the labels
- [~] Gate checked-missing exemption strictly by `artifact_type` (kwarg honored whenever passed today)

### Manager wiring — timeseries (certified)

| Manager | Method(s) | Notes |
|---------|-----------|-------|
| `vol.py` | `get_implied_volatility_timeseries` | `_certify_option_model_result`; vendor gaps via `resolve_checked_missing_dates_for_option_artifact` |
| `greeks.py` | `get_greeks_timeseries` | Same |
| `rates.py` | `get_risk_free_rate_timeseries` | Partial-cache gap fetch fixed |
| `forward.py` | `get_forward_timeseries` | |
| `option_spot.py` | `get_option_spot_timeseries` | Vendor gaps via `resolve_checked_missing_dates_for_option_contract` |
| `dividend.py` | `get_schedule_timeseries` | DISCRETE: certify; CONTINUOUS: MT upstream (skip re-cert) |
| `spot.py` | `get_spot_timeseries` | MT upstream — `is_certified=True`, no double cert |
| `market_data.py` | `_get_*_timeseries` | `certify_market_factor_payload` (sanitize + certify at getter) |

### Point-in-time (wired)

- [x] `utils/point_in_time.py` — `resolve_value_at_date()` (10 BDay lookback, L1 cert)
- [x] Wired: `get_rate`, `get_at_time` (spot), `get_option_spot`, `get_forward`, `get_schedule`,
      `get_at_time_implied_volatility`, `get_at_time_greeks`, `get_at_index`, `get_split_factor_at_index`
- [x] All `rt()` methods delegate to the above

Point-in-time paths do **not** use `@log_na_after_retrieval`; NA forensics run on the
underlying L1 lookback fetch inside certification. `resolve_value_at_date` owns fallback
selection only (see `docs/README.md` — Certification → NA logging split).

### `checked_missing_dates` policy (**resolved**)

**Decision: do not populate `result.checked_missing_dates`. Cert always resolves from vendor.**

| Concern | Resolution |
|---------|------------|
| Why was `result.checked_missing_dates` proposed? | So downstream code (LoadRequest, re-cert, notebooks) could skip a vendor call and trust the return object. |
| Why skip it? | Vendor `list_dates` / classification is the source of truth; stale cache metadata on the result would diverge from what cert actually used. Resolution at cert time is cheap and keeps one path. |
| What stays? | **Vendor resolve at cert call** (`resolve_checked_missing_dates_for_option_contract` / `_artifact`). **Cache metadata** (`_CachedData.checked_missing_dates`) for coverage math and avoiding re-fetch of known-absent dates. |
| Field on `_OptionResultsBase`? | **Removed** — cert and cache metadata only; not on manager results. |

```
vendor classify → checked_missing_dates
        ↓
_data_structure_cache_it(..., checked_missing_dates=...)  →  _CachedData (coverage / no re-fetch)
        ↓
certify_manager_result(..., checked_missing_dates=resolve_from_vendor(...))   ← always vendor
        ↓
result.checked_missing_dates                             ← removed from Result types
```

### LoadRequest `_is_missing_dates` awareness

**What it does today** (`requests.py`):

When vol/greeks/theo `LoadRequest` is constructed with pre-loaded timeseries
(e.g. `spot_timeseries=some_result`), `_validate_provided_inputs` calls
`_is_missing_dates(start, end, series)` which runs `get_missing_dates` over the full
B-day window. If any business days are absent from the index, it logs a warning,
**discards the provided result**, and forces a reload from source.

**Why this matters for certification:**

Option spot (and vol/greeks that share its calendar) can have **vendor-confirmed absent
dates** — business days ThetaData was queried for but returned no usable print. Certification
treats those as explained gaps via `checked_missing_dates`. `_is_missing_dates` does **not**
know about them: it sees a hole in the B-day grid and treats the series as incomplete.

**Effect:** A caller passing a correctly certified option-spot result with explained gaps
may have it thrown away; LoadRequest reloads from source anyway.

**Planned fix (if we keep pre-loaded inputs):**

- [x] When validating provided option-related timeseries, subtract
      `resolve_checked_missing_dates_for_option_artifact(...)` from the missing-date set
      before deciding incompleteness (`requests.py`).

Low priority until pre-loaded paths are used heavily in production flows.

### Live / today edge cases (**priority**)

Tension between cache persistence policy, certification window, and real-time fetches:

| Mechanism | Role |
|-----------|------|
| `_should_save_today` | **Cache persistence** — drop today's row unless NY time ≥ `TODAY_RELOAD_CUTOFF` (avoid partial EOD in cache). |
| `is_available_on_date` | **Availability** — is this date a valid observation date *right now*? (pre-market today → False). |
| L3 `_business_day_grid` | Builds grid through `end_date` inclusive — **should not require today when unavailable**. |
| Point-in-time | 10 BDay lookback; backward search; L1 only on fetch. |
| `option_spot.rt()` | Live QUOTE path; `is_certified=False` by design. |

**Design: cert uses `is_available_on_date`, not cache today policy.**

- [x] **`certification_required_bus_days`** — shared grid for missing-calendar check and L3
      reindex; omits today when `not is_available_on_date(today)` (pre-market).

Do **not** tie the certification grid to whether cache kept today. Cache omits today for
storage hygiene; cert asks “should we require an observation on this date?” via
`is_available_on_date` — pre-market today excluded; during market hours today in-range required.

**Remaining (moved to [FUTURE_CHANGES.md](FUTURE_CHANGES.md)):**

- Vol/greeks pre-market `rt()` NaN noise — largely addressed via `_sync_date` QUOTE pre-market clamp; monitor in prod.
- Operator guidance for `end_date` choice.

---

## Documentation

Moved to `trade/datamanager/docs/`:

| File | Role |
|------|------|
| `README.md` | Module guide |
| `UPGRADE_PLAN.md` | This document |
| `FUTURE_CHANGES.md` | Backlog |

---

### NA logging

- [x] Moved into `certify_manager_result` for certified timeseries (pre-cert forensics)
- [x] Removed `@log_na_after_retrieval` from certified timeseries methods
- [x] Removed from point-in-time / `rt()` paths (timeseries cert owns NA; `resolve_value_at_date` owns fallback)
- [x] Kept `log_model_result_pack_na` in `model.py` for orchestrator-level pack logging

### MarketTimeseries

- [x] Certification at individual `_get_*_timeseries` getters (not `get_timeseries` umbrella)
- [x] `MarketTimeseriesFactorResult` — throwaway `Result` for `Series`/`DataFrame` payloads
- [x] `SpotDataManager` defers to MT (no double certification)
- [x] Sanitize on cache load (see Sanitize section); `_sync_equity_window` on all `_get_*_timeseries` getters

### Tests & observability

- [x] Deep smoke test: `tests/run_certification_deep_test.py` (25 cases)
- [~] Notebook: `notebooks/test_certification.ipynb`
- [ ] Automated pytest unit tests (L1/L2/L3 matrix per check type)
- [~] Certification report (`report.py`); wired via `certification.report` logger (date range + duplicate labels)
- [~] `certification/__init__.py` public API

**Certification logging:** Plain-text audit reports log to
``trade.datamanager.certification.report`` (``logs/trade.datamanager.certification.report*.log``).
Structural warnings and L3 fixes remain on ``certification.pipeline`` / ``respond`` / ``fixers``.
NA forensics stay on ``trade.datamanager.utils.model_na``.

### Deferred / lower priority

- [ ] **Dividend discrete L3 on `Schedule` objects** — cert today assumes Series/DataFrame;
      discrete dividend series holds `Schedule` per day; structural checks may not apply cleanly.
- [ ] Remove stale sparse-series L3 EPIC unless a real failure mode appears in testing.

**Sparse computed-series L3 (mostly theoretical):**

In normal operation, forward / dividend yield / discrete dividends **should be complete** on the
B-day window when upstream inputs are certified:

| Series | Why it is usually dense |
|--------|-------------------------|
| **Forward** | One row per valuation date in `[start, end]`; built from aligned spot + rates + divs. |
| **Dividend yield (MT)** | `dividends / spot` on the same index, then `fillna(0)` — full grid. |
| **Dividend DISCRETE** | `_build_discrete_schedule_series` emits one `Schedule` per business day. |

Gaps appear only when **upstream** data is broken (e.g. uncertified partial spot passed into
forward) — that is a real incompleteness bug, not vendor-confirmed absence. L3 ffill on forward
would mask that bug. Prefer L2 fail or fix upstream; do not special-case sparse L3 unless we
see a production case.

---

## Configuration

### Global default

`OptionDataConfig.certification_level` (default `L2`). All managers read this when the
per-call `certification_level` kwarg is omitted.

### Per-call override (timeseries public APIs)

| Manager | Parameter |
|---------|-----------|
| `RatesDataManager.get_risk_free_rate_timeseries` | `certification_level` |
| `ForwardDataManager.get_forward_timeseries` | `certification_level` |
| `DividendDataManager.get_schedule_timeseries` | `certification_level` |
| `OptionSpotDataManager.get_option_spot_timeseries` | `certification_level` |
| `VolDataManager.get_implied_volatility_timeseries` | `certification_level` |
| `GreekDataManager.get_greeks_timeseries` | `certification_level` |

Point-in-time methods: L1 only via `resolve_value_at_date` lookback fetch.

---

## Certification semantics

### Guarantee levels

| Level | Behavior |
|-------|----------|
| **L1** | Return as-is. Log structural issues (dupes, missing B-days, NaNs). |
| **L2** | Raise `DataNotCertifiedException` on violations not explained by `checked_missing_dates`. |
| **L3** | Fix (ffill, dedupe, reindex to B-day grid); log fixes (with duplicate labels); residual re-check; write to `result.timeseries`. |

### NaN policy (option artifacts)

NaN is allowed **only** on dates in `checked_missing_dates` (vendor-confirmed absent prints).
Gaps are **always resolved from vendor** at cert time — not read from `result` or trusted from
cache alone on full hit.

### Vol / greeks sync with option spot

Vendor `list_dates` gaps are shared for spot, IV, and greeks on the same contract.
Resolved at cert call time via `resolve_checked_missing_dates_for_option_artifact`.

### Synchronization grid

Business days (`freq="B"`) excluding `HOLIDAY_SET`, clipped to `[start, end]`. For options,
`checked_missing_dates` are excluded from the “must have value” set.

### `endpoint_source` on option results

Set on `VolatilityResult`, `GreekResultSet`, and `OptionSpotResult` before certification
so L1 reports include the Thetadata endpoint (EOD vs QUOTE).

---

## Sanitize responsibilities (current)

**In `_data_structure_sanitize`:**

- DatetimeIndex conversion (`to_datetime` via helper)
- tz awareness removal
- `index.name = "datetime"`
- Lowercase DataFrame columns
- Sort index
- Clip to `[start, end]`
- Holiday filter (NYSE proxy)

**Not in sanitize (L3 fixers):**

- Dedupe
- Business-day reindex / NaN pad
- ffill

---

## Resolved design decisions

1. Certify at return boundary after sanitize; cache pre-certify.
2. NA logging pre-cert in `certify_manager_result`.
3. L3 fixes not persisted to cache.
4. `checked_missing_dates` is L2/L3 exception list for option artifacts only; **always vendor-resolved at cert**.
5. **No `checked_missing_dates` on manager results** — vendor resolve at cert; cache metadata only.
6. Vol/greeks gaps follow option spot vendor calendar (same contract).
7. MT cert at getter level; `SpotDataManager` / dividend CONTINUOUS defer to MT.
8. `certification_level` global on config; per-call override on timeseries APIs.
9. `DataNotCertifiedException` at L2.
10. Rates partial cache uses gap bounds from cache check, not full request window.
11. Point-in-time: shared `resolve_value_at_date`, 10 BDay lookback, L1 on fetch.
12. No unified `TimeseriesPullPipeline` — add sanitize to MT cache loads instead.

---

## References

- `trade/datamanager/result.py` — `Result.is_certified`
- `trade/datamanager/certification/integration.py` — `certify_manager_result`
- `trade/datamanager/certification/market_timeseries.py` — MT adapter
- `trade/datamanager/utils/cache.py` — `_CachedData`, `_should_save_today`, `_get_checked_missing_dates_from_cache`
- `trade/datamanager/utils/vol_helpers.py` — vol/greek cert + gap resolution
- `trade/datamanager/utils/data_structure.py` — `_data_structure_sanitize`
- `trade/datamanager/utils/classification.py` — `classify_option_spot_dates`, vendor `list_dates`
- `trade/datamanager/utils/point_in_time.py` — `resolve_value_at_date`
- `trade/datamanager/requests.py` — `LoadRequest._is_missing_dates`
- `trade/datamanager/config.py` — `OptionDataConfig.certification_level`
- `trade/datamanager/tests/run_certification_deep_test.py` — integration smoke test
