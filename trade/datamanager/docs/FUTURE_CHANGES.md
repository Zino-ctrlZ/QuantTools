# DataManager — Future Changes

Backlog items deferred after the certification + point-in-time rollout. Not blocking production use.

Legend: priority **high** · **medium** · **low**

---

## Testing

| Priority | Item | Notes |
|----------|------|-------|
| medium | **pytest cert matrix** | Unit tests per check (empty, dupes, NaNs, calendar gaps) and per level (L1/L2/L3). Harness: `tests/run_certification_deep_test.py`. |
| low | **Notebook harness** | Align `notebooks/test_certification.ipynb` with deep smoke test (needs explicit approval per notebook-safety). |

---

## Date / listing bounds

| Priority | Item | Notes |
|----------|------|-------|
| medium | **IPO listing date floor** | `_sync_equity_date(..., symbol=...)` — resolve per-symbol listing date and pass as `min_trade_date`. Do not use `OPTION_TIMESERIES_START_DATE` (options vendor bound only). |

---

## Certification

| Priority | Item | Notes |
|----------|------|-------|
| low | **Strict artifact gating** | Honor `checked_missing_dates` kwarg only when `artifact_type` is option-related (`OPTION_ARTIFACT_TYPES`). |
| low | **Dividend DISCRETE L3 on `Schedule`** | Structural checks assume Series/DataFrame; discrete dividend series holds `Schedule` per day. |
| low | **`result.checked_missing_dates`** | Removed by design; revisit only if a concrete consumer needs it on the return object. |

---

## Observability / ops

| Priority | Item | Notes |
|----------|------|-------|
| low | **Operator guidance** | When to use `end_date=yesterday` vs today; interaction with `certification_level` and pre-market. |

---

## References

- [UPGRADE_PLAN.md](UPGRADE_PLAN.md) — completed rollout details
- [README.md](README.md) — module guide
