# `trade.features`

Named research features with process L1 memory and CustomCache L2 persistence, orchestrated by `FeatureDataManager`.

```text
get_market_features(...)          # market-only
get_features_for_ticker(ticker)   # self-queried vendor spot → L1+L2
get_features_from_ohlcv(t, ohlcv) # caller bars → L1 only (bars=custom)

  → L1 / (L2 if durable) hit? slice and return
  → else load inputs → FeatureContext
  → FEATURE_SPECS[name].compute(ctx)
  → strip edge NaNs → write cache → slice to index
```

## Quick use

```python
from trade.features import FeatureDataManager, FeatureTheme, list_features_by_theme

mgr = FeatureDataManager()

## Market-only
out = mgr.get_market_features(index=df.index)
out = mgr.get_market_features(features=["spy_ret_20d", "vix_level"], index=df.index)

## Self-queried asset spot (datamanager); durable L1+L2
out = mgr.get_features_for_ticker("AAPL", features=["ret_20d"], index=df.index)

## Caller OHLCV — both ticker and ohlcv required; L1-only, not written to disk
out = mgr.get_features_from_ohlcv("AAPL", aapl_ohlcv, features=["ret_20d"], index=df.index)

## Theme helpers
from trade.features import FEATURE_THEMES, list_features_by_theme
names = FEATURE_THEMES["equity_market_momentum"]
names = list_features_by_theme(FeatureTheme.EQUITY_MARKET_MOMENTUM)
```

`FeatureResult.series` is a `dict[str, pd.Series]`.

## Layout

| Path | Role |
|---|---|
| `manager.py` | L1 → L2 → compute orchestration |
| `inputs.py` | Market OHLC / FRED IV pulls (process cache; `seed_*` for tests) |
| `catalog.py` | Public name resolution over the registry |
| `themes.py` | `FeatureTheme` enum + membership tables |
| `compute/specs.py` | Durable `FEATURE_SPECS`, `Feature`, base packs |
| `compute/registry.py` | `@feature` decorator and lookup helpers |
| `compute/context.py` | `FeatureContext` input bag (`close`, `iv`, `asset_close`, …) |
| `compute/transforms.py` | Shared pure math helpers |
| `compute/symbols.py` | Market symbol map and FRED IV keys |
| `compute/market/*.py` | Market feature families |
| `compute/asset/*.py` | Asset-local feature families |
| `compute/hybrid/*.py` | Hybrid (asset + market/IV) families |
| `compute/families.py` | Imports all families so `@feature` registers |

Trail for any name: **feature name → `FEATURE_SPECS[name]` → function body in its family file**.

## Scopes

| Scope | Needs | Default when |
|---|---|---|
| `market` | Market symbols / IV only | `get_market_features` default; also included on ticker/OHLCV endpoints |
| `hybrid` | Asset bars + market/IV | Default on `get_features_for_ticker` / `get_features_from_ohlcv` |
| `asset` | Asset bars (`close` / `ohlcv`) | Same as hybrid |

Use the ticker or OHLCV endpoint for non-market features. An explicit `features=` list still wins as a subset.

## How to add a new feature

### 1. Pick scope and family file

- Market-only (SPY, VIX, ratios, …) → `compute/market/<family>.py`
- Asset bars only → `compute/asset/<family>.py`
- Asset + market/IV → `compute/hybrid/<family>.py`

Reuse an existing family file when the idea fits. Create a new module only for a new family, then import it from that package’s `__init__.py` (see `compute/market/__init__.py` / `compute/asset/__init__.py`). `load_families()` already imports those packages.

### 2. Declare inputs the manager must load

Use keys the context understands:

- Market close: `"SPY"`, `"VIX"`, `"UUP"`, … (must exist in `compute/symbols.py` → `MARKET_SYMBOLS`)
- FRED IV: `"FXI_IV"`, `"EEM_IV"` (must exist in `FRED_IV_SERIES`)
- Asset: `"close"` and/or `"ohlcv"`

If you need a **new market symbol or IV series**, add it to `compute/symbols.py` first.

### 3. Implement with `@feature`

Function name **is** the public feature name and cache key.

```python
from trade.features.compute.registry import feature


@feature(
    scope="market",
    formula_id="close_return_v1",
    inputs=("SPY",),
    horizon=20,  # optional int params for introspection
)
def spy_ret_20d(ctx) -> pd.Series:
    return ctx.close("SPY").pct_change(20)
```

Asset example:

```python
@feature(scope="asset", formula_id="asset_close_return_v1", inputs=("close",), horizon=20)
def ret_20d(ctx) -> pd.Series:
    return ctx.asset_close().pct_change(20)
```

Rules:

- Return a `pd.Series` aligned to the compute calendar (same index as the closes / ohlcv the context was built with).
- Prefer helpers in `compute/transforms.py` over duplicating rolling math.
- `formula_id` is embedded in the L2 cache key. **Bump it when the math changes** so stale cache rows are not reused (`close_return_v1` → `close_return_v2`).
- Theme is required via `FEATURE_THEMES` (see next step). An optional `theme=` on `@feature` must match that table; there is no silent `unmapped` default.

### 4. Register the research theme

Add the feature name under the right bucket in `themes.py` → `FEATURE_THEMES` (keys are theme string values, e.g. `"equity_market_momentum"`).

If the feature is intentionally without a research theme, list it under `FEATURE_THEMES["unmapped"]`. Missing membership fails at import / `load_families()`.

If this is a **new theme**, add a member to `FeatureTheme` first, then a `FEATURE_THEMES` entry.

Optionally update the legacy notebook views `ASSET_FEATURE_THEMES` / `REGIME_FEATURE_THEMES` if notebooks still use those split tables.

`load_families()` runs `validate_registry()` once: every registered name must be in exactly one theme bucket, inputs must resolve (`MARKET_SYMBOLS` / `FRED_IV_SERIES` / `close`/`ohlcv`), and scope must match those inputs.

### 5. Smoke-check registration

```python
from trade.features import get_spec, list_features, FeatureTheme

spec = get_spec("spy_ret_20d")
assert spec.scope == "market"
assert spec.theme is FeatureTheme.EQUITY_MARKET_MOMENTUM
assert "spy_ret_20d" in list_features("market")
```

### 6. Compute once (tests or notebook)

Unit tests seed inputs so vendors are not hit:

```python
from trade.features import FeatureDataManager

mgr = FeatureDataManager()
mgr.inputs.seed_ohlc({"SPY": spy_ohlc_frame})
result = mgr.get_market_features(features=["spy_ret_20d"], start="2020-01-01", end="2020-06-01")
```

For asset / hybrid features in tests, prefer `seed_asset_ohlc` + `get_features_for_ticker`, or pass a frame to `get_features_from_ohlcv` (and seed any market/IV inputs the feature declares).

Add or extend a case in `tests/test_feature_data_manager.py` when the feature changes manager behavior (new scope rules, new required inputs, etc.). Pure formula tweaks can stay as a small seeded compute check.

### 7. Cache note after math changes

Changing only the function body without bumping `formula_id` will keep serving old L2 values. Bump `formula_id`, or call `get_market_features(..., recompute=True)` while iterating.

## How to deprecate a feature

Soft-retire in place (no dedicated deprecated module):

1. Move the feature name from its research theme into `FEATURE_THEMES["deprecated"]` in `themes.py`.
2. Leave the `@feature` function in its family file. Registration derives `deprecated=True` from that theme (optional explicit `deprecated=True` must agree).
3. Remove the name from legacy `ASSET_FEATURE_THEMES` / `REGIME_FEATURE_THEMES` if present.
4. Defaults (`features=None` on market / ticker / ohlcv endpoints) **omit** deprecated names.
5. Explicit `features=["old_name"]` still works and emits a `DeprecationWarning`.
6. Hard-delete later: remove the `@feature` function, drop the name from `FEATURE_THEMES["deprecated"]`, and clean related tests.

## Checklist (copy/paste)

1. [ ] Choose `market` / `asset` / `hybrid` and the family module (import new modules from the package `__init__.py`).
2. [ ] Ensure symbols / IV keys exist in `compute/symbols.py`.
3. [ ] Add `@feature(...)` function; name = public API name.
4. [ ] Set `formula_id` (bump on math changes).
5. [ ] Declare accurate `inputs=`.
6. [ ] Add name to `FEATURE_THEMES` (or `unmapped` / `deprecated`; add `FeatureTheme` if new theme).
7. [ ] Import / `load_families()` must pass `validate_registry` with no errors.
8. [ ] Smoke: `get_spec(name)` and a seeded `get_market_features` call.
9. [ ] Extend unit tests if orchestration behavior changed.

## Related

- Design notes: [`PLAN.md`](PLAN.md)
- Stop-loss notebook re-exports: `strategy_research.resources.strategies.base.features.themes`
