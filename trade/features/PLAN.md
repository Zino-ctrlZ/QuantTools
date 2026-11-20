# FeatureDataManager

## Core loop

```text
get_market_features(names, index=..., start=1970, end=last_busday)
  → L1 / L2 hit? slice and return
  → else load inputs → FeatureContext
  → FEATURE_SPECS[name].compute(ctx)
  → write L2 + L1 → slice to index
```

## Where formulas live

```text
compute/market/*.py   market features (@feature functions)
compute/asset/*.py    asset features
compute/hybrid/*.py   hybrid features
compute/transforms.py shared pure math
compute/registry.py   @feature + FEATURE_SPECS
compute/context.py    FeatureContext input bag
```

Trail: feature name → registry row → function body in its family file.
