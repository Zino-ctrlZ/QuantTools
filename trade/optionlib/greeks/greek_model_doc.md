# Greek model documentation

Units, bump conventions, and calculation paths for option greeks in `trade.optionlib`,
`GreekDataManager`, and PnL attribution (`xmultiply_attr`).

**Live default:** `GreekDataManager` → **binomial CRR** (`OptionPricingModel.BINOMIAL`).
Analytic BS and FD numerical BSM are alternate engines that emit the **same
Price-Sensitivity Units**, so attribution formulas do not depend on which engine
produced the greek.

---

## Shared vocabulary

| Term | Meaning |
|---|---|
| σ / `sigma` (in market data / DoD) | Annualized vol as a **decimal** (e.g. `0.20` = 20%). Rate `r` is also decimal. |
| 1 vol point (1 percentage point) | 20% → 21%, i.e. Δσ = `0.01` decimal. **Not** 1 basis point (`0.0001` decimal). |
| Price-Sensitivity Unit | The unit of the greek in the timeseries / cache. That **is** the greek’s meaning. |

**Attribution note:** DoD `vol_change` / `rates_change` arrive as **decimal**. Where a greek’s
Price-Sensitivity Unit is per 1 point, PnL converts the move with `× 100`
(points = decimal × 100). That is move bookkeeping only; it does not redefine the greek.

---

## Bump policy (greek estimation)

Bump sizes are **getter-only** from YAML (`get_multiplicative_bump_size`,
`get_additive_bump_size`, `get_theta_bump_size`). Policy lives in
`resolve_greek_bump` → `normalize_greek_factor` (`GreekFactor` / `GreekBumpMode`
in `config/types.py`, plus alias map for `S0` / `spot` / `vol` / …).

| Factor | Mode | Config key | Absolute step |
|---|---|---|---|
| Spot `S` | Multiplicative | `MULTIPLICATIVE_BUMP_SIZE` | `\|S\| * ε` |
| Vol `σ`, rate `r` | Additive | `ADDITIVE_BUMP_SIZE` | `ε` (fixed, on decimal) |
| Time `T` | Dedicated | `THETA_BUMP_SIZE` | `ε` years (~1/`DAILY_BASIS`) |

**General FD vs greek FD**

- `finite_diff_*_vec` — general primitives; caller passes an **absolute** `dx` (no config).
- `FiniteGreeksEstimator` — greek estimation; calls `resolve_greek_bump` then the primitives.
- Binomial `_tree_numerical*` / `vanna` — same `resolve_greek_bump` per attribute.

---

## Price-Sensitivity Units (cheat sheet)

| Greek | Exposure | Price-Sensitivity Unit (what the number means) | Attribution (unit contract) |
|---|---|---|---|
| **delta** | Spot | Per **\$1** spot | `ΔS * delta` |
| **gamma** | Spot | Per **(\$1)²** spot | `0.5 * ΔS² * gamma` |
| **vega** | Vol | Per **1 vol point** | `Δσ_decimal * vega * 100` |
| **volga** | Vol | Per **(1 vol point)²** | `0.5 * (Δσ_decimal * 100)² * volga` |
| **vanna** | Spot × vol | Per **\$1 spot and 1 vol point** (cross) | `ΔS * Δσ_decimal * vanna * 100` |
| **theta** | Time | Per **1 calendar day** | `Δdays * theta` |
| **rho** | Rate | Per **1 rate point** (e.g. 5% → 6%) | `Δr_decimal * rho * 100` |

Example: vega = `0.25` means about \$0.25 premium for vol 20% → 21%, not for 20.00% → 20.01%.

---

## Per-greek detail

### Delta

| | |
|---|---|
| **Exposure to** | Underlying spot S |
| **Price-Sensitivity Unit** | Per **\$1** spot move |
| **Analytic** | Call: `N(d1)`; put: `-N(-d1)` (forward BS; `black_scholes_math`) |
| **Numerical (FD)** | Central first difference of price w.r.t. `S` |
| **Binomial** | Tree: `(V_up - V_down) / (S_up - S_down)` at first step (**no bump**) |
| **Bump size** | Greek FD: multiplicative on `S`. Binomial: N/A (tree delta) |
| **Attribution** | `delta_pnl = ΔS * delta` |

### Gamma

| | |
|---|---|
| **Exposure to** | Spot S (convexity) |
| **Price-Sensitivity Unit** | Per **(\$1)²** spot |
| **Analytic** | `n(d1) / (F σ √T)` |
| **Numerical (FD)** | Central second difference of price w.r.t. `S` |
| **Binomial** | Second difference on tree nodes (**no bump**) |
| **Bump size** | Greek FD: multiplicative on `S`. Binomial: N/A (tree gamma) |
| **Attribution** | `gamma_pnl = 0.5 * ΔS² * gamma` |

### Vega

| | |
|---|---|
| **Exposure to** | Implied vol |
| **Price-Sensitivity Unit** | Per **1 vol point** (20% → 21%) |
| **Analytic** | Closed-form ∂V/∂σ, expressed per 1 vol point |
| **Numerical (FD)** | Central price difference w.r.t. `sigma`, expressed per 1 vol point |
| **Binomial** | Central price bump on `sigma`, expressed per 1 vol point |
| **Bump size** | Greek FD and binomial: additive on `σ` |
| **Attribution** | `vega_pnl = Δσ_decimal * vega * 100` |

### Volga (vomma)

| | |
|---|---|
| **Exposure to** | Vol (convexity) |
| **Price-Sensitivity Unit** | Per **(1 vol point)²** |
| **Analytic** | Closed-form ∂²V/∂σ², expressed per (1 vol point)² |
| **Numerical (FD)** | Central second difference w.r.t. `sigma`, expressed per (1 vol point)² |
| **Binomial** | Central second price bump on `sigma`, expressed per (1 vol point)² |
| **Bump size** | Same additive σ bump as vega |
| **Attribution** | `volga_pnl = 0.5 * (Δσ_decimal * 100)² * volga` |

### Vanna

| | |
|---|---|
| **Exposure to** | Spot **and** vol (cross) |
| **Price-Sensitivity Unit** | Per **\$1 spot and 1 vol point** together |
| **Analytic** | `-e^{-rT} n(d1) d2 / σ`, expressed in the Price-Sensitivity Unit above |
| **Numerical (FD)** | Mixed central FD on `(S, σ)`, expressed in the Price-Sensitivity Unit |
| **Binomial** | **Vol-bump of tree delta** (not price-cross FD — CRR cross FD is ill-conditioned), expressed in the Price-Sensitivity Unit |
| **Bump size** | Greek FD: multiplicative on `S`, additive on `σ`. Binomial: additive on `σ` for the delta bump |
| **Attribution** | `vanna_pnl = ΔS * Δσ_decimal * vanna * 100` |

### Theta

| | |
|---|---|
| **Exposure to** | Calendar time |
| **Price-Sensitivity Unit** | Per **1 calendar day** (long options typically negative) |
| **Analytic** | Closed-form year derivative converted to per day via `DAILY_BASIS` |
| **Numerical (FD)** | `-∂V/∂T / DAILY_BASIS` with `dx = THETA_BUMP_SIZE` years |
| **Binomial** | `-_tree_numerical("T") / DAILY_BASIS` with the same theta bump |
| **Bump size** | Dedicated `THETA_BUMP_SIZE` (years) for greek FD and binomial |
| **Attribution** | `theta_pnl = Δcalendar_days * theta` |

### Rho

| | |
|---|---|
| **Exposure to** | Risk-free rate |
| **Price-Sensitivity Unit** | Per **1 rate point** (5% → 6%) |
| **Analytic** | Closed-form, expressed per 1 rate point |
| **Numerical (FD)** | Central price difference w.r.t. `r`, expressed per 1 rate point |
| **Binomial** | Central price bump on `r`, expressed per 1 rate point |
| **Bump size** | Greek FD and binomial: additive on `r` |
| **Attribution** | `rho_pnl = Δr_decimal * rho * 100` |

---

## Calculation engines

| Engine | Entry points | Notes |
|---|---|---|
| **Analytic BS** | `trade/optionlib/core/black_scholes_math.py` | Forward Black–Scholes; European. |
| **Numerical FD (BS)** | `finite_diff.py` primitives + `FiniteGreeksEstimator`; `numerical/black_scholes.py` | Patched BSM reprice; `option_model=BSM`. Bumps via `resolve_greek_bump`. |
| **Binomial CRR** | `trade/optionlib/pricing/binomial.py`, `greeks/numerical/binomial.py` | Live default; American + discrete/continuous dividends. Same bump resolver. |

`GreekDataManager` (`trade/datamanager/greeks.py`) loads spot / rates / dividends / IV, runs one engine, caches the greek frame in Price-Sensitivity Units, and returns requested `GreekType` columns (default set includes `VANNA`).

---

## PnL attribution wiring

Module: `trade/assets/calculate/xmultiply_attr.py`.

1. Greeks are **shifted forward one session** so prior-session greeks multiply today’s DoD changes.
2. Vol/rate DoD inputs are **decimal**; `× 100` appears only where the Price-Sensitivity Unit is per point.
3. Incomplete greeks are `fillna(0)` before totaling.
4. `opt_spot` may appear on the frame for diagnostics; live DB insert strips unknown columns.

---

## Consistency rules (do not break)

1. All engines use the same Price-Sensitivity Units: vega / vanna / rho per 1 point; volga per (1 point)²; delta per \$1; gamma per (\$1)²; theta per day.
2. Binomial **vanna** uses `∂delta/∂σ`, not `_tree_numerical_cross` on price.
3. Volga PnL uses `(Δσ_decimal * 100)²`, not `Δσ_decimal² * 100`.
4. **1 vol point ≠ 1 bp.**
5. New default greek columns force cache recompute when missing (`_missing_greek_columns` in `datamanager/utils/greeks_helpers.py`).
6. Greek bumps go through `resolve_greek_bump` (or absolute override); general `finite_diff_*_vec` never reads bump config.

---

## Quick numeric check

For the same European inputs, vega / vanna / volga Price-Sensitivity Units from analytic, FD, and binomial should agree within a few percent (tree depth and FD bump size drive the gap). A ~100× disagreement usually means an engine or attribution formula is using the wrong move unit (decimal vs points), not a different greek definition.
