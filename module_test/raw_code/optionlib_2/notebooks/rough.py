import pandas as pd
import numpy as np
from math import sqrt, erf, pi

# ---- helpers ----
def _Nprime(x):
    return np.exp(-0.5 * x * x) / sqrt(2.0 * pi)

def _black_vega(F, K, T, sigma, D=1.0):
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0 or D <= 0:
        return 0.0
    v = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    return D * F * np.sqrt(T) * _Nprime(d1)

# ---- main function ----
def prepare_ssvi_input(df: pd.DataFrame, k_blend: float = 0.20) -> pd.DataFrame:
    """
    Build SSVI-ready vols from a filtered option chain with columns:
    'strike','right','log_moneyness','t','f','european_vols_equiv'
    Optionally: 'closebid','closeask' for spread weighting.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered chain for a single expiry.
    k_blend : float, default=0.20
        Width of ATM blend zone in log-moneyness.

    Returns
    -------
    pd.DataFrame
        Columns: k, merged_iv, t, w
    """
    # relative spread (optional)
    if 'closebid' in df.columns and 'closeask' in df.columns:
        mid = (pd.to_numeric(df['closebid'], errors='coerce') +
               pd.to_numeric(df['closeask'], errors='coerce')) / 2.0
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_spread = (pd.to_numeric(df['closeask'], errors='coerce') -
                          pd.to_numeric(df['closebid'], errors='coerce')) / mid.replace(0, np.nan)
    else:
        rel_spread = pd.Series(np.nan, index=df.index)

    # working DataFrame
    work = pd.DataFrame({
        'strike': pd.to_numeric(df['strike'], errors='coerce'),
        'right': df['right'].astype(str).str.upper().str[0],
        'k': pd.to_numeric(df['log_moneyness'], errors='coerce'),
        't': pd.to_numeric(df['t'], errors='coerce'),
        'f': pd.to_numeric(df['f'], errors='coerce'),
        'iv': pd.to_numeric(df['european_vols_equiv'], errors='coerce'),
        'rel_spread': rel_spread
    }).dropna(subset=['strike','right','k','t','f','iv']).copy()

    # vega for weights
    work['vega'] = [_black_vega(F, K, T, iv, 1.0)
                    for F, K, T, iv in work[['f','strike','t','iv']].to_numpy()]

    # split calls/puts
    calls = work[work['right'] == 'C'].copy()
    puts  = work[work['right'] == 'P'].copy()

    # outer merge by strike
    m = pd.merge(
        calls[['strike','k','t','f','iv','vega','rel_spread']]
             .rename(columns={'iv':'iv_c','vega':'vega_c','rel_spread':'rel_c','k':'k_c'}),
        puts [['strike','k','t','f','iv','vega','rel_spread']]
             .rename(columns={'iv':'iv_p','vega':'vega_p','rel_spread':'rel_p','k':'k_p'}),
        on=['strike','t','f'], how='outer'
    )

    m['k'] = m['k_c'].combine_first(m['k_p'])
    for c in ['iv_c','iv_p','vega_c','vega_p','rel_c','rel_p']:
        m[c] = pd.to_numeric(m[c], errors='coerce')

    m['rel_c'] = m['rel_c'].fillna(m['rel_c'].median() if m['rel_c'].notna().any() else 0.0)
    m['rel_p'] = m['rel_p'].fillna(m['rel_p'].median() if m['rel_p'].notna().any() else 0.0)
    m['vega_c'] = m['vega_c'].fillna(0.0)
    m['vega_p'] = m['vega_p'].fillna(0.0)

    # blend logic
    merged_iv = []
    for _, row in m.iterrows():
        k = row['k']
        ivc, ivp = row['iv_c'], row['iv_p']
        vc, vp   = row['vega_c'], row['vega_p']
        rc, rp   = max(row['rel_c'], 0.0) if pd.notna(row['rel_c']) else 0.0, \
                   max(row['rel_p'], 0.0) if pd.notna(row['rel_p']) else 0.0

        if not np.isfinite(k):
            merged_iv.append(np.nan)
            continue

        if abs(k) <= k_blend:
            have_c, have_p = np.isfinite(ivc), np.isfinite(ivp)
            if have_c and have_p:
                w_c = vc / (1.0 + rc)
                w_p = vp / (1.0 + rp)
                if (w_c + w_p) <= 0:
                    ivm = np.nanmean([ivc, ivp])
                else:
                    ivm = (w_c * ivc + w_p * ivp) / (w_c + w_p)
                merged_iv.append(ivm)
            elif have_c:
                merged_iv.append(ivc)
            elif have_p:
                merged_iv.append(ivp)
            else:
                merged_iv.append(np.nan)
        elif k > k_blend:
            merged_iv.append(ivc if np.isfinite(ivc) else ivp)
        else:  # k < -k_blend
            merged_iv.append(ivp if np.isfinite(ivp) else ivc)

    out = m.copy()
    out['merged_iv'] = merged_iv
    out = out.dropna(subset=['merged_iv','k','t']).copy()

    # build SSVI input
    ssvi_input = out[['k','merged_iv','t']].sort_values('k').reset_index(drop=True)
    ssvi_input['w'] = (ssvi_input['merged_iv'] ** 2) * ssvi_input['t']
    return ssvi_input




import numpy as np
import pandas as pd

# ---- utilities ---------------------------------------------------------------

def _compute_rel_spread(df: pd.DataFrame) -> pd.Series:
    """
    Compute relative spread = (ask - bid) / mid, or NaN if not available.
    """
    if 'closebid' in df.columns and 'closeask' in df.columns:
        bid = pd.to_numeric(df['closebid'], errors='coerce')
        ask = pd.to_numeric(df['closeask'], errors='coerce')
        mid = (bid + ask) / 2.0
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = (ask - bid) / mid.replace(0, np.nan)
        return rel
    return pd.Series(np.nan, index=df.index)


def _build_working_frame(df: pd.DataFrame, rel_spread: pd.Series) -> pd.DataFrame:
    """
    Coerce types, standardize columns, and drop non-finite essentials.
    """
    work = pd.DataFrame({
        'strike'    : pd.to_numeric(df['strike'], errors='coerce'),
        'right'     : df['right'].astype(str).str.upper().str[0],
        'k'         : pd.to_numeric(df['log_moneyness'], errors='coerce'),
        't'         : pd.to_numeric(df['t'], errors='coerce'),
        'f'         : pd.to_numeric(df['f'], errors='coerce'),
        'iv'        : pd.to_numeric(df['european_vols_equiv'], errors='coerce'),
        'rel_spread': rel_spread
    })
    # Require finite strike/right/k/t/f/iv
    work = work.dropna(subset=['strike', 'right', 'k', 't', 'f', 'iv']).copy()
    return work


def _attach_vega(work: pd.DataFrame, black_vega_func) -> pd.DataFrame:
    """
    Attach vega per quote using provided Black vega function:
    black_vega_func(F, K, T, iv, notional)
    """
    arr = work[['f', 'strike', 't', 'iv']].to_numpy()
    work['vega'] = [black_vega_func(F, K, T, iv, 1.0) for F, K, T, iv in arr]
    return work


def _split_and_outer_merge(work: pd.DataFrame) -> pd.DataFrame:
    """
    Split calls/puts and outer-merge by (strike, t, f). Carry both sides' fields.
    """
    calls = work[work['right'] == 'C'][['strike', 'k', 't', 'f', 'iv', 'vega', 'rel_spread']].copy()
    puts  = work[work['right'] == 'P'][['strike', 'k', 't', 'f', 'iv', 'vega', 'rel_spread']].copy()

    calls = calls.rename(columns={'iv':'iv_c', 'vega':'vega_c', 'rel_spread':'rel_c', 'k':'k_c'})
    puts  = puts .rename(columns={'iv':'iv_p', 'vega':'vega_p', 'rel_spread':'rel_p', 'k':'k_p'})

    m = pd.merge(calls, puts, on=['strike', 't', 'f'], how='outer')
    m['k'] = m['k_c'].combine_first(m['k_p'])

    # Ensure numeric types and clean NaNs
    for c in ['iv_c','iv_p','vega_c','vega_p','rel_c','rel_p', 'k', 't', 'f']:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors='coerce')

    return m


def _normalize_weights(m: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare weight inputs:
      - rel_c/rel_p: fill NaN with column median (or 0 if empty)
      - vega_c/vega_p: fill NaN with 0
    """
    if 'rel_c' in m.columns:
        m['rel_c'] = m['rel_c'].fillna(m['rel_c'].median() if m['rel_c'].notna().any() else 0.0)
    if 'rel_p' in m.columns:
        m['rel_p'] = m['rel_p'].fillna(m['rel_p'].median() if m['rel_p'].notna().any() else 0.0)
    if 'vega_c' in m.columns:
        m['vega_c'] = m['vega_c'].fillna(0.0)
    if 'vega_p' in m.columns:
        m['vega_p'] = m['vega_p'].fillna(0.0)
    return m


def _blend_iv_row(k: float,
                  ivc: float, ivp: float,
                  vc: float,  vp: float,
                  rc: float,  rp: float,
                  k_blend: float) -> float | np.floating:
    """
    Row-wise blending logic:
      - |k| <= k_blend: vega/(1+rel_spread) weighted blend if both exist.
      - k >  k_blend : prefer calls, fallback to puts.
      - k < -k_blend : prefer puts,  fallback to calls.
    """
    if not np.isfinite(k):
        return np.nan

    have_c = np.isfinite(ivc)
    have_p = np.isfinite(ivp)

    # Clamp spreads to >= 0
    rc = max(rc, 0.0) if np.isfinite(rc) else 0.0
    rp = max(rp, 0.0) if np.isfinite(rp) else 0.0

    if abs(k) <= k_blend:
        if have_c and have_p:
            w_c = (vc if np.isfinite(vc) else 0.0) / (1.0 + rc)
            w_p = (vp if np.isfinite(vp) else 0.0) / (1.0 + rp)
            w_sum = w_c + w_p
            if w_sum <= 0:
                return float(np.nanmean([ivc, ivp]))
            return (w_c * ivc + w_p * ivp) / w_sum
        elif have_c:
            return ivc
        elif have_p:
            return ivp
        return np.nan

    # Outside blend zone: natural side first, fallback to the other.
    if k > k_blend:
        return ivc if have_c else (ivp if have_p else np.nan)
    else:  # k < -k_blend
        return ivp if have_p else (ivc if have_c else np.nan)


def _apply_blend(m: pd.DataFrame, k_blend: float) -> pd.Series:
    """
    Vectorized-ish application of the row-wise blend (via iterrows for clarity).
    Returns a Series of merged_iv.
    """
    merged_iv = []
    for _, row in m.iterrows():
        k   = row.get('k', np.nan)
        ivc = row.get('iv_c', np.nan)
        ivp = row.get('iv_p', np.nan)
        vc  = row.get('vega_c', np.nan)
        vp  = row.get('vega_p', np.nan)
        rc  = row.get('rel_c', np.nan)
        rp  = row.get('rel_p', np.nan)
        merged_iv.append(_blend_iv_row(k, ivc, ivp, vc, vp, rc, rp, k_blend))
    return pd.Series(merged_iv, index=m.index, name='merged_iv')


def _assemble_ssvi_input(m: pd.DataFrame) -> pd.DataFrame:
    """
    Pick (k, merged_iv, t), sort by k, and compute w = iv^2 * t.
    """
    out = m.dropna(subset=['merged_iv', 'k', 't']).copy()
    ssvi_input = out[['k', 'merged_iv', 't']].sort_values('k').reset_index(drop=True)
    ssvi_input['w'] = (ssvi_input['merged_iv'] ** 2) * ssvi_input['t']
    return ssvi_input


# ---- main function -----------------------------------------------------------

def prepare_ssvi_input(
    df: pd.DataFrame,
    k_blend: float = 0.20,
    *,
    black_vega_func=None
) -> pd.DataFrame:
    """
    Build SSVI-ready vols from a filtered option chain with columns:
      'strike','right','log_moneyness','t','f','european_vols_equiv'
    Optionally: 'closebid','closeask' for spread weighting.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered chain for a single expiry.
    k_blend : float, default=0.20
        Half-width of ATM blend zone in log-moneyness.
    black_vega_func : callable, optional
        Function signature: black_vega_func(F, K, T, iv, notional).
        If None, expects a global `_black_vega` to exist.

    Returns
    -------
    pd.DataFrame
        Columns: ['k', 'merged_iv', 't', 'w'] sorted by k.
    """
    # 1) optional spread penalty
    rel_spread = _compute_rel_spread(df)

    # 2) normalized working frame
    work = _build_working_frame(df, rel_spread)

    # 3) attach vega
    if black_vega_func is None:
        # fall back to global if user keeps previous setup
        black_vega_func = globals().get('_black_vega')
        if black_vega_func is None:
            raise NameError("No 'black_vega_func' supplied and global '_black_vega' not found.")
    work = _attach_vega(work, black_vega_func)

    # 4) calls/puts merge
    m = _split_and_outer_merge(work)

    # 5) weight normalization
    m = _normalize_weights(m)

    # 6) blend
    m['merged_iv'] = _apply_blend(m, k_blend)

    # 7) assemble SSVI inputs
    return _assemble_ssvi_input(m)
