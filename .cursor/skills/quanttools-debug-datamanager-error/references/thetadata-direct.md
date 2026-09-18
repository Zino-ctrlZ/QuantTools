# ThetaData direct URL probes

Use when the reproduced datamanager failure implicates the ThetaData API (`ThetaDataNotFound`, empty vendor payload, HTTP errors). Hit the terminal **through the URL**, not through `dbase.DataAPI.ThetaData` helpers.

Default local terminal: `http://127.0.0.1:25510`.

## Extract the URL the wrapper would call

`dbase` is not in the QuantTools tree. Locate it, then search for URL construction:

```python
import inspect
from dbase.DataAPI import ThetaData
print(inspect.getfile(ThetaData))
```

Prefer copying `url` / query dict from the wrapper (some paths log `url=`). Honor `THETADATA_USE_V3`: if set, probe the v3 path the wrapper uses, not a guessed v2 URL.

Record the **inner** `status_code` from the proxy JSON (not the proxy HTTP status). Name it with the mapping in SKILL.md (472 → `ThetaDataNotFound`, 404 → `ThetDataNoImplementation`, etc.).

## Parameter encoding (v2 hist)

| Python kwarg | Query param | Encoding |
|---|---|---|
| `symbol` | `root` | ticker |
| `expiration` / `exp` | `exp` | `YYYYMMDD` |
| `strike` | `strike` | integer **dollars × 1000** (`220.0` → `220000`) |
| `right` | `right` | `C` or `P` |
| `start_date` / `end_date` | `start_date` / `end_date` | `YYYYMMDD` |
| CSV | `use_csv` | `true` |

Repo example (option EOD):

`http://127.0.0.1:25510/v2/hist/option/eod?end_date=20250619&root=AAPL&use_csv=true&exp=20241220&right=C&start_date=20240101&strike=220000`

Quote hist (intraday; extra `ivl`, `start_time`, `end_time`, `rth`):

`http://127.0.0.1:25510/v2/hist/option/quote?root=MSFT&exp=20240621&strike=355000&right=C&start_date=20230706&end_date=20230706&use_csv=true&ivl=1800000&start_time=34200000&rth=False&end_time=57600000`

## Common v2 paths

| Need | Path |
|---|---|
| Option EOD OHLC | `/v2/hist/option/eod` |
| Option quotes | `/v2/hist/option/quote` |
| Underlying EOD | `/v2/hist/stock/eod` (confirm in `dbase` before assuming) |
| List dates / contracts | whatever `list_dates` / `list_contracts` builds — copy from source |

Terminal liveness:

```bash
curl -sS -D - "http://127.0.0.1:25510/v2/system/mmdb/status"
```

If that fails, do not treat empty manager results as a QuantTools logic bug.

## Direct call

```bash
curl -sS -D - -o .sandbox/datamanager-debug/theta_body.txt \
  "http://127.0.0.1:25510/v2/hist/option/eod?root=AAPL&exp=20250620&strike=150000&right=C&start_date=20250115&end_date=20250115&use_csv=true"
```

Record HTTP status, `Content-Type`, body size, and first/last lines. Compare to the reproduced Python call with the same contract and window.

## Decision

- URL dead → environment (terminal / network).
- URL error or empty, wrapper agrees → vendor or bad contract encoding (strike × 1000, right, exp).
- URL has rows, wrapper empty/raises → **code structure** in `dbase` or `trade.datamanager` (mapping, date clip, cache, certification).
