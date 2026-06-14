---
name: tfp-db
description: >-
  Guides MySQL access for the FinanceDatabase stack: multi-database layout,
  prod vs test environments, master_config.database_configs registry, base_name
  resolution, and safe read-only querying via MCP. Use when working with
  FinanceDatabase, dbase/database, portfolio_data, master_config, environment
  cloning, TFP-Algo database context, or MCP mysql_query against this server.
---

# TFP Database (FinanceDatabase MySQL)

## When to use this skill

Apply before any database exploration, SQL writing, or MCP `mysql_query` work tied to **FinanceDatabase** or **TFP-Algo**. Do not assume a single database or that schema names match Python `base_name` constants without checking the registry.

## Architecture in one paragraph

One MySQL **server** hosts many **databases** (MySQL schemas). Application code uses **base names** (e.g. `portfolio_data`). The **physical** schema name depends on **environment**: prod uses the base name; test/dev uses suffixed names registered in `master_config.database_configs`. Python resolves names via `get_database_name()` in `dbase/database/db_utils.py` after `set_environment_context()`. **MCP has no environment context** — you must use **physical** `database_name` values from the registry (or ask the user which environment).

## Logical databases (base names)

Constants in `dbase/database/db_utils.py` → class `Database`:

| Base name | Typical role |
|-----------|----------------|
| `portfolio_config` | Portfolio configuration |
| `portfolio_data` | Core portfolio / trades data |
| `strategy_trades_signals` | Strategy trades & signals |
| `portfolio_signals` | Portfolio signals |
| `vol_surface` | Volatility surface |
| `securities_master` | Securities reference data |
| `master_config` | **Registry** — never suffixed; always `master_config` |

**Excluded from cloning** (do not treat as app DBs): `information_schema`, `mysql`, `performance_schema`, `sys`. `master_config` is special (registry only).

## Environment model

### How Python picks an environment

Priority (`get_environment()` in `db_utils.py`):

1. CLI `--env` argument → environment string (often `test-{name}`)
2. Git branch: `main` → `prod`; any other branch → `test`
3. `ENVIRONMENT` env var
4. Default → `prod`

Runtime context: `set_environment_context(environment, branch_name)` (usually from TFP-Algo `runner.py`).

### How physical names are resolved

| Environment | Physical DB name |
|-------------|------------------|
| `prod` | Same as `base_name` (e.g. `portfolio_data`) |
| Non-prod (e.g. `test`, `test-mean-reversion`) | Row in `master_config.database_configs` for `(base_name, environment)` |

`master_config` is **never** suffixed.

Common test pattern: `{base_name}_{environment}` (e.g. `portfolio_data_test-mean-reversion`) — **always confirm** via the registry; do not guess suffixes.

## Source of truth: `master_config.database_configs`

**This table maps environments to physical databases — not tables.**

Known columns (from `db_management.py`):

| Column | Meaning |
|--------|---------|
| `database_name` | Physical MySQL schema name (use in SQL) |
| `base_name` | Logical name (`portfolio_data`, etc.) |
| `environment` | e.g. `prod`, `test`, `test-mean-reversion` |
| `branch_name` | Optional git branch metadata |
| `is_active` | `TRUE` = environment currently uses this DB; `FALSE` = soft-deleted |
| `created_by` | Audit (often `system`) |

### Starter queries (run first)

**1. All active environments and databases**

```sql
SELECT environment, base_name, database_name, branch_name, is_active
FROM master_config.database_configs
WHERE is_active = TRUE
ORDER BY environment, base_name;
```

**2. Resolve one base name for a specific environment**

```sql
SELECT database_name
FROM master_config.database_configs
WHERE base_name = 'portfolio_data'
  AND environment = 'prod'
  AND is_active = TRUE
LIMIT 1;
```

**3. List environments only**

```sql
SELECT DISTINCT environment
FROM master_config.database_configs
WHERE is_active = TRUE
ORDER BY environment;
```

**4. Tables inside a physical database** (after you know `database_name`)

```sql
SELECT table_name, table_type, table_rows
FROM information_schema.tables
WHERE table_schema = 'portfolio_data'
  AND table_type = 'BASE TABLE'
ORDER BY table_name;
```

Replace `'portfolio_data'` with the resolved physical name from step 1 or 2.

## SQL conventions for agents

1. **Always qualify** tables: `physical_database.table` (e.g. `portfolio_data.trades`).
2. **Confirm environment** with the user if unclear (`prod` vs a test env). Wrong env → wrong or empty data.
3. **Start from `master_config`** when exploring; never assume test DB names.
4. **Read-only by default** — MCP should use a read-only MySQL user; only `SELECT` / `SHOW` / `DESCRIBE` / safe `information_schema` queries unless the user explicitly enables writes elsewhere.
5. **LIMIT** large tables — use `information_schema.tables.table_rows` as a hint, then `SELECT ... LIMIT n`.
6. Python env vars (for reference, not MCP): `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, optional `ENVIRONMENT`, `DBASE_DIR`.

## MCP vs Python

| Python (`dbase/database`) | MCP (`mysql_query`) |
|---------------------------|---------------------|
| `get_database_name('portfolio_data')` | Query `master_config.database_configs` |
| `set_environment_context(...)` | Ask user or infer from task |
| `get_engine(db)` auto-resolves | Use physical `database_name` in SQL |
| Multi-DB via context | Omit `MYSQL_DB`; use qualified names |

## Environment management (context only)

Implemented in `dbase/database/db_management.py` (CLI: `python -m dbase.database.db_management`):

- **create** — clone prod (or source) schemas into a new test environment; registers rows in `database_configs`
- **list** — environments from registry
- **delete** — drops test DBs; soft-deletes registry rows; **`prod` cannot be deleted**
- **diff / sync** — compare environments; sync is **dry-run by default** (`--apply` to change)

Agents should **not** run destructive CLI unless the user explicitly requests it.

## Common mistakes to avoid

- Treating `database_configs` as a table list → it lists **databases**, not tables.
- Querying `portfolio_data` when the user is on a feature branch that uses `portfolio_data_test-...`.
- Forgetting `is_active = TRUE` (includes retired environments).
- Using unqualified `SELECT * FROM trades` in multi-DB mode.
- Assuming MCP write guards replace a read-only MySQL user (they do not).

## Workflow checklist

1. Run registry query (all active configs) or ask user for **environment**.
2. Map each needed `base_name` → `database_name`.
3. List tables per physical DB via `information_schema` if needed.
4. Run targeted `SELECT` with qualified names and `LIMIT`.
5. If zero rows / missing table, re-check environment and physical name before debugging schema.

## Code references

- Registry query: `dbase/database/db_utils.py` → `_load_database_name_from_config`
- Environment helpers: `dbase/database/db_management.py` → `get_databases_for_environment`, `get_tables_for_database`
- Docs: `dbase/database/README.md`
