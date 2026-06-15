---
name: commit-strategy
description: >-
  Plans and executes git commits with concern-based grouping, pre-commit scan,
  quality gate, and safe git workflow. Use when the user asks to commit, create
  commits, propose a commit plan, stage changes, or review changes for commit.
  Blocks commit and push when scan findings exist until the prompter advises next steps.
---

# Commit Strategy

## Project Context

Before planning or executing commits, read `.cursor/rules/coding-standards.mdc` for repo-specific coding standards and commit triggers. Match commit message areas and conventions to that rule and recent `git log` output.

## When to Use

- User explicitly asks to commit, stage, or propose commits.
- User asks for a commit plan or message review.
- **Do not** commit proactively. **Do not** activate for general git questions unless committing is involved.

## Objectives

- Separate changes into the most relevant concerns.
- Produce a clear commit strategy before committing.
- Ensure each commit is cohesive, minimal, and reversible.

## Required Workflow

### 1. Inspect (run in parallel)

From the repo root that has changes:

```bash
git status
git diff
git log -10 --oneline
```

If scoping a branch: `git diff main...HEAD` (or the repo default base).

Multi-root workspace: confirm which repo has changes before planning.

### 2. Pre-Commit Quality Scan

Scan changed files only:

- Potential bugs and logic failures
- Misspellings (strings, docs, logs)
- Commented-out code blocks (not plain explanatory comments)
- Debug artifacts (`print`, `pdb`, local paths)
- Secrets and credential files — warn and **exclude** from commit

Record every finding with file and line when possible. Use `"none"` only when a category has zero findings.

### 3. Quality Gate (mandatory — blocks commit and push)

After the scan, check whether **any** category has findings other than `"none"`.

**If findings exist:**

1. **Stop.** Do not stage, commit, or push.
2. Present the full pre-commit scan (step 5 template, section **0**).
3. Ask the prompter what to do next. Offer explicit options:
   - **Fix** — resolve findings, then re-run inspect + scan
   - **Exclude** — leave specific files/hunks unstaged; re-present strategy
   - **Commit anyway** — proceed with commits only; still hold push unless separately approved
   - **Push anyway** — commit and push with findings acknowledged (requires explicit override)
   - **Abort** — do nothing further

**Proceed to commit only when:**

- All scan categories are `"none"`, **or**
- The prompter explicitly resolves each finding category (fix, exclude, or commit anyway)

**Proceed to push only when:**

- All scan categories are `"none"`, **or**
- The prompter explicitly says to push despite findings (note deferred items in the final summary)

Never treat "commit and push" as implicit permission to skip the gate when findings exist.

### 4. Group by Concern

- One concern per commit: feature, bugfix, refactor, docs, tests, config, tooling
- Split by subsystem when changes are independent
- Keep tests with the behavior they verify
- Isolate formatting, IDE settings, and tooling config when mixed with code
- Order commits: base/refactor first, then dependent changes
- No empty commits — if nothing to commit, say so and stop

### 5. Present Strategy (before executing)

Use this template:

**0. Pre-Commit Scan**
- Potential bugs: \<findings or "none"\>
- Logic failures: \<findings or "none"\>
- Misspellings: \<findings or "none"\>
- Commented-out code blocks: \<findings or "none"\>
- Debug artifacts: \<findings or "none"\>
- Excluded files: \<files left unstaged, with reason\>

**Gate status:** \<BLOCKED — awaiting prompter \| CLEAR — no findings \| OVERRIDDEN — prompter approved proceed\>

**N. Commit N**
- Scope: \<files/hunks\>
- Concern: \<single concern\>
- Why: \<brief grouping rationale\>
- Message: \<commit message\>

If gate status is **BLOCKED**, stop after section **0** and the commit plan preview. Do not execute until the prompter responds.

### 6. Execute (only after gate is CLEAR or OVERRIDDEN)

For each planned commit, sequentially:

1. Stage only that commit's files
2. Commit via HEREDOC:

```bash
git commit -m "$(cat <<'EOF'
<area>: <imperative message>

EOF
)"
```

3. Run `git status` to verify
4. Repeat for remaining commits; final `git status` when done

### 7. Push (only after gate allows it)

- Push only when explicitly requested by the prompter.
- Re-check gate status before pushing: if findings were deferred and the prompter did not say "push anyway", **do not push**.
- After push, note any deferred findings still open in the working tree or last commits.

## Commit Message Rules

- Imperative mood; concise subject (~50–72 chars)
- Format: `<area>: <action>`
- Explain intent, not implementation noise
- Match recent repo style from `git log` when unclear

## Git Safety

- Never update git config
- Never `--no-verify`, `--no-gpg-sign`, force push, or hard reset unless user explicitly asks
- Never force push to `main`/`master` — warn if requested
- Never push unless explicitly asked **and** the quality gate allows it (see step 3)
- Never use interactive git (`-i` flags)
- **Amend only if all apply:** user requested amend (or hook auto-modified files after your successful commit); HEAD was created by you this session; commit not pushed
- If commit fails or hook rejects: fix and create a **new** commit — do not amend

## Examples

**Bug fix + regression (one commit):**
- Scope: `path/to/module.py`, `path/to/test_module.py`
- Message: `riskmanager: fix quantity math in pnl monitor`

**Code + IDE config (two commits):**
1. `datamanager: add forward timeseries caching`
2. `config: update vscode python settings`
