---
name: commit-strategy
description: >-
  Plans and executes git commits with concern-based grouping, pre-commit scan,
  and safe git workflow. Use when the user asks to commit, create commits, propose
  a commit plan, stage changes, or review changes for commit.
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

### 3. Group by Concern

- One concern per commit: feature, bugfix, refactor, docs, tests, config, tooling
- Split by subsystem when changes are independent
- Keep tests with the behavior they verify
- Isolate formatting, IDE settings, and tooling config when mixed with code
- Order commits: base/refactor first, then dependent changes
- No empty commits — if nothing to commit, say so and stop

### 4. Present Strategy (before executing)

Use this template:

**0. Pre-Commit Scan**
- Potential bugs: \<findings or "none"\>
- Logic failures: \<findings or "none"\>
- Misspellings: \<findings or "none"\>
- Commented-out code blocks: \<findings or "none"\>
- Excluded files: \<files left unstaged, with reason\>

**N. Commit N**
- Scope: \<files/hunks\>
- Concern: \<single concern\>
- Why: \<brief grouping rationale\>
- Message: \<commit message\>

### 5. Execute (only after explicit user approval)

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

## Commit Message Rules

- Imperative mood; concise subject (~50–72 chars)
- Format: `<area>: <action>`
- Explain intent, not implementation noise
- Match recent repo style from `git log` when unclear

## Git Safety

- Never update git config
- Never `--no-verify`, `--no-gpg-sign`, force push, or hard reset unless user explicitly asks
- Never force push to `main`/`master` — warn if requested
- Never push unless explicitly asked
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
