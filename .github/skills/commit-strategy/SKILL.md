# Commit Strategy Skill

Use this skill whenever the user asks to create commits, propose commits, or requests a commit plan.

## Objectives
- Separate code changes into the most relevant concerns.
- Produce a clear commit strategy before committing.
- Ensure each commit is cohesive, minimal, and reversible.

## Required Workflow
1. Inspect changed files and classify each change by concern.
2. Run a pre-commit quality scan on changed files for:
   - Potential bugs.
   - Failure in logic.
   - Misspellings.
   - Commented-out code blocks (do not flag plain explanatory comments).
3. Group files/hunks into concern-based commit buckets.
4. Present a commit strategy that lists:
   - What each commit includes.
   - Why the grouping is correct.
   - The commit message for each commit.
   - A concise scan summary of findings (or explicitly state no findings).
5. Keep unrelated changes out of the same commit.
6. Prefer multiple small focused commits over one mixed commit.

## Concern Grouping Rules
- Separate by behavior surface (feature, bug fix, refactor, docs, tests, config).
- Separate by subsystem when changes are independent.
- Keep tests with the behavior they verify.
- Keep formatting-only changes isolated when possible.
- Do not mix broad cleanup with functional changes.

## Commit Message Rules
- Use imperative mood.
- Keep the subject concise and specific.
- Explain intent, not implementation noise.
- Suggested style: `<area>: <action>`

## Output Template
When a commit is requested, provide:

0. Pre-Commit Scan
   - Potential bugs: <findings or "none">
   - Logic failures: <findings or "none">
   - Misspellings: <findings or "none">
   - Commented-out code blocks: <findings or "none">

1. Commit 1
   - Scope: <files/hunks>
   - Concern: <single concern>
   - Message: <commit message>

2. Commit 2
   - Scope: <files/hunks>
   - Concern: <single concern>
   - Message: <commit message>

## Example
1. Commit 1
   - Scope: `EventDriven/riskmanager/position/cogs/pnl_monitor.py`
   - Concern: Fix quantity math and logging edge cases.
   - Message: `riskmanager: fix pnl monitor quantity and logging bugs`

2. Commit 2
   - Scope: `EventDriven/riskmanager/position/cogs/pnl_monitor.py`, `EventDriven/riskmanager/position/cogs/test_pnl_monitor.py`
   - Concern: Add regression coverage for updated behavior.
   - Message: `tests: add pnl monitor regression cases`
