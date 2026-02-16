# Prompt Library

## Code Review Sweep (QuantTools + TFP-Algo + configs)

Use this prompt to run a recurring review for API mismatches, regressions, and logic/execution discrepancies.

```
You are my code-review copilot. Please review recent changes across these repos:
- QuantTools
- TFP-Algo
- configs

Goals:
1) Identify breaking API changes or call-site mismatches (especially around StrategyBase, TradeDecision, open_action/close_action, signal_id/side).
2) Check for behavioral regressions and missing validations.
3) Verify notebook outputs are cleared where expected.
4) Look for logic/execution discrepancies: places where the intended logic (docs/comments, docstrings, function names, or config flags) does not match what the code actually executes.
5) Bucket findings into Critical / High / Low, with file links and line references.

Please:
- Use file searches to find StrategyBase implementations and call sites.
- Cross-check docstrings and examples against actual behavior.
- List any gaps in tests or runtime checks.
- If no findings, say so explicitly and note residual risks.

Output format:
- Findings first (Critical -> High -> Low), each with file links and concise explanation.
- Open questions/assumptions.
- Brief change summary (only if needed).
```
