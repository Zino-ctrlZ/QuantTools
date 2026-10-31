"""Error types for SystematicMultiAsset that say where, what, and how to fix.

Config problems are collected and raised together, so one run shows every
mistake instead of one per attempt. A child that fails to build is wrapped
with its key and resolved params, keeping the original as ``__cause__``. A
book hook that returns the wrong type raises ``BookHookError`` naming the hook,
date, and key; exceptions raised inside a hook propagate unchanged.

Core Dataclasses:
    ConfigIssue: One configuration problem with its location and a hint.

Core Classes:
    BookConfigError: Every config problem found before anything is built.
    ChildBuildError: A child strategy's constructor or ``setup`` raised.
    BookHookError: Re-exported from the simulator, which raises it.

Core Functions:
    suggest: "Did you mean ..." hint from close string matches.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

## The engine raises it, so it lives in simulator.core; re-exported so book users import errors from one place.
from trade.backtester_.simulator.core import BookHookError  # noqa: F401


def suggest(name: str, valid: Iterable[str]) -> Optional[str]:
    """Return a "did you mean" hint for ``name``, or None when nothing is close.

    Examples:
        >>> suggest("windw", ["window", "z_entry"])
        "did you mean 'window'?"
    """
    matches = difflib.get_close_matches(name, list(valid), n=3, cutoff=0.6)
    return f"did you mean {', '.join(repr(m) for m in matches)}?" if matches else None


@dataclass(frozen=True)
class ConfigIssue:
    """One configuration problem.

    Attributes:
        where: Location in the config, e.g. ``"sleeve 'mr' asset_params['AAPL']"``.
        problem: What is wrong.
        hint: How to fix it, when there is an obvious fix.
    """

    where: str
    problem: str
    hint: Optional[str] = None

    def __str__(self) -> str:
        """Render as ``where: problem (hint)``."""
        return f"{self.where}: {self.problem}" + (f" ({self.hint})" if self.hint else "")


class BookConfigError(ValueError):
    """Every config problem found before anything is built."""

    def __init__(self, book: str, issues: List[ConfigIssue]) -> None:
        """Format all issues as a numbered list.

        Args:
            book: Book class name, for the header line.
            issues: Problems found, in discovery order.
        """
        self.book = book
        self.issues = list(issues)
        lines = [f"{book} config has {len(issues)} problem(s):"]
        lines += [f"  {i}. {issue}" for i, issue in enumerate(issues, 1)]
        super().__init__("\n".join(lines))


class ChildBuildError(RuntimeError):
    """A child strategy's constructor or ``setup`` raised."""

    def __init__(self, key: str, strategy_class: type, params: Dict[str, Any], cause: BaseException) -> None:
        """Name the child and show the exact params it was built with.

        Args:
            key: Strategy key ``"{sleeve}:{ticker}"``.
            strategy_class: Child class.
            params: Fully resolved params passed to the constructor.
            cause: The original exception.
        """
        self.key, self.params = key, params
        super().__init__(
            f"Building {strategy_class.__name__} for {key!r} failed: {type(cause).__name__}: {cause}\n"
            f"  resolved params: {params}"
        )
