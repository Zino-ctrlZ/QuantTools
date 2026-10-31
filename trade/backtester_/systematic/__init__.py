"""Systematic multi-asset books: many strategy sleeves trading one shared capital pool.

A book holds sleeves (a strategy class over a ticker universe), resolves each
child's params from layered config, and is asked before every fill inside
``Simulator``. Subclass ``SystematicMultiAsset`` and override its hooks to add
portfolio behavior such as caps, ranking, or regime filters.

Core Classes:
    SystematicMultiAsset: Inheritable book with default hooks.
    PortfolioBook: Abstract base ``Simulator`` uses to recognize a book.

Core Dataclasses:
    Sleeve: One strategy class over a ticker universe.
    BookSpec: Frozen recipe a book rebuilds children from.
    OpenCandidate, PortfolioContext, Veto: What hooks see and return.

Usage:
    >>> from trade.backtester_.systematic import Sleeve, SystematicMultiAsset
    >>> book = SystematicMultiAsset([Sleeve("mr", MeanRev)], data, max_open_trades=3)
    >>> result = book.simulate(capital=100_000, percent_of_capital=0.25)
"""

from trade.backtester_.simulator.core import OpenCandidate, BookHookError, PortfolioContext, Veto
from trade.backtester_.systematic.book import BookSpec, PortfolioBook, SystematicMultiAsset
from trade.backtester_.systematic.core import Sleeve, resolve_params, selector_matches
from trade.backtester_.systematic.errors import BookConfigError, ChildBuildError, ConfigIssue

__all__ = [
    "BookConfigError",
    "BookSpec",
    "ChildBuildError",
    "ConfigIssue",
    "OpenCandidate",
    "BookHookError",
    "PortfolioBook",
    "PortfolioContext",
    "Sleeve",
    "SystematicMultiAsset",
    "Veto",
    "resolve_params",
    "selector_matches",
]
