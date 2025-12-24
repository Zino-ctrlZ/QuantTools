"""Position Analysis Package for Risk Management.

This package provides comprehensive position analysis infrastructure using a modular
cog-based architecture. Each cog analyzes positions from a specific perspective
(risk limits, signals, expiration, P&L) and generates action recommendations that
are reconciled into final strategy changes.

Core Components:
    PositionAnalyzer: Main orchestrator coordinating multiple cogs
    BaseCog: Abstract base class for all analysis cogs
    LimitsAndSizingCog: Greek limit enforcement and position sizing

Cog Architecture:
    The cog system provides:
    - Modular analysis components with clear responsibilities
    - Independent operation enabling parallel analysis
    - Priority-based action reconciliation
    - Easy addition/removal of analysis modules
    - Configurable enable/disable per cog

Available Cogs:
    Risk Limits (LimitsAndSizingCog):
        - Monitors Greek exposures (delta, gamma, vega, theta)
        - Enforces position size constraints
        - Recommends CLOSE/ADJUST on violations

    Signal Tracking (planned):
        - Monitors strategy entry/exit signals
        - Tracks signal strength changes
        - Recommends actions based on signal evolution

    Roll Management (planned):
        - Tracks DTE thresholds
        - Monitors moneyness drift
        - Recommends position rolls

    Expiration Handling (planned):
        - Manages positions approaching expiration
        - Handles ITM exercise decisions
        - Processes assignment scenarios

Usage:
    from EventDriven.riskmanager.position import PositionAnalyzer
    from EventDriven.riskmanager.position.cogs import LimitsAndSizingCog

    analyzer = PositionAnalyzer(
        cogs=[LimitsAndSizingCog(config=limits_config)],
        config=analyzer_config
    )

    actions = analyzer.analyze_position(
        position_data=position,
        market_data=market,
        analysis_date=date
    )

See Also:
    - analyzer.py: Main PositionAnalyzer implementation
    - base.py: BaseCog abstract class and core structures
    - cogs/: Individual cog implementations
"""
