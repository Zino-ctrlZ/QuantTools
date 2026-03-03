##TODO: DELETE FILE IF UNUSED##

"""
Strategy Base Classes for Backtesting Framework

This module provides abstract base classes that serve as standardized skeletons for
building trading strategies. These classes enforce consistent structure and ensure
all critical components are properly implemented.

AVAILABLE BASE CLASSES:
    1. StrategyBase - Clean base for custom strategies (RECOMMENDED)
    2. TrailingStrategyBase - Base for strategies using TrailingStrategy features

QUICK START:
    from trade.backtester_._sample import StrategyBase
    from datetime import datetime
    
    class MyStrategy(StrategyBase):
        _name = "AAPL"
        _live = False
        start_date = datetime(2024, 1, 1).date()
        
        # Implement required methods...
        def is_open_signal(self) -> bool: ...
        def is_close_signal(self) -> bool: ...
        # ... etc

See individual class docstrings for detailed documentation and usage examples.
"""

from backtesting.backtesting import Strategy
from backtesting.lib import TrailingStrategy
from abc import ABC, abstractmethod


class StrategyBase(Strategy, ABC):
    """
    Abstract base class providing a standardized skeleton for building custom trading strategies.
    
    This class serves as a foundation for all strategy implementations, enforcing consistent
    structure and ensuring critical components are properly defined. It inherits from 
    backtesting.Strategy to provide full access to the backtesting framework while adding
    abstract method enforcement for strategy-specific logic.
    
    DESIGN PHILOSOPHY:
        - Separation of concerns: Signal generation (is_open/close_signal) is decoupled 
          from position management (should_open/close)
        - Flexibility: Subclasses have complete control over entry/exit logic
        - Safety: Required attributes are validated at class definition time
        - Testability: Each component can be tested independently
    
    REQUIRED CLASS ATTRIBUTES:
        _name : str
            Strategy identifier, typically the ticker symbol (e.g., "AAPL", "SPY")
        _live : bool
            Execution mode flag. True for live trading, False for backtesting/paper trading
        start_date : datetime.date
            The earliest date this strategy should begin trading. Used to prevent
            trading during warmup/initialization periods
    
    REQUIRED METHODS TO IMPLEMENT:
        is_open_signal() -> bool
            Pure signal generation logic for entry conditions. Should return True when
            technical/fundamental conditions indicate a buy signal, False otherwise.
            
        is_close_signal() -> bool
            Pure signal generation logic for exit conditions. Should return True when
            conditions indicate position should be closed, False otherwise.
            
        pre_start_checks() -> bool
            Pre-execution validation logic. Typically used to enforce start_date 
            requirements or other preconditions. Return True to BLOCK trading,
            False to allow trading.
            
        have_position() -> bool
            Check if strategy currently holds an open position. Usually implements
            as `return bool(self.position)` or `return self.position is not None`.
            
        should_open() -> bool
            Position management logic for entries. Combines is_open_signal() with
            additional checks (e.g., not already in position, risk limits).
            
        should_close() -> bool
            Position management logic for exits. Combines is_close_signal() with
            stop loss, trailing stops, or other exit management logic.
            
        filter() -> bool
            Optional filtering logic to validate signals. Can include market regime
            filters, volatility checks, or other conditions that must be met
            before any trade signal is considered.
            
        stop_triggered() -> bool
            Stop loss evaluation logic. Return True if stop loss conditions are met.
            
        calculate_stop_loss() -> float
            Compute the stop loss price level based on strategy parameters, ATR,
            support/resistance levels, or other methods.
    
    USAGE EXAMPLE:
        class MyStrategy(StrategyBase):
            # Required attributes
            _name = "AAPL"
            _live = False
            start_date = datetime(2024, 1, 1).date()
            
            # Strategy parameters
            entry_threshold = 50
            stop_loss_pct = 0.02
            
            def init(self):
                # Initialize indicators
                self.sma = self.I(ta.sma, self.data.Close, 50)
            
            def next(self):
                # Main execution loop - full control over logic
                if self.should_open():
                    self.buy()
                elif self.should_close():
                    self.position.close()
            
            def is_open_signal(self) -> bool:
                return self.data.Close[-1] > self.sma[-1]
            
            def is_close_signal(self) -> bool:
                return self.data.Close[-1] < self.sma[-1]
            
            def pre_start_checks(self) -> bool:
                return self.data.index[-1].date() < self.start_date
            
            def have_position(self) -> bool:
                return bool(self.position)
            
            def should_open(self) -> bool:
                return self.is_open_signal() and not self.have_position()
            
            def should_close(self) -> bool:
                return self.is_close_signal() or self.stop_triggered()
            
            def filter(self) -> bool:
                return True  # No additional filters
            
            def stop_triggered(self) -> bool:
                if not self.have_position():
                    return False
                return self.data.Close[-1] < self.position.entry_price * (1 - self.stop_loss_pct)
            
            def calculate_stop_loss(self) -> float:
                return self.data.Close[-1] * (1 - self.stop_loss_pct)
    
    NOTES:
        - This base class does NOT call super().next(), giving you complete control
        - All backtesting.Strategy methods are available (buy, sell, position, data, etc.)
        - Validation happens at class definition time, catching errors before runtime
    """
    
    def __init_subclass__(cls, **kwargs):
        """Validate that subclasses define required attributes."""
        super().__init_subclass__(**kwargs)
        enforced_attributes = ['_name', '_live', 'start_date']
        for attr in enforced_attributes:
            if not hasattr(cls, attr):
                raise NotImplementedError(
                    f"Subclass '{cls.__name__}' must define class attribute: {attr}"
                )

    @abstractmethod
    def is_open_signal(self) -> bool:
        """
        Define the entry signal logic.
        
        Returns:
            bool: True if entry conditions are met, False otherwise
        """
        pass

    @abstractmethod
    def is_close_signal(self) -> bool:
        """
        Define the exit signal logic.
        
        Returns:
            bool: True if exit conditions are met, False otherwise
        """
        pass

    @abstractmethod
    def pre_start_checks(self) -> bool:
        """
        Perform any pre-start validation checks.
        Typically used to enforce start_date requirements.
        
        Returns:
            bool: True if checks fail (block trading), False if checks pass
        """
        pass

    @abstractmethod
    def have_position(self) -> bool:
        """
        Check if there is an open position.

        Returns:
            bool: True if there is an open position, False otherwise
        """
        pass

    @abstractmethod
    def should_open(self) -> bool:
        """
        Determine if the strategy should open a position.

        Returns:
            bool: True if conditions to open a position are met, False otherwise
        """
        pass

    @abstractmethod
    def should_close(self) -> bool:
        """
        Determine if the strategy should close a position.

        Returns:
            bool: True if conditions to close a position are met, False otherwise
        """
        pass

    @abstractmethod
    def filter(self) -> bool:
        """
        Additional filtering logic for trade signals.

        Returns:
            bool: True if filter conditions are met, False otherwise
        """
        pass

    @abstractmethod
    def stop_triggered(self) -> bool:
        """
        Check if stop loss conditions are triggered.

        Returns:
            bool: True if stop loss is triggered, False otherwise
        """
        pass

    @abstractmethod
    def calculate_stop_loss(self) -> float:
        """
        Calculate the stop loss level.

        Returns:
            float: The calculated stop loss price
        """
        pass


class TrailingStrategyBase(TrailingStrategy, ABC):
    """
    Abstract base class providing a standardized skeleton for building trailing stop strategies.
    
    This class extends backtesting.lib.TrailingStrategy while enforcing a consistent structure
    for strategy implementations. It provides access to TrailingStrategy's utility methods
    while adding abstract method enforcement for custom logic.
    
    ⚠️ CRITICAL WARNING:
        Do NOT call super().next() in your next() method implementation!
        TrailingStrategy.next() contains built-in trailing stop logic that will
        automatically close positions based on ITS rules, not yours. This causes
        unexpected "random" position closes that bypass your custom exit logic.
        
        ✅ SAFE:  super().init()  # Call in init() is fine
        ❌ AVOID: super().next()  # Will interfere with your logic
    
    DESIGN PHILOSOPHY:
        - Builds on TrailingStrategy's foundation while maintaining full control
        - Separation of concerns: Signal generation vs position management
        - Explicit over implicit: Your logic should be clearly defined
        - Safety: Required attributes validated at class definition time
    
    WHEN TO USE THIS VS StrategyBase:
        Use TrailingStrategyBase if:
            - You need TrailingStrategy's specific utility methods
            - You want access to trailing stop helper functions
            - Your strategy concept aligns with trailing stop patterns
        
        Use StrategyBase if:
            - You want a simpler, cleaner base with no parent logic concerns
            - You don't need TrailingStrategy-specific features
            - You want complete control without any parent class interference
        
        **Recommendation**: Start with StrategyBase unless you specifically need
        TrailingStrategy features. It's simpler and has fewer gotchas.
    
    REQUIRED CLASS ATTRIBUTES:
        _name : str
            Strategy identifier, typically the ticker symbol (e.g., "AAPL", "SPY")
        _live : bool
            Execution mode flag. True for live trading, False for backtesting/paper trading
        start_date : datetime.date
            The earliest date this strategy should begin trading. Used to prevent
            trading during warmup/initialization periods
    
    REQUIRED METHODS TO IMPLEMENT:
        is_open_signal() -> bool
            Pure signal generation logic for entry conditions. Should return True when
            technical/fundamental conditions indicate a buy signal, False otherwise.
            
        is_close_signal() -> bool
            Pure signal generation logic for exit conditions. Should return True when
            conditions indicate position should be closed, False otherwise.
            
        pre_start_checks() -> bool
            Pre-execution validation logic. Typically used to enforce start_date 
            requirements or other preconditions. Return True to BLOCK trading,
            False to allow trading.
            
        have_position() -> bool
            Check if strategy currently holds an open position. Usually implements
            as `return bool(self.position)` or `return self.position is not None`.
            
        should_open() -> bool
            Position management logic for entries. Combines is_open_signal() with
            additional checks (e.g., not already in position, risk limits).
            
        should_close() -> bool
            Position management logic for exits. Combines is_close_signal() with
            stop loss, trailing stops, or other exit management logic.
            
        filter() -> bool
            Optional filtering logic to validate signals. Can include market regime
            filters, volatility checks, or other conditions that must be met
            before any trade signal is considered.
            
        stop_triggered() -> bool
            Stop loss evaluation logic. Return True if stop loss conditions are met.
            
        calculate_stop_loss() -> float
            Compute the stop loss price level based on strategy parameters, ATR,
            support/resistance levels, or other methods.
    
    USAGE EXAMPLE:
        class MyTrailingStrategy(TrailingStrategyBase):
            # Required attributes
            _name = "MSFT"
            _live = False
            start_date = datetime(2024, 1, 1).date()
            
            # Strategy parameters
            atr_multiplier = 2.5
            trail_percent = 0.05
            
            def init(self):
                # ✅ SAFE: Calling super().init() is fine
                super().init()
                
                # Initialize your indicators
                self.atr = self.I(ta.atr, self.data.High, self.data.Low, 
                                  self.data.Close, 14)
                self.bbands = self.I(ta.bbands, self.data.Close, 20, 2)
            
            def next(self):
                # ❌ DO NOT call super().next() here!
                # Implement your complete logic
                
                if self.should_open():
                    self.buy()
                    self.stop = self.calculate_stop_loss()
                elif self.should_close():
                    self.position.close()
                    self.stop = None
            
            def is_open_signal(self) -> bool:
                # Entry signal: Price breaks above upper Bollinger Band
                return self.data.Close[-1] > self.bbands.upper[-1]
            
            def is_close_signal(self) -> bool:
                # Exit signal: Price crosses below middle band OR stop triggered
                return (self.data.Close[-1] < self.bbands.middle[-1] or 
                        self.stop_triggered())
            
            def pre_start_checks(self) -> bool:
                return self.data.index[-1].date() < self.start_date
            
            def have_position(self) -> bool:
                return bool(self.position)
            
            def should_open(self) -> bool:
                return self.is_open_signal() and not self.have_position()
            
            def should_close(self) -> bool:
                return self.is_close_signal() and self.have_position()
            
            def filter(self) -> bool:
                # Only trade when volatility is within acceptable range
                return 0.01 < self.atr[-1] / self.data.Close[-1] < 0.05
            
            def stop_triggered(self) -> bool:
                if self.stop is None or not self.have_position():
                    return False
                return self.data.Close[-1] < self.stop
            
            def calculate_stop_loss(self) -> float:
                # ATR-based stop loss
                return self.data.Close[-1] - (self.atr[-1] * self.atr_multiplier)
    
    COMMON PITFALLS:
        1. ❌ Calling super().next() - This is the #1 cause of mysterious position closes
        2. ❌ Not resetting self.stop to None after exit - Can cause issues on re-entry
        3. ❌ Checking stop_triggered() without verifying have_position() first
        4. ❌ Forgetting to call super().init() when you need parent initialization
    
    NOTES:
        - TrailingStrategy provides utility methods but its next() will interfere
        - All backtesting.Strategy methods available (buy, sell, position, data, etc.)
        - Validation happens at class definition time, catching errors before runtime
        - If you're not using TrailingStrategy-specific features, consider StrategyBase instead
    
    Required methods to implement:
        - is_open_signal() -> bool: Define entry conditions
        - is_close_signal() -> bool: Define exit conditions
        - pre_start_checks() -> bool: Pre-execution validation logic
    """
    
    def __init_subclass__(cls, **kwargs):
        """Validate that subclasses define required attributes."""
        super().__init_subclass__(**kwargs)
        enforced_attributes = ['_name', '_live', 'start_date']
        for attr in enforced_attributes:
            if not hasattr(cls, attr):
                raise NotImplementedError(
                    f"Subclass '{cls.__name__}' must define class attribute: {attr}"
                )

    @abstractmethod
    def is_open_signal(self) -> bool:
        """
        Define the entry signal logic.
        
        Returns:
            bool: True if entry conditions are met, False otherwise
        """
        pass

    @abstractmethod
    def is_close_signal(self) -> bool:
        """
        Define the exit signal logic.
        
        Returns:
            bool: True if exit conditions are met, False otherwise
        """
        pass

    @abstractmethod
    def pre_start_checks(self) -> bool:
        """
        Perform any pre-start validation checks.
        Typically used to enforce start_date requirements.
        
        Returns:
            bool: True if checks fail (block trading), False if checks pass
        """
        pass

    @abstractmethod
    def have_position(self) -> bool:
        """
        Check if there is an open position.
        
        Returns:
            bool: True if there is an open position, False otherwise
        """
        pass

    @abstractmethod
    def should_open(self) -> bool:
        """
        Determine if the strategy should open a position.
        
        Returns:
            bool: True if conditions to open a position are met, False otherwise
        """
        pass

    @abstractmethod
    def should_close(self) -> bool:
        """
        Determine if the strategy should close a position.
        
        Returns:
            bool: True if conditions to close a position are met, False otherwise
        """
        pass

    @abstractmethod
    def filter(self) -> bool:
        """
        Additional filtering logic for trade signals.
        
        Returns:
            bool: True if filter conditions are met, False otherwise
        """
        pass

    @abstractmethod
    def stop_triggered(self) -> bool:
        """
        Check if stop loss conditions are triggered.
        
        Returns:
            bool: True if stop loss is triggered, False otherwise
        """
        pass

    @abstractmethod
    def calculate_stop_loss(self) -> float:
        """
        Calculate the stop loss level.
        
        Returns:
            float: The calculated stop loss price
        """
        pass


# ============================================================================
# USAGE EXAMPLES
# ============================================================================
# 
# Option 1: Use StrategyBase (inherits from Strategy)
#   - Full control over all logic
#   - No built-in trailing stops
#   - Best for custom strategies
#
# Option 2: Use TrailingStrategyBase (inherits from TrailingStrategy)
#   - Has built-in trailing stop capability
#   - WARNING: Don't call super().next() unless you want parent's logic
#   - Use if you want TrailingStrategy's utility methods
# ============================================================================
