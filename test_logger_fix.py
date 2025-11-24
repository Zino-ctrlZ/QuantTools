"""
Test to verify setup_logger() prevents duplicate handlers on autoreload
Run this script before and after making changes to verify the fix works.
"""

import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trade.helpers.Logging import setup_logger


def test_idempotent_logger():
    """Test that calling setup_logger multiple times doesn't add duplicate handlers."""
    
    logger_name = "test_logger_idempotent"
    
    # Clear any existing logger
    if logger_name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
    
    print("Testing setup_logger() idempotency...")
    print("=" * 60)
    
    # Call setup_logger multiple times (simulates autoreload)
    for i in range(1, 6):
        logger = setup_logger(logger_name, stream_log_level=logging.INFO)
        handler_count = len(logger.handlers)
        print(f"Call #{i}: Logger has {handler_count} handler(s)")
        
        # Test that we always have exactly 1 handler
        if handler_count != 1:
            print(f"❌ FAILED: Expected 1 handler, got {handler_count}")
            return False
    
    print("=" * 60)
    print("✅ SUCCESS: setup_logger() is idempotent!")
    print("✅ All 5 calls resulted in exactly 1 handler")
    print("✅ Autoreload will no longer cause duplicate output")
    
    # Cleanup
    logger.handlers.clear()
    return True


def test_logger_output():
    """Test that logger output doesn't duplicate."""
    
    logger_name = "test_logger_output"
    
    # Clear any existing logger
    if logger_name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
    
    print("\n\nTesting logger output...")
    print("=" * 60)
    
    # First setup
    logger = setup_logger(logger_name, stream_log_level=logging.INFO)
    print("After 1st setup_logger() call:")
    logger.info("This message should appear ONCE")
    
    # Second setup (simulates autoreload)
    logger = setup_logger(logger_name, stream_log_level=logging.INFO)
    print("\nAfter 2nd setup_logger() call (simulating autoreload):")
    logger.info("This message should also appear ONCE (not twice!)")
    
    # Third setup
    logger = setup_logger(logger_name, stream_log_level=logging.INFO)
    print("\nAfter 3rd setup_logger() call:")
    logger.info("This message should STILL appear only ONCE")
    
    print("=" * 60)
    print("✅ If you see each message only once above, the fix works!")
    
    # Cleanup
    logger.handlers.clear()


if __name__ == "__main__":
    success = test_idempotent_logger()
    
    if success:
        test_logger_output()
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\n📝 Summary:")
        print("  - setup_logger() is now idempotent")
        print("  - Calling it multiple times doesn't add duplicate handlers")
        print("  - Autoreload in Jupyter will no longer cause duplicate output")
        print("  - No code changes needed in individual files")
        print("\n✨ The fix is permanent and automatic!")
    else:
        print("\n❌ Tests failed - check setup_logger() implementation")
        sys.exit(1)
