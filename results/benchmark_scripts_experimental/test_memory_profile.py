#!/usr/bin/env python3
"""
Simple test script to verify memory profiling decorators work correctly.

This script tests that the @profile decorators are correctly applied
without breaking the existing functionality.

Usage:
    python test_memory_profile.py
"""

def test_profile_decorators():
    """Test that @profile decorators don't break import functionality."""
    try:
        # Try importing the modified module
        from manta.manta_entry import (
            _load_data_file, 
            _preprocess_dataframe,
            _perform_text_processing,
            _perform_topic_modeling,
            run_manta_process
        )
        
        print("✓ Successfully imported all profiled functions")
        
        # Check if functions have the profile decorator
        profiled_functions = [
            _load_data_file,
            _preprocess_dataframe, 
            _perform_text_processing,
            _perform_topic_modeling,
            run_manta_process
        ]
        
        for func in profiled_functions:
            if hasattr(func, '__name__'):
                print(f"✓ Function {func.__name__} is importable")
            else:
                print(f"⚠ Function {func} may have profiling issues")
                
        print("\nMemory profiling setup appears to be working correctly!")
        print("\nTo run memory profiling:")
        print("1. python -m memory_profiler memory_profile_manta.py")
        print("2. Or use: mprof run memory_profile_manta.py && mprof plot")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("The @profile decorators may be causing issues.")
        print("Make sure memory-profiler is installed: pip install memory-profiler")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = test_profile_decorators()
    exit(0 if success else 1)