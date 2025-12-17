"""
Console utilities for MANTA topic modeling.

This module provides enhanced console output functionality including:
- Rich formatting and progress bars
- Structured configuration display
- Professional status messages
- Analysis summaries
"""

from .console_manager import ConsoleManager, get_console, set_console

# Singleton instance for convenient access
console = get_console()

__all__ = ["ConsoleManager", "get_console", "set_console", "console"]