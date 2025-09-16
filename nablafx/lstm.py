"""
DEPRECATED: This module has been moved to nablafx.processors.lstm

This file is maintained for backward compatibility.
The LSTM architecture is now in the nablafx.processors subpackage.
"""

# Import from new location for backward compatibility
from .processors.lstm import LSTM

__all__ = ["LSTM"]
