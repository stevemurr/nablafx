"""
Backward compatibility module for nablafx.interfaces.

This module has been moved to nablafx.core.interfaces.
This file provides backward compatibility for existing imports.
"""

# Import all classes from the new location
from nablafx.core.interfaces import *

# For explicit re-exports to ensure compatibility
from nablafx.core.interfaces import Processor, Controller
