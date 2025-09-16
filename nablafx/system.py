"""
Backward compatibility module for nablafx.system.

This module has been moved to nablafx.core.system.
This file provides backward compatibility for existing imports.
"""

# Import all classes from the new location
from nablafx.core.system import *

# For explicit re-exports to ensure compatibility
from nablafx.core.system import BaseSystem, BlackBoxSystem, BlackBoxSystemWithTBPTT, GreyBoxSystem, GreyBoxSystemWithTBPTT
