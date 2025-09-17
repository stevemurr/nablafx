"""
Backward compatibility module for nablafx.system.

This module has been moved to nablafx.core submodules.
This file provides backward compatibility for existing imports.
"""

# Import all classes from the new locations
from nablafx.core.base_system import BaseSystem
from nablafx.core.blackbox_system import BlackBoxSystem, BlackBoxSystemWithTBPTT
from nablafx.core.greybox_system import GreyBoxSystem, GreyBoxSystemWithTBPTT

# Export all for backward compatibility
__all__ = ["BaseSystem", "BlackBoxSystem", "BlackBoxSystemWithTBPTT", "GreyBoxSystem", "GreyBoxSystemWithTBPTT"]
