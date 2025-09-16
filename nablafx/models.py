"""
Backward compatibility module for nablafx.models.

This module has been moved to nablafx.core.models.
This file provides backward compatibility for existing imports.
"""

# Import all classes from the new location
from nablafx.core.models import *

# For explicit re-exports to ensure compatibility
from nablafx.core.models import BlackBoxModel, GreyBoxModel
