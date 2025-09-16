"""
DEPRECATED: This module has been moved to nablafx.processors.siren

This file is maintained for backward compatibility.
The SIREN networks are now in the nablafx.processors subpackage.
"""

# Import from new location for backward compatibility
from .processors.siren import Sine, Siren, SirenNet, Modulator

__all__ = ["Sine", "Siren", "SirenNet", "Modulator"]
