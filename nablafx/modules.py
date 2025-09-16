"""
Backward Compatibility Module

This module provides backward compatibility for imports from the old modules.py file.
All components have been reorganized into:
- nablafx.processors.components: General-purpose components (MLP, FiLM, etc.)
- nablafx.processors.blocks: Architecture-specific blocks (TCNCondBlock, GCNCondBlock, etc.)

This file will be deprecated in future versions. Please update your imports to use
the new locations.
"""

# Import everything from the new locations for backward compatibility
from .processors.components import (
    center_crop,
    causal_crop,
    MLP,
    FiLM,
    TFiLM,
    TinyTFiLM,
    TVFiLMMod,
    TVFiLMCond,
)

from .processors.blocks import (
    TCNCondBlock,
    GCNCondBlock,
    S4CondBlock,
    DSSM,
)

# Re-export everything for backward compatibility
__all__ = [
    # Utility functions
    "center_crop",
    "causal_crop",
    # General components
    "MLP",
    "FiLM",
    "TFiLM",
    "TinyTFiLM",
    "TVFiLMMod",
    "TVFiLMCond",
    # Architecture-specific blocks
    "TCNCondBlock",
    "GCNCondBlock",
    "S4CondBlock",
    "DSSM",
]
