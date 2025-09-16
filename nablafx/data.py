"""
Backward Compatibility Module for Data

This module provides backward compatibility for imports from the old data.py file.
All data-related classes have been reorganized into:
- nablafx.data.datasets: Dataset classes (PluginDataset, ParametricPluginDataset)
- nablafx.data.datamodules: DataModule classes (DryWetFilesPluginDataModule)

This file will be deprecated in future versions. Please update your imports to use
the new locations.
"""

# Import everything from the new locations for backward compatibility
from .data.datamodules import (
    DryWetFilesPluginDataModule,
)

from .data.datasets import (
    PluginDataset,
    ParametricPluginDataset,
)

# Re-export everything for backward compatibility
__all__ = [
    "DryWetFilesPluginDataModule",
    "PluginDataset",
    "ParametricPluginDataset",
]
