#!/usr/bin/env python3
"""
Script to add type hints to data/datasets.py
"""
import re

def add_type_hints_to_datasets():
    file_path = "nablafx/data/datasets.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add imports
    if "from typing import" not in content:
        content = content.replace(
            "import numpy as np\n\nfrom natsort import natsorted",
            "import numpy as np\nfrom typing import List, Tuple, Optional, Dict, Any\n\nfrom natsort import natsorted"
        )
    
    # Pattern 1: __init__ methods
    patterns = [
        # PluginDataset.__init__
        (
            r'(class PluginDataset.*?\n.*?\n.*?\n.*?\n    def __init__\(\n        self,\n        root_dir_dry,\n        root_dir_wet,\n        data_to_use=1\.0,\n        sample_length=48000,\n        sample_rate=48000,\n        preload=False,\n    \):)',
            r'class PluginDataset(torch.utils.data.Dataset):\n    """\n    Dataset of pre-rendered audio examples from VST plugin\n    """\n\n    def __init__(\n        self,\n        root_dir_dry: str,\n        root_dir_wet: str,\n        data_to_use: float = 1.0,\n        sample_length: int = 48000,\n        sample_rate: int = 48000,\n        preload: bool = False,\n    ):'
        ),
        # __len__ and __getitem__
        (r'    def __len__\(self\):', r'    def __len__(self) -> int:'),
        (r'    def __getitem__\(self, idx\):', r'    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:'),
        # _load method
        (r'    def _load\(self, filepath, frame_offset=0, num_frames=-1\):', 
         r'    def _load(self, filepath: str, frame_offset: int = 0, num_frames: int = -1) -> Tuple[torch.Tensor, int]:'),
        # print method
        (r'    def print\(self\):', r'    def print(self) -> None:'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
    
    # ParametricPluginDataset patterns
    content = re.sub(
        r'(class ParametricPluginDataset.*?def __init__\(\n        self,\n        root_dir_dry,\n        root_dir_wet,\n        params_idxs_to_use,)',
        r'class ParametricPluginDataset(torch.utils.data.Dataset):\n    """\n    Dataset of pre-rendered audio examples from VST plugin with parameter conditioning\n    """\n\n    def __init__(\n        self,\n        root_dir_dry: str,\n        root_dir_wet: str,\n        params_idxs_to_use: Optional[List[int]],',
        content,
        flags=re.MULTILINE | re.DOTALL
    )
    
    # Fix remaining parameter types in ParametricPluginDataset.__init__
    content = re.sub(
        r'(params_idxs_to_use: Optional\[List\[int\]\],\n        data_to_use=1\.0,\n        sample_length=48000,\n        sample_rate=48000,\n        preload=False,)',
        r'params_idxs_to_use: Optional[List[int]],\n        data_to_use: float = 1.0,\n        sample_length: int = 48000,\n        sample_rate: int = 48000,\n        preload: bool = False,',
        content
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Added type hints to {file_path}")

if __name__ == "__main__":
    add_type_hints_to_datasets()
