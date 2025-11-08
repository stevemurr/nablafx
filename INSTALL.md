# NablaFX Installation Guide

## Important: Rational Activations Installation

The `rational-activations` package has a metadata dependency on `torch==1.7.1`, but it works perfectly fine with PyTorch 2.x. We install it separately with `--no-deps` to bypass this outdated dependency declaration.

## Installation Steps

### Step 1: Install with PyPI (Recommended)

#### Option A: Using pip

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Step 1: Install NablaFX
pip install nablafx

# Step 2: Install rational-activations (ignoring its torch==1.7.1 requirement)
pip install rational-activations==0.2.0 --no-deps

# Step 3: Copy rational config file
python -c "
import shutil
from pathlib import Path
try:
    import rational
    import urllib.request
    
    rational_dir = Path(rational.__file__).parent
    config_target = rational_dir / 'rationals_config.json'
    
    # Always download and overwrite to ensure correct config
    # Download from GitHub
    url = 'https://raw.githubusercontent.com/mcomunita/nablafx/master/weights/rationals_config.json'
    urllib.request.urlretrieve(url, config_target)
    print(f'✅ Downloaded config to {config_target}')
except Exception as e:
    print(f'⚠️  Please manually download rationals_config.json')
    print(f'   from: https://github.com/mcomunita/nablafx/blob/master/weights/rationals_config.json')
    print(f'   Error: {e}')
"
```

#### Option B: Using conda

```bash
# Create and activate conda environment
conda create -n nablafx python=3.9
conda activate nablafx

# Step 1: Install NablaFX
pip install nablafx

# Step 2: Install rational-activations (ignoring its torch==1.7.1 requirement)
pip install rational-activations==0.2.0 --no-deps

# Step 3: Copy rational config file (same as above)
python -c "
import shutil
from pathlib import Path
try:
    import rational
    import urllib.request
    
    rational_dir = Path(rational.__file__).parent
    config_target = rational_dir / 'rationals_config.json'
    
    # Always download and overwrite to ensure correct config
    url = 'https://raw.githubusercontent.com/mcomunita/nablafx/master/weights/rationals_config.json'
    urllib.request.urlretrieve(url, config_target)
    print(f'✅ Downloaded config to {config_target}')
except Exception as e:
    print(f'⚠️  Please manually download rationals_config.json')
"
```

### Step 2: Install from Source (For Development)

```bash
# Clone the repository
git clone https://github.com/mcomunita/nablafx.git
cd nablafx

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Step 1: Install temporary dependencies
pip install -r requirements_temp-for-rnl.txt

# Step 2: Install full requirements
pip install -r requirements.txt

# Step 3: Install NablaFX in editable mode
pip install -e .

# Step 4: Copy rational config
cp weights/rationals_config.json \
   .venv/lib/python3.9/site-packages/rational/rationals_config.json
```

## Why This Two-Step Process?

The `rational-activations` package requires an older version of PyTorch (1.7.1) to install correctly, but NablaFX requires PyTorch 2.2.2 for full functionality. Installing in this order ensures:

1. The rational-activations package builds correctly with PyTorch 1.7.1
2. PyTorch is then upgraded to 2.2.2 for NablaFX
3. The pre-built rational-activations remains compatible

## Verification

After installation, verify everything works:

```python
import nablafx
print(f"NablaFX version: {nablafx.__version__}")

# Test imports
from nablafx.core import GreyBoxModel, BlackBoxModel
from nablafx.processors import Gain, ParametricEQ
from nablafx.controllers import DynamicController
print("✅ All imports successful!")
```

## Troubleshooting

### Issue: rational-activations fails to import

**Solution**: Ensure you copied the `rationals_config.json` file to the correct location in step 3.

### Issue: CUDA/GPU not available

**Solution**: Ensure you have the correct PyTorch version installed for your CUDA version. Visit [PyTorch website](https://pytorch.org/get-started/locally/) for CUDA-specific installation instructions.

### Issue: Import errors

**Solution**: Make sure you followed the two-step installation process. Try reinstalling:
```bash
pip uninstall nablafx rational-activations torch
# Then repeat the installation steps
```

## Next Steps

After installation, check out:
- [Quick Start Guide](README.md#quick-start)
- [Configuration Examples](cfg/)
- [Documentation](README.md)
