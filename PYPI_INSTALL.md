# PyPI Installation Quick Reference

## ⚠️ Important: Two-Step Installation Required

Due to the `rational-activations` dependency, NablaFX requires a specific installation order.

### Quick Install

```bash
# Step 1: Create environment and install rational-activations
python -m venv .venv
source .venv/bin/activate

# Step 2: Install with old PyTorch (required for rational-activations)
pip install torch==1.7.1 rational-activations==0.2.0

# Step 3: Install NablaFX (will upgrade PyTorch to 2.2.2)
pip install nablafx

# Step 4: Copy rational config file
python -c "
import shutil
from pathlib import Path
import rational
import sys

# Find rational package location
rational_dir = Path(rational.__file__).parent
config_path = rational_dir / 'rationals_config.json'

if config_path.exists():
    print('✅ Config already exists')
else:
    # Try to find from installation
    try:
        import pkg_resources
        weights_file = pkg_resources.resource_filename('nablafx', '../weights/rationals_config.json')
        shutil.copy(weights_file, config_path)
        print(f'✅ Copied config to {config_path}')
    except:
        print('⚠️  Please manually copy weights/rationals_config.json')
        print(f'   to: {config_path}')
"
```

### Verify Installation

```python
import nablafx
print(f"NablaFX {nablafx.__version__} installed successfully! ✅")
```

For detailed instructions, see [INSTALL.md](INSTALL.md)
