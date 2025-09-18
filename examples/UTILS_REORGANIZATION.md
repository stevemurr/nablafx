# Utils Module Reorganization

## Overview

The utility functions and classes have been reorganized into a proper `utils/` package for better modularity and organization.

## Changes Made

### File Structure
```
nablafx/
├── utils/                    # ✅ NEW: Utils package
│   ├── __init__.py          # Package initialization with exports
│   ├── utilities.py         # ✅ MOVED: from nablafx/utils.py
│   └── plotting.py          # ✅ MOVED: from nablafx/plotting.py
```

### What was moved:

1. **`nablafx/utils.py`** → **`nablafx/utils/utilities.py`**
   - Contains utility classes: `Rearrange`, `PTanh`

2. **`nablafx/plotting.py`** → **`nablafx/utils/plotting.py`** 
   - Contains all plotting functions for model visualization

## New Import Patterns

### ✅ **New (Recommended)**:
```python
# Import specific utilities
from nablafx.utils import Rearrange, PTanh

# Import specific plotting functions
from nablafx.utils import plot_gb_model, plot_frequency_response_steps

# Import entire utils module
import nablafx.utils as utils
```

### ✅ **Also works (direct module imports)**:
```python
# Import from specific submodules
from nablafx.utils.utilities import Rearrange, PTanh
from nablafx.utils.plotting import plot_gb_model
```

### ❌ **Old (No longer works)**:
```python
# These imports will now fail:
from nablafx.plotting import plot_gb_model        # ❌ Module removed
from nablafx.utils import Rearrange              # ❌ File moved
```

## Updated Files

The following files were updated to use the new import paths:

- `nablafx/core/base_system.py`
- `nablafx/core/greybox_system.py` 
- `nablafx/callbacks/frequency_response.py`
- `nablafx/callbacks/parameter_logging.py`

## Available Functions

### Utility Classes (`nablafx.utils.utilities`)
- `Rearrange`: Tensor reshape layer using einops
- `PTanh`: Parametric tanh activation function

### Plotting Functions (`nablafx.utils.plotting`)
- `fig2img`: Convert matplotlib figure to image
- `plot_static_params`: Plot static parameters
- `plot_phase_inv`: Plot phase inversion
- `plot_gain`: Plot gain parameters
- `plot_dc_offset`: Plot DC offset
- `plot_parametric_eq`: Plot parametric EQ response
- `plot_lowpass`: Plot lowpass filter response
- `plot_highpass`: Plot highpass filter response
- `plot_fir_filter`: Plot FIR filter response
- `plot_static_mlp_nonlinearity`: Plot MLP nonlinearity
- `plot_static_rational_nonlinearity`: Plot rational nonlinearity
- `plot_gb_model`: Plot gray-box model parameters
- `plot_frequency_response_steps`: Plot frequency response over steps
- `plot_frequency_response`: Plot frequency response

## Benefits

1. **Better Organization**: Related functions grouped together in a package
2. **Cleaner Namespace**: `nablafx.utils` is now a proper package
3. **Easier Maintenance**: Plotting and utility functions have clear separation
4. **Consistent Imports**: All utilities available from a single import location
5. **Future Extensibility**: Easy to add more utility modules (e.g., `nablafx.utils.audio`, `nablafx.utils.math`)

## Migration Guide

**If you have code using the old imports**, update them as follows:

```python
# OLD CODE:
from nablafx.plotting import plot_gb_model, plot_frequency_response_steps

# NEW CODE:
from nablafx.utils import plot_gb_model, plot_frequency_response_steps
```

```python  
# OLD CODE:
from nablafx.utils import Rearrange, PTanh

# NEW CODE: (same, but now imports from the package)
from nablafx.utils import Rearrange, PTanh
```
