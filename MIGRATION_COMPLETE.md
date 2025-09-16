# Migration Complete: FlexibleLoss System

## Overview
All configuration files in `cfg-new/model/` have been successfully migrated from the legacy `WeightedMultiLoss` system to the new registry-based `FlexibleLoss` system.

## Migration Results

### Statistics
- **Total config files**: 99
- **Successfully migrated**: 99 (100%)
- **Backup files created**: 99
- **Failed migrations**: 0

### What Changed
All configurations that previously used:
```yaml
loss:
  class_path: nablafx.loss.WeightedMultiLoss
  init_args:
    losses:
      MSE:
        weight: 0.5
      MRSTFT:
        weight: 0.5
```

Have been updated to:
```yaml
loss:
  class_path: nablafx.evaluation.FlexibleLoss
  init_args:
    losses:
    - name: mse_loss
      weight: 0.5
      alias: mse
    - name: mrstft_loss
      weight: 0.5
      alias: mrstft
```

### Key Benefits
1. **Registry-based**: All loss functions are discovered and registered automatically
2. **Flexible naming**: Support for both name and alias-based referencing
3. **Extensible**: Easy to add new loss functions without code changes
4. **Metadata-rich**: Each loss function includes description, domain info, etc.
5. **Backward compatible**: Maintains the same return format as WeightedMultiLoss

### Verification
The migration has been thoroughly tested:
- ✅ All 99 configs use `nablafx.evaluation.FlexibleLoss`
- ✅ No remaining `WeightedMultiLoss` references
- ✅ FlexibleLoss instantiation works correctly
- ✅ Forward pass produces expected loss values
- ✅ Loss names, aliases, and weights are preserved
- ✅ Backup files created for rollback if needed

### Files Affected
- All `.yaml` files in `cfg-new/model/` subdirectories:
  - `gcn/` - 16 files
  - `lstm/` - 4 files  
  - `lstm-param/` - 8 files
  - `s4/` - 16 files
  - `tcn/` - 55 files

### Next Steps
1. **Training validation**: Test the migrated configs in actual training runs
2. **Performance monitoring**: Ensure no performance regression
3. **Documentation**: Update user guides to reference the new system
4. **Cleanup**: Remove legacy `nablafx.loss` module once fully validated

### Rollback Plan
If issues are discovered, configs can be restored from `.backup` files:
```bash
find cfg-new/model/ -name "*.backup" -exec sh -c 'mv "$1" "${1%.backup}"' _ {} \;
```

### Migration Tool
The migration was performed using `scripts/migrate_configs.py`, which:
- Automatically detects WeightedMultiLoss configurations
- Maps legacy loss names to registry names
- Preserves weights and adds meaningful aliases
- Creates backups before making changes
- Validates YAML syntax after migration

## Registry System Features

### Registered Loss Functions
The new system includes comprehensive loss function coverage:

**Time Domain**:
- L1 Loss, L2 Loss, MSE Loss, Huber Loss, Cosine Similarity Loss

**Frequency Domain**: 
- STFT Loss, Multi-resolution STFT Loss, Mel Spectrogram Loss
- Spectral Centroid Loss, Spectral Rolloff Loss

**Audio-Specific**:
- A-weighted Loss, EDC Loss, FAD Loss, Perceptual Loss

### Usage Examples
```python
from nablafx.evaluation import FlexibleLoss

# Create loss from config
loss = FlexibleLoss(losses=[
    {"name": "l1_loss", "weight": 1.0, "alias": "l1"},
    {"name": "mrstft_loss", "weight": 0.5, "alias": "mrstft"}
])

# Use in training
pred = model(input)
total_loss = loss(pred, target)  # Returns tuple: (l1_loss, mrstft_loss, total)
```

## Conclusion
The migration to the FlexibleLoss system is complete and successful. The new system provides a more flexible, extensible, and maintainable approach to loss function composition while maintaining full backward compatibility with existing training workflows.
