# Migration Guide: WeightedMultiLoss → FlexibleLoss

This document shows how to migrate from the old `WeightedMultiLoss` system to the new `FlexibleLoss` system using the evaluation function registry.

## Key Changes

### Old Format (WeightedMultiLoss)
```yaml
loss:
  class_path: nablafx.loss.WeightedMultiLoss
  init_args:
    losses:
      - loss:
          class_path: torch.nn.L1Loss
        weight: 0.5
        name: "l1"
      - loss:
          class_path: auraloss.freq.MultiResolutionSTFTLoss
          init_args:
            fft_sizes: [1024, 2048]
        weight: 0.5
        name: "mrstft"
```

### New Format (FlexibleLoss)
```yaml
loss:
  class_path: nablafx.evaluation.FlexibleLoss
  init_args:
    losses:
      - name: "l1_loss"
        weight: 0.5
        alias: "l1"
      - name: "mrstft_loss"
        weight: 0.5
        alias: "mrstft"
        params:
          fft_sizes: [1024, 2048]
          hop_sizes: [256, 512]
          win_lengths: [1024, 2048]
```

## Benefits of FlexibleLoss

1. **Cleaner Configuration**: No more nested `class_path` definitions
2. **Registry-Based**: Uses pre-registered, tested functions
3. **Audio-Specific Functions**: Access to guitar/instrument modeling losses
4. **Better Validation**: Automatic checking of function availability
5. **Consistent Interface**: All functions follow the same pattern

## Function Name Mapping

| Old Class Path | New Registry Name | Notes |
|---------------|-------------------|-------|
| `torch.nn.L1Loss` | `l1_loss` | Direct equivalent |
| `torch.nn.MSELoss` | `mse_loss` | Direct equivalent |
| `auraloss.time.ESRLoss` | `esr_loss` | Direct equivalent |
| `auraloss.time.DCLoss` | `dc_loss` | Direct equivalent |
| `auraloss.time.SISDRLoss` | `si_sdr_loss` | Direct equivalent |
| `auraloss.freq.STFTLoss` | `stft_loss` | Direct equivalent |
| `auraloss.freq.MultiResolutionSTFTLoss` | `mrstft_loss` | Direct equivalent |
| `auraloss.freq.MelSTFTLoss` | `melstft_loss` | Direct equivalent |
| `auraloss.freq.RandomResolutionSTFTLoss` | `random_stft_loss` | Direct equivalent |

## New Audio-Specific Functions

These are new functions specifically designed for guitar/instrument modeling:

| Registry Name | Description | Use Case |
|---------------|-------------|----------|
| `edc_loss` | Energy Decay Curve | Reverb, sustain, amp modeling |
| `spectral_centroid_loss` | Brightness/tonal character | EQ, distortion effects |
| `spectral_rolloff_loss` | High-frequency content | Filters, amp simulation |
| `a_weighting_loss` | Perceptual accuracy | Human-like evaluation |
| `zero_crossing_rate_metric` | Roughness/distortion | Distortion analysis |

## Migration Examples

### Simple Time + Frequency Loss
```yaml
# OLD
loss:
  class_path: nablafx.loss.WeightedMultiLoss
  init_args:
    losses:
      - loss:
          class_path: torch.nn.L1Loss
        weight: 1.0
        name: "l1"
      - loss:
          class_path: auraloss.freq.MultiResolutionSTFTLoss
        weight: 1.0
        name: "mrstft"

# NEW
loss:
  class_path: nablafx.evaluation.FlexibleLoss
  init_args:
    losses:
      - name: "l1_loss"
        weight: 1.0
        alias: "l1"
      - name: "mrstft_loss"
        weight: 1.0
        alias: "mrstft"
```

### Complex Multi-Loss Setup
```yaml
# OLD
loss:
  class_path: nablafx.loss.WeightedMultiLoss
  init_args:
    losses:
      - loss:
          class_path: auraloss.time.ESRLoss
        weight: 0.4
        name: "esr"
      - loss:
          class_path: auraloss.time.DCLoss
        weight: 0.2
        name: "dc"
      - loss:
          class_path: auraloss.freq.MultiResolutionSTFTLoss
          init_args:
            fft_sizes: [1024, 2048, 512]
        weight: 0.4
        name: "multi_stft"

# NEW
loss:
  class_path: nablafx.evaluation.FlexibleLoss
  init_args:
    losses:
      - name: "esr_loss"
        weight: 0.4
        alias: "esr"
      - name: "dc_loss"
        weight: 0.2
        alias: "dc"
      - name: "mrstft_loss"
        weight: 0.4
        alias: "multi_stft"
        params:
          fft_sizes: [1024, 2048, 512]
          hop_sizes: [256, 512, 128]
          win_lengths: [1024, 2048, 512]
```

### Guitar-Specific Enhancement
```yaml
# NEW: Enhanced with audio-specific functions
loss:
  class_path: nablafx.evaluation.FlexibleLoss
  init_args:
    losses:
      # Traditional losses
      - name: "esr_loss"
        weight: 0.3
        alias: "esr"
      - name: "mrstft_loss"
        weight: 0.3
        alias: "mrstft"
      
      # Guitar-specific enhancements
      - name: "edc_loss"
        weight: 0.2
        alias: "decay"
        params:
          remove_dc: true
          min_db: -80
      - name: "spectral_centroid_loss"
        weight: 0.1
        alias: "brightness"
        params:
          n_fft: 1024
      - name: "a_weighting_loss"
        weight: 0.1
        alias: "perceptual"
        params:
          sample_rate: 48000
```

## Automatic Migration Script

A migration script can be created to automatically convert old configs:

```python
# Example migration function
def migrate_config(old_config):
    """Convert WeightedMultiLoss config to FlexibleLoss config."""
    
    mapping = {
        "torch.nn.L1Loss": "l1_loss",
        "torch.nn.MSELoss": "mse_loss", 
        "auraloss.time.ESRLoss": "esr_loss",
        "auraloss.time.DCLoss": "dc_loss",
        "auraloss.freq.MultiResolutionSTFTLoss": "mrstft_loss",
        # ... etc
    }
    
    new_losses = []
    for loss_def in old_config["losses"]:
        class_path = loss_def["loss"]["class_path"]
        if class_path in mapping:
            new_loss = {
                "name": mapping[class_path],
                "weight": loss_def["weight"],
                "alias": loss_def["name"]
            }
            if "init_args" in loss_def["loss"]:
                new_loss["params"] = loss_def["loss"]["init_args"]
            new_losses.append(new_loss)
    
    return {
        "class_path": "nablafx.evaluation.FlexibleLoss",
        "init_args": {"losses": new_losses}
    }
```

## Testing Migration

To test your migrated configs:

```python
from nablafx.evaluation import FlexibleLoss
import yaml

# Load your config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Test FlexibleLoss creation
loss_fn = FlexibleLoss(config["loss"]["init_args"]["losses"])

# Test with dummy data
import torch
pred = torch.randn(2, 1, 1024)
target = torch.randn(2, 1, 1024)
result = loss_fn(pred, target)
print(f"Loss computation successful: {result}")
```

## Available Registry Functions

Run this to see all available functions:

```python
from nablafx.evaluation import list_available_losses, list_available_metrics

print("Available losses:", list_available_losses())
print("Available metrics:", list_available_metrics())
```
