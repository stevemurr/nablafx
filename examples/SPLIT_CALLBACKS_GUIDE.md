# Split Parameter Visualization Callbacks

## Overview

The original `ParameterVisualizationCallback` has been **removed** and replaced with two specialized callbacks for better modularity and control:

### 1. ParameterLoggingCallback
**File**: `nablafx/callbacks/parameter_logging.py`

**Purpose**: Logs frequency response plots and parameter visualizations for each block in gray-box models.

**Features**:
- Generates and logs parameter plots using `plot_gb_model`
- Logs frequency response visualizations
- Configurable logging at train start, validation, and test phases
- Controls number of test batches and samples per batch to log

**Configuration**:
```yaml
- class_path: nablafx.callbacks.ParameterLoggingCallback
  init_args:
    log_on_train_start: true      # Log at training start
    log_on_validation: true       # Log during validation
    log_on_test: true            # Log during testing
    log_test_batches: 10         # Number of test batches to log
    max_samples_per_batch: 5     # Max samples per batch to visualize
```

### 2. AudioChainLoggingCallback
**File**: `nablafx/callbacks/audio_chain_logging.py`

**Purpose**: Logs audio output at each processing block in gray-box models for debugging and analysis.

**Features**:
- Captures audio signal at each processing stage
- Logs input audio and output from each block
- Configurable sample rate for audio logging
- More conservative defaults (fewer batches/samples) to reduce logging overhead
- Clear naming convention for audio files

**Configuration**:
```yaml
- class_path: nablafx.callbacks.AudioChainLoggingCallback
  init_args:
    log_on_train_start: false    # Usually disabled for train start
    log_on_validation: true      # Log during validation
    log_on_test: true           # Log during testing
    log_test_batches: 5         # Number of test batches to log
    max_samples_per_batch: 3    # Max samples per batch to log
    sample_rate: 48000          # Sample rate for audio logging
```

## Benefits of the Split

### 1. **Better Performance**
- You can enable/disable parameter plotting and audio logging independently
- Audio logging is more resource-intensive, so you can control it separately
- Reduced memory usage when only one type of logging is needed

### 2. **More Granular Control**
- Different logging frequencies for plots vs audio
- Different sample counts for each type of logging
- Independent configuration of when each type of logging occurs

### 3. **Cleaner Code**
- Each callback has a single, focused responsibility
- Easier to maintain and debug
- More modular architecture

### 4. **Flexible Usage**
- Use both callbacks together for complete visualization
- Use only `ParameterLoggingCallback` for lightweight parameter monitoring
- Use only `AudioChainLoggingCallback` for audio analysis without plots

## Usage Examples

### Complete Gray-box Model Monitoring
```yaml
callbacks:
  - class_path: nablafx.callbacks.ParameterLoggingCallback
    init_args:
      log_on_train_start: true
      log_on_validation: true
      log_on_test: true
      log_test_batches: 10
      max_samples_per_batch: 5
  
  - class_path: nablafx.callbacks.AudioChainLoggingCallback
    init_args:
      log_on_train_start: false
      log_on_validation: true
      log_on_test: true
      log_test_batches: 5
      max_samples_per_batch: 3
      sample_rate: 48000
```

### Lightweight Parameter Monitoring Only
```yaml
callbacks:
  - class_path: nablafx.callbacks.ParameterLoggingCallback
    init_args:
      log_on_train_start: true
      log_on_validation: true
      log_on_test: false  # Skip test logging for faster testing
      max_samples_per_batch: 3
```

### Audio Analysis Only
```yaml
callbacks:
  - class_path: nablafx.callbacks.AudioChainLoggingCallback
    init_args:
      log_on_validation: false
      log_on_test: true
      log_test_batches: 10
      max_samples_per_batch: 5
      sample_rate: 48000
```

## Migration Required

**Breaking Change**: The original `ParameterVisualizationCallback` has been removed. You must update your configuration to use the new split callbacks.

**Old Configuration** (no longer works):
```yaml
- class_path: nablafx.callbacks.ParameterVisualizationCallback  # ❌ REMOVED
  init_args:
    log_on_train_start: true
    log_on_validation: true
    log_on_test: true
    log_test_batches: 10
    max_samples_per_batch: 5
```

**New Configuration** (required):
```yaml
- class_path: nablafx.callbacks.ParameterLoggingCallback  # ✅ NEW
  init_args:
    log_on_train_start: true
    log_on_validation: true
    log_on_test: true
    log_test_batches: 10
    max_samples_per_batch: 5

- class_path: nablafx.callbacks.AudioChainLoggingCallback  # ✅ NEW
  init_args:
    log_on_train_start: false  # Adjust as needed
    log_on_validation: true
    log_on_test: true
    log_test_batches: 5        # Adjust as needed
    max_samples_per_batch: 3   # Adjust as needed
    sample_rate: 48000
```
