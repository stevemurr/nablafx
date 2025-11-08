# NablaFX Changelog

## Version 1.0.0 (November 2025)

### Major Features

#### 1. Modular Package Structure
- Reorganized into logical subpackages: `core/`, `processors/`, `controllers/`, `data/`, `evaluation/`, `callbacks/`, `utils/`
- Clear API boundaries with well-defined public interfaces
- Clean separation between black-box and gray-box modeling systems

#### 2. Flexible Loss Composition System
- **FlexibleLoss**: Dynamic loss composition from YAML configuration
- **EvaluationRegistry**: Registry-based function discovery for losses and metrics
- Support for differentiable and non-differentiable evaluation functions
- Comprehensive evaluation functions: time-domain, frequency-domain, and audio-specific metrics

#### 3. Callback-Based Logging System
- Modular PyTorch Lightning callback architecture
- **AudioLoggingCallback**: Configurable audio sample logging with W&B integration
- **MetricsLoggingCallback**: Flexible metrics computation and logging
- **FrequencyResponseCallback**: Frequency response visualization
- **FADComputationCallback**: Multi-model FAD score computation (VGGish, PANN, CLAP, AFX-Rep)
- **ParameterVisualizationCallback**: Gray-box model parameter and response visualization
- **AudioChainLoggingCallback**: Audio logging at each processing block
- Full YAML configuration support

#### 4. Comprehensive Type Hints
- Complete type annotations across all modules
- Support for modern Python type checking with mypy
- Better IDE support and error detection
- PEP 561 compliant with `py.typed` marker

#### 5. PyPI Distribution Package
- Available on PyPI: `pip install nablafx`
- Comprehensive installation documentation

#### 6. Testing Infrastructure
- Unit tests for all processor architectures (GCN, LSTM, S4, TCN)
- Direct comparison tests with reference implementations
- Gradient and training equivalence validation
- Integration tests for datamodules and datasets

### Configuration Changes

- **101+ configuration files** migrated to new FlexibleLoss system
- All trainer configs updated to use callback system

### Dependencies

- Python >= 3.9
- PyTorch >= 2.2.2
- PyTorch Lightning >= 2.3.2
- See `pyproject.toml` for full dependency list

### Documentation

- [README.md](README.md) - Overview and quick start
- [INSTALL.md](INSTALL.md) - Detailed installation guide
- Configuration examples in `cfg/` directory

### Credits

See [README.md](README.md#credits) for acknowledgments of the architectures and libraries used.