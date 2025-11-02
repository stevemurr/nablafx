# NablAFx Updates Roadmap

## 🚀 Implementation Progress

### ✅ Completed (Week 1)
- **WeightedMultiLoss System**: Implemented flexible loss function composition
  - ✅ `WeightedMultiLoss` class with validation and error handling
  - ✅ Updated `system.py` to handle new loss format with backward compatibility
  - ✅ Support for multiple loss functions with individual weights and names
  - ✅ Created test suite and configuration examples
  - 📍 **Location**: `nablafx/loss.py`, `tests-code/test_weighted_multi_loss.py`

### ✅ Completed (Week 1-2)  
- **Configuration Migration**: Updated all existing YAML configs to use new loss system
  - ✅ **101 configuration files migrated** from `TimeAndFrequencyDomainLoss` to `WeightedMultiLoss`
  - ✅ **Automated migration script** created (`scripts/migrate_loss_configs.py`)
  - ✅ **Safe migration process** with backup files for all changes
  - ✅ **Full verification** - no old loss configurations remain
  - 📍 **Location**: `cfg-new/` directory, `MIGRATION_SUMMARY.md`

### ✅ Completed (Week 2)
- **Callback System**: Extract logging functions from system.py into callback classes
  - ✅ Created modular callback architecture using PyTorch Lightning's callback system
  - ✅ **AudioLoggingCallback** - Configurable audio sample logging with W&B integration
  - ✅ **MetricsLoggingCallback** - Flexible metrics computation and logging system
  - ✅ **FrequencyResponseCallback** - Frequency response visualization at configurable intervals
  - ✅ **FADComputationCallback** - Multi-model FAD score computation (VGGish, PANN, CLAP, AFX-Rep)
  - ✅ **ParameterVisualizationCallback** - Gray-box model parameter and response visualization
  - ✅ **AudioChainLoggingCallback** - Audio logging at each processing block
  - ✅ **Configuration-driven setup** - Full YAML configuration support for all callbacks
  - ✅ **Backward compatibility** - Works with existing system classes without breaking changes
  - ✅ **Trainer configuration migration** - Updated all `cfg-new/trainer/` configs to use callbacks
  - 📍 **Location**: `nablafx/callbacks/`, `cfg-new/trainer/`, `examples/`

### ✅ Completed (Week 2-3)
- **Package Structure Refactoring**: Reorganized modules into logical subpackages
  - ✅ **Modular architecture** - Separated into `core/`, `processors/`, `controllers/`, `data/`, `evaluation/`, `callbacks/`, `utils/`
  - ✅ **Clear API boundaries** - Public APIs exported via `__init__.py` files
  - ✅ **Clean separation of concerns** - Core interfaces, models, and systems properly organized
  - ✅ **Version tracking** - Package version defined in `pyproject.toml`
  - 📍 **Location**: `nablafx/` directory structure

### ✅ Completed (Week 2-3)
- **Evaluation Registry System**: Flexible loss and metric composition
  - ✅ **EvaluationRegistry** - Registry-based function discovery system
  - ✅ **FlexibleLoss** - Dynamic loss composition from configuration
  - ✅ **FlexibleLossWithMetrics** - Combined loss and metrics computation
  - ✅ **Comprehensive evaluation functions** - Time-domain, frequency-domain, and audio-specific metrics
  - ✅ **Differentiability tracking** - Automatic validation of loss function compatibility
  - 📍 **Location**: `nablafx/evaluation/`

### ✅ Completed (Week 3)
- **Testing Infrastructure**: Comprehensive test suite
  - ✅ **Unit tests** - Processor tests for GCN, LSTM, S4, TCN with direct comparisons
  - ✅ **Integration tests** - Datamodules, datasets, and model tests
  - ✅ **Test organization** - Structured in `tests/unit/` and `tests/code/`
  - ✅ **Equivalence testing** - Gradient and training equivalence tests for processors
  - 📍 **Location**: `tests/`

### ⏳ Next Up
- **Type Hints Enhancement**: Add comprehensive type hints to all modules
- **Distribution & Packaging**: PyPI packaging, conda packages, Docker containers

---

## Immediate Priority: Loss Functions & Logging 🔥

### Why These Are Critical
The current hardcoded loss instantiation and embedded logging in system classes limit:

- **Experimental flexibility** - Researchers can't easily try different loss combinations
- **Code reusability** - Logging logic is tightly coupled to specific system classes
- **Configuration transparency** - Loss configurations are scattered across code and configs
- **Callback extensibility** - Adding new logging features requires modifying core classes

### Quick Implementation Plan (Weeks 1-2)

#### Week 1: Loss Function System
1. ✅ **~~Create `nablafx.loss.registry`~~** **→ Implemented `WeightedMultiLoss`** - Flexible loss composition system
2. ✅ **Implement `WeightedMultiLoss`** - Base class for combining multiple loss functions ✅ **COMPLETED**
3. ✅ **Add loss validation** - Check compatibility and parameter constraints ✅ **COMPLETED**
4. 🔄 **Update existing configurations** - Migrate current loss setups to new system **IN PROGRESS**

#### Week 2: Callback System  
1. **Extract logging methods** from system.py into callback classes
2. **Create base callback interface** - Common functionality and lifecycle hooks
3. **Update trainer configurations** - Enable callback specification in YAML
4. **Add callback examples** - Documentation and configuration templates

### Expected Impact
- **Faster experimentation** - New loss functions configurable without code changes
- **Cleaner architecture** - Separation of concerns between training logic and logging
- **Better reproducibility** - Complete experimental setup captured in configuration
- **Easier contribution** - Community can add callbacks without touching core training loop

---

## Streamlined Roadmap (Based on User Priorities)

## Overview
This document tracks the development of NablAFx framework improvements, focusing on practical enhancements that support research productivity.

## Current State Analysis

### Strengths
- ✅ Well-structured black-box and gray-box modeling framework
- ✅ Clean separation between processors, controllers, and models (modular package structure)
- ✅ Comprehensive callback system for flexible logging
- ✅ Registry-based evaluation system with flexible loss composition
- ✅ Good integration with PyTorch Lightning and Weights & Biases
- ✅ Extensive test coverage for processors and data modules
- ✅ Comprehensive examples and configurations

### Remaining Work
- Type hints coverage (partial implementation exists)
- Distribution and packaging for wider adoption

---

## Completed Work Summary

### 1. ✅ Package Structure & Organization
**Status**: COMPLETED  
**Timeline**: Week 2-3

- Modular structure with `core/`, `processors/`, `controllers/`, `data/`, `evaluation/`, `callbacks/`, `utils/`
- Clean separation of concerns with proper interfaces
- Clear API boundaries via `__init__.py` files
- Version tracking in `pyproject.toml`

### 2. ✅ Loss Function Configuration System
**Status**: COMPLETED  
**Timeline**: Week 1

- `FlexibleLoss` with registry-based composition
- `EvaluationRegistry` for function discovery
- Support for differentiable and non-differentiable functions
- Comprehensive evaluation functions (time-domain, frequency-domain, audio-specific)
- Full YAML configuration support

### 3. ✅ Callback-Based Logging System
**Status**: COMPLETED  
**Timeline**: Week 2

- 6 callback types: Audio, Metrics, Frequency Response, FAD, Parameter, Audio Chain
- Full PyTorch Lightning integration
- Configuration-driven setup via YAML
- Backward compatibility with existing systems

### 4. ✅ Testing Infrastructure
**Status**: COMPLETED  
**Timeline**: Week 3

- Unit tests for all processors (GCN, LSTM, S4, TCN)
- Direct comparison tests with reference implementations
- Gradient and training equivalence tests
- Integration tests for datamodules, datasets, and models

### 5. ✅ Configuration Migration
**Status**: COMPLETED  
**Timeline**: Week 1-2

- 101+ configuration files migrated to new loss system
- Automated migration scripts
- Full verification with backups

---

## Remaining Work

### Priority 1: Type Hints Enhancement
**Status**: IN PROGRESS (Partial implementation exists)  
**Estimated Time**: 1-2 weeks

#### Scope
- Add comprehensive type hints to all public methods and functions
- Focus on: processors, controllers, models, data modules, evaluation functions
- Use `typing` module for better IDE support and error detection

#### Benefits
- Better IDE autocomplete and error detection
- Easier refactoring and maintenance
- Catches bugs earlier in development

---

### Priority 2: Distribution & Packaging  
**Status**: PLANNED  
**Estimated Time**: 1-2 weeks

#### Scope

**2.1 PyPI Package**
- Complete `pyproject.toml` with full metadata and dependencies
- Add `setup.py` or modern `pyproject.toml`-only setup
- Create distribution scripts
- Test installation in clean environment
- Publish to PyPI (or TestPyPI first)

**2.2 Conda Package**
- Create `meta.yaml` for conda-forge
- Define conda dependencies (especially for CUDA/audio libs)
- Build and test conda package
- Submit to conda-forge (optional)

**2.3 Docker Containers**
- Create Dockerfile with all dependencies
- Include CUDA support for GPU training
- Add docker-compose for easy setup
- Document Docker usage

**2.4 Documentation**
- Installation instructions for all methods
- Dependency management guide
- Environment setup guide
- Quick start guide

#### Benefits
- Easy installation for collaborators and users
- Reproducible environments across systems
- Professional presentation for papers/releases
- Easier adoption by research community

---

## Items NOT Implemented (User Decision)

The following items from the original roadmap were discussed and decided against:

1. **CI/CD & Automated Testing** - Not needed for current research phase
2. **Pydantic Configuration Validation** - Current YAML validation is sufficient  
3. **Hierarchical Configuration System** - Current flat configs work well
4. **Processor Architecture Refactoring** - Current design is adequate
5. **Model Factory & Builder Patterns** - Not needed, direct instantiation works
6. **Data Pipeline Improvements** - Current pipeline is sufficient
7. **Performance Optimization** - Not a bottleneck currently
8. **Developer Experience Tools** - Current workflow is efficient
9. **Comprehensive API Documentation** - Code and configs are self-documenting
10. **Examples & Tutorials** - Configuration files serve as sufficient examples

These can be revisited in the future if needs change.

---

## Implementation Strategy

### Phase 1: Foundation (Weeks 1-4) - ✅ **COMPLETED**
1. ✅ **Package structure reorganization** - Modular architecture with clear API boundaries
2. ✅ **Testing infrastructure** - Unit and integration tests for processors, models, data
3. ✅ **Loss function configuration system** - Flexible loss composition via YAML
4. ✅ **Callback-based logging system** - Modular, configurable logging

### Phase 2: Remaining Items - 🎯 **SELECTED FOR IMPLEMENTATION**
1. ⏳ **Type hints enhancement** - Add comprehensive type hints to all modules
2. ⏳ **Distribution & packaging** - PyPI packaging, conda packages, Docker containers

### Phase 3: Items NOT Selected (Skipped by User Decision)
1. ❌ **CI/CD & Automated Testing** - GitHub Actions, coverage, pre-commit hooks (not needed for now)
2. ❌ **Configuration schema validation** - Pydantic validation (not needed)
3. ❌ **Unified configuration system** - Hierarchical configs with inheritance (not needed)
4. ❌ **Processor architecture refactoring** - Abstract base classes and plugin system (not needed)
5. ❌ **Model factory patterns** - Builder patterns for model construction (not needed)
6. ❌ **Data pipeline improvements** - Advanced data loading and transforms (not needed)
7. ❌ **Performance optimization** - Memory and computation improvements (not needed for now)
8. ❌ **Developer experience tools** - CLI improvements, debugging utilities (not needed)
9. ❌ **Comprehensive documentation** - API docs with Sphinx, ADRs (too complex)
10. ❌ **Examples & tutorials** - Jupyter notebooks, tutorials (configs are sufficient)

### 📊 Progress Summary
- **Completed**: 6/8 selected items (75%)
- **In Progress**: 0/8 selected items (0%)
- **Remaining**: 2/8 selected items (25%)
- **Current Focus**: Type hints enhancement, then distribution & packaging

---

## Success Metrics

### Code Quality
- ✅ Comprehensive test suite for core components
- ✅ Clean modular architecture with clear separation of concerns
- ✅ Configuration-driven development approach

### Developer Experience  
- ✅ Fast experimentation with flexible loss and callback configuration
- ✅ Clear examples through comprehensive configuration files
- ⏳ Type hints for better IDE support (in progress)

### Maintainability
- ✅ Loss function architecture with clear separation of concerns
- ✅ Modular callback system for extensibility
- ✅ Test suite with equivalence validation
- ⏳ Packaging for distribution (planned)

---

## Long-term Vision

The streamlined roadmap focuses NablAFx on becoming a practical, research-oriented framework that:

1. **Enables** rapid experimentation with configuration-driven development
2. **Maintains** scientific rigor with comprehensive testing
3. **Supports** reproducible research through proper packaging
4. **Balances** academic needs with practical software engineering

This focused approach ensures the framework serves its primary purpose: supporting high-quality audio effects modeling research without unnecessary complexity.
