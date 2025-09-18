# Benchmarks

This directory contains performance benchmarks and model analysis scripts for nablafx.

## Structure

- **`flops/`** - FLOPS, MACs, and parameter counting scripts
- **`speed/`** - CPU and GPU speed benchmarking scripts  
- **`analysis/`** - Model analysis and parameter counting utilities

## Usage

### FLOPS/MACs Analysis
```bash
python benchmarks/flops/flops-macs-params_tcn.py
python benchmarks/flops/flops-macs-params_lstm.py
```

### Speed Benchmarking
```bash
python benchmarks/speed/speed_cpu_tcn.py
python benchmarks/speed/speed_cpu_lstm.py

# Run with priority (if available)
./benchmarks/speed/speed_run_with_priority.sh
```

### Parameter Analysis
```bash
python benchmarks/analysis/nparams_tcn.py
python benchmarks/analysis/nparams_lstm.py
```

## Notes

- Speed benchmarks are CPU-focused unless otherwise specified
- FLOPS calculations use the `calflops` library
- Scripts are designed to be run independently
- Some benchmarks may require specific hardware or datasets

## Adding New Benchmarks

When adding new benchmark scripts:
1. Place FLOPS/computation analysis in `flops/`
2. Place timing/speed tests in `speed/`
3. Place parameter counting/model analysis in `analysis/`
4. Follow existing naming conventions
5. Include proper documentation and comments
