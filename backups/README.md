# Backup Files Directory

This directory contains backup files from the nablafx codebase refactoring and migration process.

## Directory Structure

### `nablafx-core/`

Contains backup files from the `nablafx.core` module:

- `system_backup.py` - Original monolithic system.py file before modular refactoring

### `nablafx-main/`

Contains backup files from the main `nablafx` module:

- `data_backup.py` - Backup of data module
- `data_old_backup.py` - Older backup of data module
- `modules_backup.py` - Backup of modules
- `processors_old_backup.py` - Backup of processors module

### `config-files/`

Contains backup configuration files (`.yaml.backup` files) from the `cfg-new/` directory.
These were created during various configuration updates and migrations.

## Purpose

These backup files are preserved for:

1. **Safety** - Reference in case of issues with new implementations
2. **History** - Documentation of code evolution
3. **Recovery** - Ability to restore previous functionality if needed

## Maintenance

These backup files can be safely removed once the new implementations have been thoroughly tested and validated in production use.

## Migration Timeline

- System refactoring: September 2025
- CLAP model restoration: September 2025
- Callback system migration: September 2025
