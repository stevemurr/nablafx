#!/usr/bin/env python3
"""
Config Migration Script

This script automatically migrates WeightedMultiLoss configurations
to the new FlexibleLoss format.
"""

import yaml
import argparse
import os
from pathlib import Path
from typing import Dict, Any, List


class ConfigMigrator:
    """Migrates WeightedMultiLoss configs to FlexibleLoss format."""

    # Mapping from old class paths to new registry names
    CLASS_PATH_MAPPING = {
        "torch.nn.L1Loss": "l1_loss",
        "torch.nn.MSELoss": "mse_loss",
        "torch.nn.SmoothL1Loss": "smooth_l1_loss",
        "torch.nn.HuberLoss": "huber_loss",
        "auraloss.time.ESRLoss": "esr_loss",
        "auraloss.time.DCLoss": "dc_loss",
        "auraloss.time.SISDRLoss": "si_sdr_loss",
        "auraloss.time.LogCoshLoss": "log_cosh_loss",
        "auraloss.freq.STFTLoss": "stft_loss",
        "auraloss.freq.MultiResolutionSTFTLoss": "mrstft_loss",
        "auraloss.freq.MelSTFTLoss": "melstft_loss",
        "auraloss.freq.RandomResolutionSTFTLoss": "random_stft_loss",
    }

    def migrate_loss_config(self, old_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate a single loss configuration from WeightedMultiLoss to FlexibleLoss.

        Args:
            old_config: Old WeightedMultiLoss configuration

        Returns:
            New FlexibleLoss configuration
        """
        if old_config.get("class_path") != "nablafx.loss.WeightedMultiLoss":
            return old_config  # Not a WeightedMultiLoss, return as-is

        old_losses = old_config.get("init_args", {}).get("losses", [])
        new_losses = []

        for loss_def in old_losses:
            if "loss" not in loss_def:
                continue

            class_path = loss_def["loss"].get("class_path", "")

            if class_path in self.CLASS_PATH_MAPPING:
                new_loss = {
                    "name": self.CLASS_PATH_MAPPING[class_path],
                    "weight": loss_def.get("weight", 1.0),
                    "alias": loss_def.get("name", self.CLASS_PATH_MAPPING[class_path]),
                }

                # Add parameters if they exist
                if "init_args" in loss_def["loss"]:
                    new_loss["params"] = loss_def["loss"]["init_args"]

                new_losses.append(new_loss)
            else:
                print(f"Warning: Unknown class path '{class_path}', skipping...")

        return {"class_path": "nablafx.evaluation.FlexibleLoss", "init_args": {"losses": new_losses}}

    def migrate_config_file(self, file_path: Path, backup: bool = True) -> bool:
        """
        Migrate a YAML config file.

        Args:
            file_path: Path to the config file
            backup: Whether to create a backup

        Returns:
            True if migration was performed, False otherwise
        """
        try:
            with open(file_path, "r") as f:
                config = yaml.safe_load(f)

            # Check if this config needs migration
            needs_migration = False

            # Check for WeightedMultiLoss in various locations
            if self._has_weighted_multi_loss(config):
                needs_migration = True

                if backup:
                    backup_path = file_path.with_suffix(file_path.suffix + ".backup")
                    with open(backup_path, "w") as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                    print(f"Created backup: {backup_path}")

                # Perform migration
                migrated_config = self._migrate_config_recursive(config)

                # Write migrated config
                with open(file_path, "w") as f:
                    yaml.dump(migrated_config, f, default_flow_style=False, sort_keys=False)

                print(f"✅ Migrated: {file_path}")
                return True
            else:
                print(f"⏭️  No migration needed: {file_path}")
                return False

        except Exception as e:
            print(f"❌ Error migrating {file_path}: {e}")
            return False

    def _has_weighted_multi_loss(self, config: Any) -> bool:
        """Check if config contains WeightedMultiLoss."""
        if isinstance(config, dict):
            for key, value in config.items():
                if key == "class_path" and value == "nablafx.loss.WeightedMultiLoss":
                    return True
                if self._has_weighted_multi_loss(value):
                    return True
        elif isinstance(config, list):
            for item in config:
                if self._has_weighted_multi_loss(item):
                    return True
        return False

    def _migrate_config_recursive(self, config: Any) -> Any:
        """Recursively migrate configuration."""
        if isinstance(config, dict):
            # Check if this is a loss configuration
            if config.get("class_path") == "nablafx.loss.WeightedMultiLoss":
                return self.migrate_loss_config(config)
            else:
                # Recursively process other dict entries
                return {key: self._migrate_config_recursive(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._migrate_config_recursive(item) for item in config]
        else:
            return config


def main():
    parser = argparse.ArgumentParser(description="Migrate WeightedMultiLoss configs to FlexibleLoss")
    parser.add_argument("path", help="Path to config file or directory")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backup files")
    parser.add_argument("--recursive", "-r", action="store_true", help="Process directories recursively")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without making changes")

    args = parser.parse_args()

    migrator = ConfigMigrator()
    path = Path(args.path)

    if path.is_file():
        if args.dry_run:
            print(f"Would migrate: {path}")
        else:
            migrator.migrate_config_file(path, backup=not args.no_backup)
    elif path.is_dir():
        pattern = "**/*.yaml" if args.recursive else "*.yaml"
        yaml_files = list(path.glob(pattern))

        print(f"Found {len(yaml_files)} YAML files")

        migrated_count = 0
        for yaml_file in yaml_files:
            if args.dry_run:
                # Quick check if file contains WeightedMultiLoss
                try:
                    with open(yaml_file, "r") as f:
                        content = f.read()
                    if "nablafx.loss.WeightedMultiLoss" in content:
                        print(f"Would migrate: {yaml_file}")
                        migrated_count += 1
                except:
                    pass
            else:
                if migrator.migrate_config_file(yaml_file, backup=not args.no_backup):
                    migrated_count += 1

        print(f"\\n{'Would migrate' if args.dry_run else 'Migrated'} {migrated_count} files")
    else:
        print(f"Error: {path} is not a valid file or directory")


if __name__ == "__main__":
    main()
