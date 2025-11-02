#!/usr/bin/env python3
"""
Post-installation helper for NablaFX

This script helps users complete the NablaFX installation by:
1. Checking if rational-activations is properly installed
2. Helping copy the rationals_config.json to the correct location
"""

import os
import sys
import shutil
from pathlib import Path


def find_rational_package():
    """Find the rational package installation directory."""
    try:
        import rational

        return Path(rational.__file__).parent
    except ImportError:
        return None


def find_nablafx_weights():
    """Find the NablaFX weights directory."""
    try:
        import nablafx

        nablafx_dir = Path(nablafx.__file__).parent
        weights_dir = nablafx_dir / "weights"
        if weights_dir.exists():
            return weights_dir
        # Try parent directory (for development installs)
        weights_dir = nablafx_dir.parent / "weights"
        if weights_dir.exists():
            return weights_dir
        return None
    except ImportError:
        return None


def main():
    print("=" * 60)
    print("NablaFX Post-Installation Setup")
    print("=" * 60)
    print()

    # Check rational-activations
    rational_dir = find_rational_package()
    if not rational_dir:
        print("❌ rational-activations package not found!")
        print()
        print("Please install it first:")
        print("  pip install torch==1.7.1 rational-activations==0.2.0")
        print("  pip install --upgrade torch==2.2.2  # upgrade to required version")
        return 1

    print(f"✅ Found rational package at: {rational_dir}")

    # Check for rationals_config.json
    config_target = rational_dir / "rationals_config.json"

    if config_target.exists():
        print(f"✅ rationals_config.json already exists at: {config_target}")
        print()
        print("Setup complete! You're ready to use NablaFX. 🎉")
        return 0

    print(f"⚠️  rationals_config.json not found at: {config_target}")
    print()

    # Try to find the config file
    weights_dir = find_nablafx_weights()
    if not weights_dir:
        print("❌ Could not find NablaFX weights directory")
        print()
        print("Please manually copy rationals_config.json:")
        print(f"  cp <nablafx_install>/weights/rationals_config.json {config_target}")
        return 1

    config_source = weights_dir / "rationals_config.json"
    if not config_source.exists():
        print(f"❌ Source config file not found at: {config_source}")
        print()
        print("Please download rationals_config.json from:")
        print("  https://github.com/mcomunita/nablafx/blob/main/weights/rationals_config.json")
        print(f"and copy it to: {config_target}")
        return 1

    print(f"✅ Found source config at: {config_source}")
    print()

    # Attempt to copy
    try:
        print(f"Copying rationals_config.json to {config_target}...")
        shutil.copy2(config_source, config_target)
        print("✅ Successfully copied rationals_config.json!")
        print()
        print("Setup complete! You're ready to use NablaFX. 🎉")
        return 0
    except Exception as e:
        print(f"❌ Failed to copy: {e}")
        print()
        print("Please manually copy the file:")
        print(f"  cp {config_source} {config_target}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
