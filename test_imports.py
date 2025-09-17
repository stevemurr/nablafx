#!/usr/bin/env python3

# Test the new modular system imports
import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.getcwd())

try:
    from nablafx.core import BaseSystem, BlackBoxSystem, GreyBoxSystem, BlackBoxSystemWithTBPTT, GreyBoxSystemWithTBPTT

    print("✅ All system imports working correctly!")

    # Test that we can access the classes
    print(f"✅ BaseSystem: {BaseSystem}")
    print(f"✅ BlackBoxSystem: {BlackBoxSystem}")
    print(f"✅ GreyBoxSystem: {GreyBoxSystem}")
    print(f"✅ BlackBoxSystemWithTBPTT: {BlackBoxSystemWithTBPTT}")
    print(f"✅ GreyBoxSystemWithTBPTT: {GreyBoxSystemWithTBPTT}")

    print("\n🎉 Modular system refactoring successful!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
