#!/usr/bin/env python3

# Test script for the simple_flops_calculator
import sys

sys.path.append(".")

from benchmarks.flops.simple_flops_calculator import calculate_flops_from_yaml

if __name__ == "__main__":
    print("Testing the simple FLOPs calculator:")
    print("=" * 50)

    # Test TCN (black-box)
    print("\n1. Testing TCN (Black-box) model:")
    calculate_flops_from_yaml("cfg/model/tcn/model_bb_tcn-45-l-16.yaml")

    print("\n" + "=" * 50)

    # Test GreyBox model
    print("\n2. Testing GreyBox model:")
    calculate_flops_from_yaml("cfg/model/gb/gb_comp/model_gb_comp_peq.s+g.d+peq.s+g.s.yaml")

    print("\n" + "=" * 50)
    print("\nAll tests completed!")
